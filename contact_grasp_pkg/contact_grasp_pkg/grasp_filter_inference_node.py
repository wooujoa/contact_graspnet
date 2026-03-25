#!/usr/bin/env python3
import os
import sys
import time
import importlib.util
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R, Slerp

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BASE_DIR = os.environ.get("CGN_BASE_DIR", "/opt/contact_graspnet")
CGN_ROOT = os.path.join(BASE_DIR, "contact_graspnet")

CGN_CONFIG_UTILS = os.path.join(CGN_ROOT, "config_utils.py")
CGN_ESTIMATOR = os.path.join(CGN_ROOT, "contact_grasp_estimator.py")
CGN_MODEL = os.path.join(CGN_ROOT, "contact_graspnet.py")

CGN_POINTNET_UTILS = os.path.join(BASE_DIR, "pointnet2", "utils")
CGN_POINTNET_MODELS = os.path.join(BASE_DIR, "pointnet2", "models")
CGN_POINTNET_SAMPLING = os.path.join(BASE_DIR, "pointnet2", "tf_ops", "sampling")
CGN_POINTNET_GROUPING = os.path.join(BASE_DIR, "pointnet2", "tf_ops", "grouping")
CGN_POINTNET_INTERP = os.path.join(BASE_DIR, "pointnet2", "tf_ops", "3d_interpolation")


def _append_path(p: str):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)


def _load_module_from_file(module_name: str, file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Required module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for p in [
    CGN_POINTNET_UTILS,
    CGN_POINTNET_MODELS,
    CGN_POINTNET_SAMPLING,
    CGN_POINTNET_GROUPING,
    CGN_POINTNET_INTERP,
    CGN_ROOT,
]:
    _append_path(p)

config_utils = _load_module_from_file("config_utils", CGN_CONFIG_UTILS)
contact_graspnet_model = _load_module_from_file("contact_graspnet", CGN_MODEL)
contact_grasp_estimator_mod = _load_module_from_file("contact_grasp_estimator", CGN_ESTIMATOR)

if not hasattr(contact_graspnet_model, "placeholder_inputs"):
    raise ImportError(
        f"Loaded wrong contact_graspnet module: {getattr(contact_graspnet_model, '__file__', None)}"
    )

GraspEstimator = contact_grasp_estimator_mod.GraspEstimator


class ContactGraspRoiFilterNode(Node):
    def __init__(self):
        super().__init__("contact_grasp_node")

        # =========================
        # Topics
        # =========================
        self.pc_sub_topic = "/yolo/target_pc"
        self.obj_pc_sub_topic = "/yolo/object_pc"
        self.bg_pc_sub_topic = "/yolo/background_pc"

        self.pose_raw_pub_topic = "/grasp/best_pose_raw"
        self.pose_vis_pub_topic = "/grasp/best_pose_vis"
        self.marker_pub_topic = "/grasp/markers"
        self.best_marker_pub_topic = "/grasp/best_pose_marker"
        self.best_contact_pub_topic = "/grasp/best_contact_marker"
        self.best_contact_point_pub_topic = "/grasp/best_contact_point"

        # =========================
        # CGN Config
        # =========================
        self.ckpt_dir = os.path.join(
            BASE_DIR, "checkpoints/scene_test_2048_bs3_hor_sigma_001"
        )

        self.max_input_points = 8000
        self.min_input_points = 80

        self.max_gripper_width = 0.14
        self.min_gripper_width = 0.005

        # confidence threshold
        self.primary_conf_thresh = 0.23
        self.fallback_conf_thresh = 0.19
        self.min_confident_grasps = 8

        # FPS / representative candidates
        self.enable_fps = True
        self.max_fps_candidates = 8

        # contact/object/background filtering
        self.max_contact_to_object_dist = 0.03
        self.min_contact_bg_margin = 0.005

        # collision proxy around grasp
        self.enable_bg_collision_check = True
        self.approach_collision_radius = 0.03
        self.approach_collision_length = 0.08
        self.palm_collision_radius = 0.04

        # optional vertical filter
        self.enable_vertical_filter = False
        self.vertical_axis_in_camera = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.vertical_axis_in_camera /= np.linalg.norm(self.vertical_axis_in_camera)
        self.max_vertical_alignment = 0.95

        # visualization fix
        self.R_fix = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # temporal stabilization
        self.enable_tracking = True
        self.track_max_pos_dist = 0.05
        self.track_max_angle_deg = 25.0
        self.track_score_bonus = 0.08

        self.enable_pose_ema = True
        self.ema_alpha_pos = 0.75
        self.ema_alpha_rot = 0.75

        self.enable_hold_last = True
        self.max_hold_frames = 2
        self.lost_frame_count = 0

        # candidate confirmation
        self.enable_confirmation = True
        self.confirm_pos_dist = 0.035
        self.confirm_angle_deg = 20.0
        self.confirm_required_frames = 2
        self.pending_candidate = None
        self.pending_count = 0

        self.topk_debug = 10
        self.marker_topk = 5

        self.best_arrow_length = 0.06
        self.best_arrow_shaft_diameter = 0.010
        self.best_arrow_head_diameter = 0.015

        self.candidate_arrow_length = 0.045
        self.candidate_arrow_shaft_diameter = 0.006
        self.candidate_arrow_head_diameter = 0.010

        self.best_contact_scale = 0.012
        self.candidate_contact_scale = 0.008

        # temporal state
        self.prev_best_raw = None
        self.prev_best_vis = None
        self.prev_best_contact = None
        self.prev_best_score = None
        self.prev_best_width = None
        self.prev_best_header = None

        self.smoothed_raw = None
        self.smoothed_vis = None

        # =========================
        # ROS IO
        # =========================
        self.sub_pc = self.create_subscription(PointCloud2, self.pc_sub_topic, self.pc_cb, 10)
        self.sub_obj_pc = self.create_subscription(PointCloud2, self.obj_pc_sub_topic, self.obj_pc_cb, 10)
        self.sub_bg_pc = self.create_subscription(PointCloud2, self.bg_pc_sub_topic, self.bg_pc_cb, 10)

        self.pub_pose_raw = self.create_publisher(PoseStamped, self.pose_raw_pub_topic, 10)
        self.pub_pose_vis = self.create_publisher(PoseStamped, self.pose_vis_pub_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_pub_topic, 10)
        self.pub_best_marker = self.create_publisher(Marker, self.best_marker_pub_topic, 10)
        self.pub_best_contact = self.create_publisher(Marker, self.best_contact_pub_topic, 10)
        self.pub_best_contact_point = self.create_publisher(PointStamped, self.best_contact_point_pub_topic, 10)

        self.latest_obj_pc = None
        self.latest_bg_pc = None
        self.latest_obj_stamp = None
        self.latest_bg_stamp = None

        # =========================
        # Initialize CGN
        # =========================
        self.get_logger().info("Loading Contact-GraspNet to GPU...")

        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        global_config = config_utils.load_config(self.ckpt_dir, batch_size=1)
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        saver = tf.train.Saver(save_relative_paths=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        self.grasp_estimator.load_weights(self.sess, saver, self.ckpt_dir, mode="test")
        self.get_logger().info("Contact-GraspNet Initialized! Waiting for PointCloud...")

    # -------------------------
    # Point cloud helpers
    # -------------------------
    def read_xyz_from_cloud(self, msg: PointCloud2):
        try:
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if len(pts) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            arr = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
            return arr
        except Exception as e:
            self.get_logger().error(f"read_xyz_from_cloud failed: {e}")
            return np.zeros((0, 3), dtype=np.float32)

    def obj_pc_cb(self, msg: PointCloud2):
        self.latest_obj_pc = self.read_xyz_from_cloud(msg)
        self.latest_obj_stamp = msg.header.stamp

    def bg_pc_cb(self, msg: PointCloud2):
        self.latest_bg_pc = self.read_xyz_from_cloud(msg)
        self.latest_bg_stamp = msg.header.stamp

    def deterministic_downsample(self, pc_np: np.ndarray, max_points: int) -> np.ndarray:
        n = pc_np.shape[0]
        if n <= max_points:
            return pc_np
        idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
        return pc_np[idx]

    def normalize_candidate_arrays(self, grasps, scores, widths):
        grasps = np.asarray(grasps)
        scores = np.asarray(scores)
        widths = np.asarray(widths)

        if grasps.ndim == 2 and grasps.shape == (4, 4):
            grasps = grasps[np.newaxis, ...]
        elif grasps.ndim == 0:
            raise ValueError(f"Invalid grasps shape: {grasps.shape}")

        scores = np.atleast_1d(scores).astype(np.float32)
        widths = np.atleast_1d(widths).astype(np.float32)

        n = grasps.shape[0]

        if scores.shape[0] != n:
            if scores.shape[0] == 1:
                scores = np.repeat(scores, n)
            else:
                raise ValueError(f"Scores shape mismatch: grasps={grasps.shape}, scores={scores.shape}")

        if widths.shape[0] != n:
            if widths.shape[0] == 1:
                widths = np.repeat(widths, n)
            else:
                raise ValueError(f"Widths shape mismatch: grasps={grasps.shape}, widths={widths.shape}")

        return grasps, scores, widths

    def normalize_contact_points(self, contact_pts, n_grasps: int):
        pts = np.asarray(contact_pts, dtype=np.float32)

        if pts.ndim == 1 and pts.shape[0] == 3:
            pts = pts[np.newaxis, :]
        elif pts.ndim == 0:
            raise ValueError("Invalid contact_pts: scalar")
        elif pts.ndim > 2:
            pts = pts.reshape(-1, 3)

        if pts.shape[-1] != 3:
            raise ValueError(f"Invalid contact_pts shape: {pts.shape}")

        if pts.shape[0] != n_grasps:
            if pts.shape[0] == 1:
                pts = np.repeat(pts, n_grasps, axis=0)
            else:
                raise ValueError(f"Contact shape mismatch: grasps={n_grasps}, contacts={pts.shape}")

        return pts

    def transform_grasp_matrix_for_vis(self, grasp_mat: np.ndarray) -> np.ndarray:
        T_vis = np.array(grasp_mat, dtype=np.float32, copy=True)
        T_vis[:3, :3] = T_vis[:3, :3] @ self.R_fix
        return T_vis

    # -------------------------
    # Distance / geometry
    # -------------------------
    def min_distance_to_cloud(self, point: np.ndarray, cloud: np.ndarray) -> float:
        if cloud is None or cloud.shape[0] == 0:
            return np.inf
        d = np.linalg.norm(cloud - point[None, :], axis=1)
        return float(np.min(d))

    def line_segment_point_distances(self, pts, a, b):
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return np.linalg.norm(pts - a[None, :], axis=1)
        t = np.sum((pts - a[None, :]) * ab[None, :], axis=1) / ab2
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        return np.linalg.norm(pts - proj, axis=1)

    def background_collision_proxy(self, grasp_raw: np.ndarray, bg_cloud: np.ndarray) -> bool:
        if (bg_cloud is None) or (bg_cloud.shape[0] == 0):
            return False

        g_vis = self.transform_grasp_matrix_for_vis(grasp_raw)
        p = g_vis[:3, 3]
        approach = g_vis[:3, 0]
        approach = approach / max(np.linalg.norm(approach), 1e-8)

        a = p - 0.5 * self.approach_collision_length * approach
        b = p + 0.5 * self.approach_collision_length * approach

        d_line = self.line_segment_point_distances(bg_cloud, a, b)
        if np.any(d_line < self.approach_collision_radius):
            return True

        d_palm = np.linalg.norm(bg_cloud - p[None, :], axis=1)
        if np.any(d_palm < self.palm_collision_radius):
            return True

        return False

    def rotation_angle_deg(self, R_a: np.ndarray, R_b: np.ndarray) -> float:
        R_rel = R_a.T @ R_b
        tr = np.trace(R_rel)
        cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def grasp_distance(self, g_a: np.ndarray, g_b: np.ndarray):
        p_a = g_a[:3, 3]
        p_b = g_b[:3, 3]
        pos_dist = float(np.linalg.norm(p_a - p_b))
        ang_deg = self.rotation_angle_deg(g_a[:3, :3], g_b[:3, :3])
        return pos_dist, ang_deg

    # -------------------------
    # FPS / tracking / EMA
    # -------------------------
    def farthest_point_sample_indices(self, points: np.ndarray, k: int) -> np.ndarray:
        n = points.shape[0]
        if n == 0:
            return np.array([], dtype=np.int32)
        if n <= k:
            return np.arange(n, dtype=np.int32)

        selected = np.zeros(k, dtype=np.int32)
        centroid = np.mean(points, axis=0)
        selected[0] = int(np.argmax(np.linalg.norm(points - centroid[None, :], axis=1)))

        dist = np.linalg.norm(points - points[selected[0]][None, :], axis=1)

        for i in range(1, k):
            selected[i] = int(np.argmax(dist))
            new_dist = np.linalg.norm(points - points[selected[i]][None, :], axis=1)
            dist = np.minimum(dist, new_dist)

        return selected

    def apply_fps(self, grasps, scores, widths, contacts):
        if (not self.enable_fps) or (grasps.shape[0] <= self.max_fps_candidates):
            return {
                "grasps": grasps,
                "scores": scores,
                "widths": widths,
                "contacts": contacts,
            }

        idx = self.farthest_point_sample_indices(contacts, self.max_fps_candidates)
        self.get_logger().info(f"[fps] selected {len(idx)}/{grasps.shape[0]} representative candidates")
        return {
            "grasps": grasps[idx],
            "scores": scores[idx],
            "widths": widths[idx],
            "contacts": contacts[idx],
        }

    def select_best_with_tracking(self, grasps, scores, contacts):
        if grasps.shape[0] == 0:
            return None

        if (not self.enable_tracking) or (self.prev_best_raw is None):
            return int(np.argmax(scores))

        best_idx = None
        best_track_score = -1e9

        for i, (g, s, c) in enumerate(zip(grasps, scores, contacts)):
            pos_dist, ang_deg = self.grasp_distance(g, self.prev_best_raw)

            bonus = 0.0
            if pos_dist <= self.track_max_pos_dist and ang_deg <= self.track_max_angle_deg:
                pos_term = 1.0 - (pos_dist / max(self.track_max_pos_dist, 1e-6))
                ang_term = 1.0 - (ang_deg / max(self.track_max_angle_deg, 1e-6))
                bonus = self.track_score_bonus * (0.5 * pos_term + 0.5 * ang_term)

            track_score = float(s) + bonus
            if track_score > best_track_score:
                best_track_score = track_score
                best_idx = i

        self.get_logger().info(
            f"[track] selected idx={best_idx} raw_score={scores[best_idx]:.4f} track_score={best_track_score:.4f}"
        )
        return int(best_idx)

    def ema_pose_matrix(self, prev_mat: np.ndarray, cur_mat: np.ndarray, alpha_pos: float, alpha_rot: float):
        if prev_mat is None:
            return np.array(cur_mat, dtype=np.float32, copy=True)

        out = np.array(cur_mat, dtype=np.float32, copy=True)

        p_prev = prev_mat[:3, 3]
        p_cur = cur_mat[:3, 3]
        out[:3, 3] = alpha_pos * p_prev + (1.0 - alpha_pos) * p_cur

        r_prev = R.from_matrix(prev_mat[:3, :3])
        r_cur = R.from_matrix(cur_mat[:3, :3])

        key_times = [0.0, 1.0]
        key_rots = R.concatenate([r_prev, r_cur])
        slerp = Slerp(key_times, key_rots)
        mixed = slerp([1.0 - alpha_rot])[0]
        out[:3, :3] = mixed.as_matrix().astype(np.float32)

        return out

    def maybe_confirm_candidate(self, grasp_raw: np.ndarray):
        if not self.enable_confirmation:
            return True

        if self.pending_candidate is None:
            self.pending_candidate = np.array(grasp_raw, copy=True)
            self.pending_count = 1
            self.get_logger().info("[confirm] start pending candidate")
            return False

        pos_dist, ang_deg = self.grasp_distance(grasp_raw, self.pending_candidate)
        if pos_dist <= self.confirm_pos_dist and ang_deg <= self.confirm_angle_deg:
            self.pending_count += 1
        else:
            self.pending_candidate = np.array(grasp_raw, copy=True)
            self.pending_count = 1

        self.get_logger().info(
            f"[confirm] pending_count={self.pending_count}/{self.confirm_required_frames}"
        )

        if self.pending_count >= self.confirm_required_frames:
            self.pending_candidate = np.array(grasp_raw, copy=True)
            return True

        return False

    # -------------------------
    # Filtering
    # -------------------------
    def threshold_by_confidence(self, grasps, scores, widths, contacts):
        th = self.primary_conf_thresh
        keep = scores >= th
        if int(np.sum(keep)) < self.min_confident_grasps:
            th = self.fallback_conf_thresh
            keep = scores >= th

        self.get_logger().info(
            f"[conf] threshold={th:.3f}, kept={int(np.sum(keep))}/{len(scores)}"
        )

        return {
            "grasps": grasps[keep],
            "scores": scores[keep],
            "widths": widths[keep],
            "contacts": contacts[keep],
        }

    def filter_candidates_with_obj_bg(self, grasps, scores, widths, contacts, obj_pc, bg_pc):
        kept_g = []
        kept_s = []
        kept_w = []
        kept_c = []

        reject_width = 0
        reject_nan = 0
        reject_vertical = 0
        reject_obj_dist = 0
        reject_bg_prefer = 0
        reject_bg_collision = 0

        for g_raw, s, w, c in zip(grasps, scores, widths, contacts):
            if not np.isfinite(s) or not np.isfinite(w) or not np.all(np.isfinite(c)):
                reject_nan += 1
                continue

            if (w > self.max_gripper_width) or (w < self.min_gripper_width):
                reject_width += 1
                continue

            g_vis = self.transform_grasp_matrix_for_vis(g_raw)
            approach_vis = g_vis[:3, 0]
            approach_vis = approach_vis / max(np.linalg.norm(approach_vis), 1e-8)

            if self.enable_vertical_filter:
                vertical_alignment = abs(float(np.dot(approach_vis, self.vertical_axis_in_camera)))
                if vertical_alignment > self.max_vertical_alignment:
                    reject_vertical += 1
                    continue

            obj_dist = self.min_distance_to_cloud(c, obj_pc)
            bg_dist = self.min_distance_to_cloud(c, bg_pc)

            if np.isfinite(obj_dist) and (obj_dist > self.max_contact_to_object_dist):
                reject_obj_dist += 1
                continue

            if np.isfinite(obj_dist) and np.isfinite(bg_dist):
                if (bg_dist + self.min_contact_bg_margin) < obj_dist:
                    reject_bg_prefer += 1
                    continue

            if self.enable_bg_collision_check and self.background_collision_proxy(g_raw, bg_pc):
                reject_bg_collision += 1
                continue

            kept_g.append(g_raw)
            kept_s.append(s)
            kept_w.append(w)
            kept_c.append(c)

        self.get_logger().info(
            f"[filter] reject_width={reject_width}, reject_nan={reject_nan}, "
            f"reject_vertical={reject_vertical}, reject_obj_dist={reject_obj_dist}, "
            f"reject_bg_prefer={reject_bg_prefer}, reject_bg_collision={reject_bg_collision}, "
            f"kept={len(kept_g)}"
        )

        if len(kept_g) == 0:
            return {"grasps": [], "scores": [], "widths": [], "contacts": []}

        return {
            "grasps": np.asarray(kept_g),
            "scores": np.asarray(kept_s),
            "widths": np.asarray(kept_w),
            "contacts": np.asarray(kept_c),
        }

    # -------------------------
    # Publishing / visualization
    # -------------------------
    def publish_pose(self, pub, grasp_mat, header):
        tx, ty, tz = grasp_mat[0:3, 3]
        rot_mat = grasp_mat[0:3, 0:3]
        q = R.from_matrix(rot_mat).as_quat()

        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = float(tx)
        pose_msg.pose.position.y = float(ty)
        pose_msg.pose.position.z = float(tz)
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])
        pub.publish(pose_msg)

    def publish_best_contact_point(self, contact_pt, header):
        msg = PointStamped()
        msg.header = header
        msg.point.x = float(contact_pt[0])
        msg.point.y = float(contact_pt[1])
        msg.point.z = float(contact_pt[2])
        self.pub_best_contact_point.publish(msg)

    def publish_best_grasp_marker(self, grasp_mat_vis, header):
        marker = Marker()
        marker.header = header
        marker.ns = "best_grasp"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        tx, ty, tz = grasp_mat_vis[0:3, 3]
        rot_mat = grasp_mat_vis[0:3, 0:3]
        q = R.from_matrix(rot_mat).as_quat()

        marker.pose.position.x = float(tx)
        marker.pose.position.y = float(ty)
        marker.pose.position.z = float(tz)

        marker.pose.orientation.x = float(q[0])
        marker.pose.orientation.y = float(q[1])
        marker.pose.orientation.z = float(q[2])
        marker.pose.orientation.w = float(q[3])

        marker.scale.x = self.best_arrow_length
        marker.scale.y = self.best_arrow_shaft_diameter
        marker.scale.z = self.best_arrow_head_diameter

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.pub_best_marker.publish(marker)

    def publish_best_contact_marker(self, contact_pt, header):
        marker = Marker()
        marker.header = header
        marker.ns = "best_contact"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(contact_pt[0])
        marker.pose.position.y = float(contact_pt[1])
        marker.pose.position.z = float(contact_pt[2])
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.best_contact_scale
        marker.scale.y = self.best_contact_scale
        marker.scale.z = self.best_contact_scale

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.pub_best_contact.publish(marker)

    def publish_grasp_markers(self, grasps_raw, rank_scores, contacts, header, topk=5):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        order = np.argsort(rank_scores)[::-1][:topk]
        top_vals = rank_scores[order]
        s_min = float(np.min(top_vals))
        s_max = float(np.max(top_vals))
        denom = max(s_max - s_min, 1e-6)

        marker_id = 0
        for idx in order:
            g_vis = self.transform_grasp_matrix_for_vis(grasps_raw[idx])
            score = float(rank_scores[idx])
            c_pt = contacts[idx]

            q = R.from_matrix(g_vis[0:3, 0:3]).as_quat()
            norm_score = (score - s_min) / denom

            arrow = Marker()
            arrow.header = header
            arrow.ns = "grasp_candidates"
            arrow.id = marker_id
            marker_id += 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.pose.position.x = float(g_vis[0, 3])
            arrow.pose.position.y = float(g_vis[1, 3])
            arrow.pose.position.z = float(g_vis[2, 3])

            arrow.pose.orientation.x = float(q[0])
            arrow.pose.orientation.y = float(q[1])
            arrow.pose.orientation.z = float(q[2])
            arrow.pose.orientation.w = float(q[3])

            arrow.scale.x = self.candidate_arrow_length
            arrow.scale.y = self.candidate_arrow_shaft_diameter
            arrow.scale.z = self.candidate_arrow_head_diameter

            arrow.color.a = 0.8
            arrow.color.r = float(1.0 - norm_score)
            arrow.color.g = float(norm_score)
            arrow.color.b = 0.0

            marker_array.markers.append(arrow)

            contact = Marker()
            contact.header = header
            contact.ns = "contact_candidates"
            contact.id = marker_id
            marker_id += 1
            contact.type = Marker.SPHERE
            contact.action = Marker.ADD

            contact.pose.position.x = float(c_pt[0])
            contact.pose.position.y = float(c_pt[1])
            contact.pose.position.z = float(c_pt[2])
            contact.pose.orientation.w = 1.0

            contact.scale.x = self.candidate_contact_scale
            contact.scale.y = self.candidate_contact_scale
            contact.scale.z = self.candidate_contact_scale

            contact.color.a = 0.85
            contact.color.r = 0.0
            contact.color.g = 0.7
            contact.color.b = 1.0

            marker_array.markers.append(contact)

        self.pub_markers.publish(marker_array)

    def log_top_candidates(self, grasps, scores, widths, contacts, topk=10):
        order = np.argsort(scores)[::-1][:topk]
        self.get_logger().info(f"[final] top {len(order)} candidates:")
        for rank, idx in enumerate(order, start=1):
            p = grasps[idx][0:3, 3]
            c = contacts[idx]
            self.get_logger().info(
                f"  #{rank:02d} idx={idx:03d} raw={scores[idx]:.4f} "
                f"width={widths[idx]*100.0:.2f}cm "
                f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                f"contact=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})"
            )

    def publish_from_state(self, raw_mat, vis_mat, contact_pt, score, width, header,
                           grasps_for_markers=None, scores_for_markers=None, contacts_for_markers=None):
        self.publish_pose(self.pub_pose_raw, raw_mat, header)
        self.publish_pose(self.pub_pose_vis, vis_mat, header)
        self.publish_best_grasp_marker(vis_mat, header)
        self.publish_best_contact_marker(contact_pt, header)
        self.publish_best_contact_point(contact_pt, header)

        if grasps_for_markers is not None and scores_for_markers is not None and contacts_for_markers is not None:
            self.publish_grasp_markers(
                grasps_for_markers,
                scores_for_markers,
                contacts_for_markers,
                header,
                topk=self.marker_topk
            )

        self.prev_best_raw = np.array(raw_mat, copy=True)
        self.prev_best_vis = np.array(vis_mat, copy=True)
        self.prev_best_contact = np.array(contact_pt, copy=True)
        self.prev_best_score = float(score)
        self.prev_best_width = float(width)
        self.prev_best_header = header

    # -------------------------
    # Main callback
    # -------------------------
    def pc_cb(self, msg: PointCloud2):
        cb_t0 = time.time()
        self.get_logger().info(
            f"[pc_cb entered] width={msg.width}, height={msg.height}, frame={msg.header.frame_id}"
        )

        pc_np = self.read_xyz_from_cloud(msg)
        self.get_logger().info(f"[pc_cb] valid xyz points = {pc_np.shape[0]}")

        if pc_np.shape[0] < self.min_input_points:
            self.get_logger().warn("Point cloud is too sparse. Skipping.")
            return

        pc_np = self.deterministic_downsample(pc_np, self.max_input_points)
        self.get_logger().info(f"[pc_cb] np shape after downsample = {pc_np.shape}")

        try:
            self.get_logger().info("[pc_cb] starting Contact-GraspNet inference...")
            infer_t0 = time.time()

            pred_grasps_cam, scores, contact_pts, metadata = self.grasp_estimator.predict_scene_grasps(
                self.sess,
                pc_full=pc_np,
                pc_segments={},
                local_regions=False,
                filter_grasps=False,
                forward_passes=1
            )

            infer_dt = time.time() - infer_t0
            self.get_logger().info(f"[pc_cb] inference returned in {infer_dt:.3f}s")
        except Exception as e:
            self.get_logger().error(f"[pc_cb] inference failed: {e}")
            if self.enable_hold_last and self.prev_best_raw is not None and self.lost_frame_count < self.max_hold_frames:
                self.lost_frame_count += 1
                self.get_logger().warn(f"[hold] inference failed, reuse previous grasp ({self.lost_frame_count}/{self.max_hold_frames})")
                self.publish_from_state(
                    self.prev_best_raw,
                    self.prev_best_vis,
                    self.prev_best_contact,
                    self.prev_best_score,
                    self.prev_best_width,
                    msg.header,
                )
            return

        if (-1 not in pred_grasps_cam) or (pred_grasps_cam[-1] is None):
            self.get_logger().warn("No grasp poses found!")
            if self.enable_hold_last and self.prev_best_raw is not None and self.lost_frame_count < self.max_hold_frames:
                self.lost_frame_count += 1
                self.get_logger().warn(f"[hold] no grasp, reuse previous grasp ({self.lost_frame_count}/{self.max_hold_frames})")
                self.publish_from_state(
                    self.prev_best_raw,
                    self.prev_best_vis,
                    self.prev_best_contact,
                    self.prev_best_score,
                    self.prev_best_width,
                    msg.header,
                )
            return

        original_grasps = pred_grasps_cam[-1]
        original_scores = scores[-1]
        original_widths = metadata[-1]
        original_contacts = contact_pts[-1]

        try:
            original_grasps, original_scores, original_widths = self.normalize_candidate_arrays(
                original_grasps, original_scores, original_widths
            )
            original_contacts = self.normalize_contact_points(
                original_contacts, original_grasps.shape[0]
            )
        except Exception as e:
            self.get_logger().error(f"[pc_cb] candidate normalization failed: {e}")
            return

        if original_grasps.shape[0] == 0:
            self.get_logger().warn("No grasp poses found after normalization!")
            return

        self.get_logger().info(f"[pc_cb] raw candidates = {original_grasps.shape[0]}")

        conf_filtered = self.threshold_by_confidence(
            original_grasps, original_scores, original_widths, original_contacts
        )

        if len(conf_filtered["grasps"]) == 0:
            self.get_logger().warn("All grasps removed by confidence threshold.")
            if self.enable_hold_last and self.prev_best_raw is not None and self.lost_frame_count < self.max_hold_frames:
                self.lost_frame_count += 1
                self.get_logger().warn(f"[hold] confidence empty, reuse previous grasp ({self.lost_frame_count}/{self.max_hold_frames})")
                self.publish_from_state(
                    self.prev_best_raw,
                    self.prev_best_vis,
                    self.prev_best_contact,
                    self.prev_best_score,
                    self.prev_best_width,
                    msg.header,
                )
            return

        fps_filtered = self.apply_fps(
            conf_filtered["grasps"],
            conf_filtered["scores"],
            conf_filtered["widths"],
            conf_filtered["contacts"]
        )

        obj_pc = self.latest_obj_pc
        bg_pc = self.latest_bg_pc

        filtered = self.filter_candidates_with_obj_bg(
            fps_filtered["grasps"],
            fps_filtered["scores"],
            fps_filtered["widths"],
            fps_filtered["contacts"],
            obj_pc,
            bg_pc,
        )

        if len(filtered["grasps"]) == 0:
            self.get_logger().warn("All grasps filtered! No valid grasp remains.")
            if self.enable_hold_last and self.prev_best_raw is not None and self.lost_frame_count < self.max_hold_frames:
                self.lost_frame_count += 1
                self.get_logger().warn(f"[hold] filtered empty, reuse previous grasp ({self.lost_frame_count}/{self.max_hold_frames})")
                self.publish_from_state(
                    self.prev_best_raw,
                    self.prev_best_vis,
                    self.prev_best_contact,
                    self.prev_best_score,
                    self.prev_best_width,
                    msg.header,
                )
            return

        filtered_grasps = filtered["grasps"]
        filtered_scores = filtered["scores"]
        filtered_widths = filtered["widths"]
        filtered_contacts = filtered["contacts"]

        best_idx = self.select_best_with_tracking(
            filtered_grasps,
            filtered_scores,
            filtered_contacts
        )
        if best_idx is None:
            self.get_logger().warn("Best index selection failed.")
            return

        best_grasp_raw = filtered_grasps[best_idx]
        best_grasp_vis = self.transform_grasp_matrix_for_vis(best_grasp_raw)
        best_score = float(filtered_scores[best_idx])
        best_width = float(filtered_widths[best_idx])
        best_contact_pt = filtered_contacts[best_idx]

        confirmed = self.maybe_confirm_candidate(best_grasp_raw)
        if not confirmed:
            if self.enable_hold_last and self.prev_best_raw is not None:
                self.get_logger().info("[confirm] not confirmed yet, keep previous grasp")
                self.publish_from_state(
                    self.prev_best_raw,
                    self.prev_best_vis,
                    self.prev_best_contact,
                    self.prev_best_score,
                    self.prev_best_width,
                    msg.header,
                    grasps_for_markers=filtered_grasps,
                    scores_for_markers=filtered_scores,
                    contacts_for_markers=filtered_contacts,
                )
            return

        self.lost_frame_count = 0

        out_raw = np.array(best_grasp_raw, copy=True)
        out_vis = np.array(best_grasp_vis, copy=True)

        if self.enable_pose_ema:
            self.smoothed_raw = self.ema_pose_matrix(
                self.smoothed_raw,
                best_grasp_raw,
                self.ema_alpha_pos,
                self.ema_alpha_rot
            )
            self.smoothed_vis = self.ema_pose_matrix(
                self.smoothed_vis,
                best_grasp_vis,
                self.ema_alpha_pos,
                self.ema_alpha_rot
            )
            out_raw = self.smoothed_raw
            out_vis = self.smoothed_vis
        else:
            self.smoothed_raw = np.array(best_grasp_raw, copy=True)
            self.smoothed_vis = np.array(best_grasp_vis, copy=True)

        self.publish_from_state(
            out_raw,
            out_vis,
            best_contact_pt,
            best_score,
            best_width,
            msg.header,
            grasps_for_markers=filtered_grasps,
            scores_for_markers=filtered_scores,
            contacts_for_markers=filtered_contacts,
        )

        total_dt = time.time() - cb_t0

        grasp_pos_raw = out_raw[0:3, 3]
        q_raw = R.from_matrix(out_raw[0:3, 0:3]).as_quat()
        q_vis = R.from_matrix(out_vis[0:3, 0:3]).as_quat()

        obj_dist = self.min_distance_to_cloud(best_contact_pt, obj_pc)
        bg_dist = self.min_distance_to_cloud(best_contact_pt, bg_pc)

        self.get_logger().info("--- Inference Report ---")
        self.get_logger().info(f"Total Callback Time: {total_dt:.3f}s | Best Raw Score: {best_score:.4f}")
        self.get_logger().info(f"Predicted Width: {best_width * 100.0:.2f} cm")
        self.get_logger().info(
            f"Smoothed Raw Position: x={grasp_pos_raw[0]:.4f}, y={grasp_pos_raw[1]:.4f}, z={grasp_pos_raw[2]:.4f}"
        )
        self.get_logger().info(
            f"Smoothed Raw Orientation (Quat): x={q_raw[0]:.4f}, y={q_raw[1]:.4f}, z={q_raw[2]:.4f}, w={q_raw[3]:.4f}"
        )
        self.get_logger().info(
            f"Smoothed Vis Orientation (Quat): x={q_vis[0]:.4f}, y={q_vis[1]:.4f}, z={q_vis[2]:.4f}, w={q_vis[3]:.4f}"
        )
        self.get_logger().info(
            f"Best contact point: x={best_contact_pt[0]:.4f}, y={best_contact_pt[1]:.4f}, z={best_contact_pt[2]:.4f}"
        )
        self.get_logger().info(f"contact->object min dist: {obj_dist:.4f} m")
        self.get_logger().info(f"contact->background min dist: {bg_dist:.4f} m")

        self.log_top_candidates(
            filtered_grasps,
            filtered_scores,
            filtered_widths,
            filtered_contacts,
            topk=self.topk_debug
        )


def main(args=None):
    rclpy.init(args=args)
    node = ContactGraspRoiFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()