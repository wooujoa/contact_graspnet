#!/usr/bin/env python3
import os
import sys
import time
import importlib.util
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

# ---- Contact-GraspNet Imports ----
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


class ContactGraspNode(Node):
    def __init__(self):
        super().__init__("contact_grasp_node")

        # ---- Topics ----
        self.pc_sub_topic = "/yolo/target_pc"

        self.pose_raw_pub_topic = "/grasp/best_pose_raw"
        self.pose_vis_pub_topic = "/grasp/best_pose_vis"
        self.marker_pub_topic = "/grasp/markers"
        self.best_marker_pub_topic = "/grasp/best_pose_marker"
        self.best_contact_pub_topic = "/grasp/best_contact_marker"

        # ---- Config ----
        self.ckpt_dir = os.path.join(
            BASE_DIR, "checkpoints/scene_test_2048_bs3_hor_sigma_001"
        )

        self.max_input_points = 8000
        self.min_input_points = 50

        # 물리 그리퍼 폭 한계
        self.max_gripper_width = 0.15   # 10 cm
        self.min_gripper_width = 0.005   # 1 cm

        # Contact-GraspNet raw frame -> visualization/use frame
        # raw R = [b, a×b, a]
        # new x = old z (= approach a)
        self.R_fix = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # 수직 접근 필터
        # camera optical frame에서 기본적으로 y가 아래 방향이라고 보고,
        # vis frame의 approach(+X)가 이 축과 너무 평행하면 제거
        self.vertical_axis_in_camera = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.vertical_axis_in_camera /= np.linalg.norm(self.vertical_axis_in_camera)
        self.max_vertical_alignment = 0.70  # 더 엄격

        # Re-ranking weights
        self.w_model_score = 1.0
        self.w_center = 0.60
        self.w_lateral = 1.20
        self.w_axial = 0.55
        self.w_axis_alignment = 0.35
        self.w_width_pref = 0.15

        # ---- Visualization / logging ----
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

        # ---- ROS IO ----
        self.sub_pc = self.create_subscription(
            PointCloud2, self.pc_sub_topic, self.pc_cb, 10
        )
        self.pub_pose_raw = self.create_publisher(PoseStamped, self.pose_raw_pub_topic, 10)
        self.pub_pose_vis = self.create_publisher(PoseStamped, self.pose_vis_pub_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_pub_topic, 10)
        self.pub_best_marker = self.create_publisher(Marker, self.best_marker_pub_topic, 10)
        self.pub_best_contact = self.create_publisher(Marker, self.best_contact_pub_topic, 10)

        # ---- Initialize Contact-GraspNet ----
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

        self.grasp_estimator.load_weights(
            self.sess, saver, self.ckpt_dir, mode="test"
        )

        self.get_logger().info("Contact-GraspNet Initialized! Waiting for PointCloud...")

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
                raise ValueError(
                    f"Contact shape mismatch: grasps={n_grasps}, contacts={pts.shape}"
                )

        return pts

    def transform_grasp_matrix_for_vis(self, grasp_mat: np.ndarray) -> np.ndarray:
        T_vis = np.array(grasp_mat, dtype=np.float32, copy=True)
        T_vis[:3, :3] = T_vis[:3, :3] @ self.R_fix
        return T_vis

    def deterministic_downsample(self, pc_np: np.ndarray, max_points: int) -> np.ndarray:
        n = pc_np.shape[0]
        if n <= max_points:
            return pc_np
        idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
        return pc_np[idx]

    def compute_object_stats(self, pc_np: np.ndarray):
        center = np.mean(pc_np, axis=0)

        centered = pc_np - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        major_axis = eigvecs[:, 0]
        if major_axis[2] < 0:
            major_axis = -major_axis

        proj_major = centered @ major_axis
        axial_min = float(np.min(proj_major))
        axial_max = float(np.max(proj_major))
        axial_extent = max(axial_max - axial_min, 1e-6)

        band_lo = axial_min + 0.30 * axial_extent
        band_hi = axial_min + 0.70 * axial_extent
        body_center_scalar = 0.5 * (band_lo + band_hi)

        residual = centered - np.outer(proj_major, major_axis)
        lateral_dist = np.linalg.norm(residual, axis=1)
        lateral_scale = max(float(np.percentile(lateral_dist, 90)), 1e-4)

        width_pref = min(0.06, self.max_gripper_width * 0.6)

        return {
            "center": center,
            "major_axis": major_axis,
            "eigvals": eigvals,
            "axial_min": axial_min,
            "axial_max": axial_max,
            "axial_extent": axial_extent,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "body_center_scalar": body_center_scalar,
            "lateral_scale": lateral_scale,
            "width_pref": width_pref,
        }

    def log_object_stats(self, obj_stats: dict):
        c = obj_stats["center"]
        a = obj_stats["major_axis"]
        self.get_logger().info(
            f"[object] center=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})"
        )
        self.get_logger().info(
            f"[object] major_axis=({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}) | axial_extent={obj_stats['axial_extent']:.4f} | lateral_scale={obj_stats['lateral_scale']:.4f}"
        )

    def filter_candidates(self, grasps, scores, widths, contacts):
        kept_g = []
        kept_s = []
        kept_w = []
        kept_c = []

        reject_width = 0
        reject_nan = 0
        reject_vertical = 0

        for g_raw, s, w, c in zip(grasps, scores, widths, contacts):
            if not np.isfinite(s) or not np.isfinite(w) or not np.all(np.isfinite(c)):
                reject_nan += 1
                continue

            if w > self.max_gripper_width or w < self.min_gripper_width:
                reject_width += 1
                continue

            g_vis = self.transform_grasp_matrix_for_vis(g_raw)
            approach_vis = g_vis[:3, 0]
            approach_vis = approach_vis / max(np.linalg.norm(approach_vis), 1e-8)

            vertical_alignment = abs(float(np.dot(approach_vis, self.vertical_axis_in_camera)))
            if vertical_alignment > self.max_vertical_alignment:
                reject_vertical += 1
                continue

            kept_g.append(g_raw)
            kept_s.append(s)
            kept_w.append(w)
            kept_c.append(c)

        self.get_logger().info(
            f"[filter] reject_width={reject_width}, reject_nan={reject_nan}, reject_vertical={reject_vertical}, kept={len(kept_g)}"
        )

        if len(kept_g) == 0:
            return {"grasps": [], "scores": [], "widths": [], "contacts": []}

        return {
            "grasps": np.asarray(kept_g),
            "scores": np.asarray(kept_s),
            "widths": np.asarray(kept_w),
            "contacts": np.asarray(kept_c),
        }

    def rerank_candidates(self, grasps, scores, widths, obj_stats):
        center = obj_stats["center"]
        axis = obj_stats["major_axis"]
        axial_extent = obj_stats["axial_extent"]
        lateral_scale = obj_stats["lateral_scale"]
        body_center_scalar = obj_stats["body_center_scalar"]
        width_pref = obj_stats["width_pref"]

        final_scores = []
        center_dists = []
        lateral_norms = []
        axial_norms = []
        align_penalties = []
        width_penalties = []

        for g_raw, s, w in zip(grasps, scores, widths):
            p = g_raw[0:3, 3]
            g_vis = self.transform_grasp_matrix_for_vis(g_raw)
            approach = g_vis[0:3, 0]

            rel = p - center
            axial = float(np.dot(rel, axis))
            axial_norm = abs(axial - body_center_scalar) / max(0.5 * axial_extent, 1e-6)

            lateral_vec = rel - axial * axis
            lateral_dist = float(np.linalg.norm(lateral_vec))
            lateral_norm = lateral_dist / max(lateral_scale, 1e-6)

            center_dist = float(np.linalg.norm(rel))
            center_norm = center_dist / max(axial_extent, 1e-6)

            align_penalty = abs(float(np.dot(approach, axis)))
            width_penalty = abs(float(w) - width_pref) / max(self.max_gripper_width, 1e-6)

            rank_score = (
                self.w_model_score * float(s)
                - self.w_center * center_norm
                - self.w_lateral * lateral_norm
                - self.w_axial * axial_norm
                - self.w_axis_alignment * align_penalty
                - self.w_width_pref * width_penalty
            )

            final_scores.append(rank_score)
            center_dists.append(center_norm)
            lateral_norms.append(lateral_norm)
            axial_norms.append(axial_norm)
            align_penalties.append(align_penalty)
            width_penalties.append(width_penalty)

        final_scores = np.asarray(final_scores, dtype=np.float32)
        best_idx = int(np.argmax(final_scores))

        return {
            "best_idx": best_idx,
            "final_scores": final_scores,
            "center_norms": np.asarray(center_dists, dtype=np.float32),
            "lateral_norms": np.asarray(lateral_norms, dtype=np.float32),
            "axial_norms": np.asarray(axial_norms, dtype=np.float32),
            "align_penalties": np.asarray(align_penalties, dtype=np.float32),
            "width_penalties": np.asarray(width_penalties, dtype=np.float32),
        }

    def log_top_candidates(self, grasps, scores, widths, contacts, ranking, topk=10):
        order = np.argsort(ranking["final_scores"])[::-1][:topk]
        self.get_logger().info(f"[rerank] top {len(order)} candidates:")
        for rank, idx in enumerate(order, start=1):
            p = grasps[idx][0:3, 3]
            c = contacts[idx]
            g_vis = self.transform_grasp_matrix_for_vis(grasps[idx])
            approach_vis = g_vis[:3, 0]
            vertical_alignment = abs(float(np.dot(
                approach_vis / max(np.linalg.norm(approach_vis), 1e-8),
                self.vertical_axis_in_camera
            )))
            gc_dist = float(np.linalg.norm(p - c))
            self.get_logger().info(
                f"  #{rank:02d} idx={idx:03d} "
                f"final={ranking['final_scores'][idx]:.4f} "
                f"raw={scores[idx]:.4f} "
                f"width={widths[idx]*100.0:.2f}cm "
                f"center={ranking['center_norms'][idx]:.3f} "
                f"lat={ranking['lateral_norms'][idx]:.3f} "
                f"ax={ranking['axial_norms'][idx]:.3f} "
                f"align={ranking['align_penalties'][idx]:.3f} "
                f"vert={vertical_alignment:.3f} "
                f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                f"contact=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
                f"g2c={gc_dist:.3f}"
            )

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

    def pc_cb(self, msg: PointCloud2):
        cb_t0 = time.time()
        self.get_logger().info(
            f"[pc_cb entered] width={msg.width}, height={msg.height}, frame={msg.header.frame_id}"
        )

        try:
            pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pc_list = list(pc_data)
        except Exception as e:
            self.get_logger().error(f"[pc_cb] read_points failed: {e}")
            return

        self.get_logger().info(f"[pc_cb] valid xyz points = {len(pc_list)}")

        if len(pc_list) < self.min_input_points:
            self.get_logger().warn("Point cloud is too sparse. Skipping.")
            return

        try:
            pc_np = np.array([[p[0], p[1], p[2]] for p in pc_list], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"[pc_cb] numpy conversion failed: {e}")
            return

        self.get_logger().info(f"[pc_cb] np shape before downsample = {pc_np.shape}")

        pc_np = self.deterministic_downsample(pc_np, self.max_input_points)
        self.get_logger().info(f"[pc_cb] np shape after downsample = {pc_np.shape}")

        obj_stats = self.compute_object_stats(pc_np)
        self.log_object_stats(obj_stats)

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
            return

        if -1 not in pred_grasps_cam:
            self.get_logger().warn("No grasp poses found!")
            return

        if pred_grasps_cam[-1] is None:
            self.get_logger().warn("No grasp poses found!")
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

        self.get_logger().info(
            f"[pc_cb] raw candidates = {original_grasps.shape[0]}"
        )

        filtered = self.filter_candidates(
            original_grasps,
            original_scores,
            original_widths,
            original_contacts,
        )

        if len(filtered["grasps"]) == 0:
            self.get_logger().warn("All grasps filtered! No valid grasp remains.")
            return

        filtered_grasps = filtered["grasps"]
        filtered_scores = filtered["scores"]
        filtered_widths = filtered["widths"]
        filtered_contacts = filtered["contacts"]
        self.get_logger().info(
            f"Filtered grasps: {len(filtered_grasps)} / {len(original_grasps)} remaining."
        )

        ranking = self.rerank_candidates(
            filtered_grasps,
            filtered_scores,
            filtered_widths,
            obj_stats
        )

        best_idx = ranking["best_idx"]

        best_grasp_raw = filtered_grasps[best_idx]
        best_grasp_vis = self.transform_grasp_matrix_for_vis(best_grasp_raw)

        best_score = float(filtered_scores[best_idx])
        best_width = float(filtered_widths[best_idx])
        best_rank_score = float(ranking["final_scores"][best_idx])
        best_contact_pt = filtered_contacts[best_idx]

        grasp_pos_raw = best_grasp_raw[0:3, 3]
        grasp_pos_vis = best_grasp_vis[0:3, 3]
        q_raw = R.from_matrix(best_grasp_raw[0:3, 0:3]).as_quat()
        q_vis = R.from_matrix(best_grasp_vis[0:3, 0:3]).as_quat()

        pc_center = obj_stats["center"]
        dist_diff = float(np.linalg.norm(pc_center - grasp_pos_raw))
        contact_dist = float(np.linalg.norm(best_contact_pt - grasp_pos_raw))
        approach_vis = best_grasp_vis[:3, 0]
        approach_vis = approach_vis / max(np.linalg.norm(approach_vis), 1e-8)
        vertical_alignment = abs(float(np.dot(approach_vis, self.vertical_axis_in_camera)))

        self.publish_pose(self.pub_pose_raw, best_grasp_raw, msg.header)
        self.publish_pose(self.pub_pose_vis, best_grasp_vis, msg.header)
        self.publish_best_grasp_marker(best_grasp_vis, msg.header)
        self.publish_best_contact_marker(best_contact_pt, msg.header)
        self.publish_grasp_markers(
            filtered_grasps,
            ranking["final_scores"],
            filtered_contacts,
            msg.header,
            topk=self.marker_topk
        )

        total_dt = time.time() - cb_t0

        self.get_logger().info("--- Inference Report ---")
        self.get_logger().info(
            f"Total Callback Time: {total_dt:.3f}s | Raw Score: {best_score:.4f} | Final Rank: {best_rank_score:.4f}"
        )
        self.get_logger().info(f"Predicted Width: {best_width * 100.0:.2f} cm")
        self.get_logger().info(
            f"Raw Position: x={grasp_pos_raw[0]:.4f}, y={grasp_pos_raw[1]:.4f}, z={grasp_pos_raw[2]:.4f}"
        )
        self.get_logger().info(
            f"Raw Orientation (Quat): x={q_raw[0]:.4f}, y={q_raw[1]:.4f}, z={q_raw[2]:.4f}, w={q_raw[3]:.4f}"
        )
        self.get_logger().info(
            f"Vis Position: x={grasp_pos_vis[0]:.4f}, y={grasp_pos_vis[1]:.4f}, z={grasp_pos_vis[2]:.4f}"
        )
        self.get_logger().info(
            f"Vis Orientation (Quat): x={q_vis[0]:.4f}, y={q_vis[1]:.4f}, z={q_vis[2]:.4f}, w={q_vis[3]:.4f}"
        )
        self.get_logger().info(
            f"Best contact point: x={best_contact_pt[0]:.4f}, y={best_contact_pt[1]:.4f}, z={best_contact_pt[2]:.4f}"
        )
        self.get_logger().info(f"Distance (Object Center to Raw-Grasp): {dist_diff:.4f}m")
        self.get_logger().info(f"Distance (Raw Grasp Origin to Contact): {contact_dist:.4f}m")
        self.get_logger().info(f"Vertical alignment (vis approach vs vertical axis): {vertical_alignment:.4f}")

        self.log_top_candidates(
            filtered_grasps,
            filtered_scores,
            filtered_widths,
            filtered_contacts,
            ranking,
            topk=self.topk_debug
        )


def main(args=None):
    rclpy.init(args=args)
    node = ContactGraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()