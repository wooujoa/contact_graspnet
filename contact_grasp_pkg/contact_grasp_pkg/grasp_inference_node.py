#!/usr/bin/env python3
import os
import sys
import numpy as np
import time

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
CGN_POINTNET_UTILS = os.path.join(BASE_DIR, "pointnet2", "utils")
CGN_POINTNET_MODELS = os.path.join(BASE_DIR, "pointnet2", "models")
CGN_POINTNET_SAMPLING = os.path.join(BASE_DIR, "pointnet2", "tf_ops", "sampling")
CGN_POINTNET_GROUPING = os.path.join(BASE_DIR, "pointnet2", "tf_ops", "grouping")
CGN_POINTNET_INTERP = os.path.join(BASE_DIR, "pointnet2", "tf_ops", "3d_interpolation")

for p in [
    BASE_DIR,
    CGN_ROOT,
    CGN_POINTNET_UTILS,
    CGN_POINTNET_MODELS,
    CGN_POINTNET_SAMPLING,
    CGN_POINTNET_GROUPING,
    CGN_POINTNET_INTERP,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config_utils
from contact_grasp_estimator import GraspEstimator


class ContactGraspNode(Node):
    def __init__(self):
        super().__init__('contact_grasp_node')

        # ---- Topics ----
        self.pc_sub_topic = '/yolo/target_pc'
        self.pose_pub_topic = '/grasp/best_pose'
        self.marker_pub_topic = '/grasp/markers'
        self.best_marker_pub_topic = '/grasp/best_pose_marker'

        # ---- Parameters ----
        self.ckpt_dir = os.path.join(
            BASE_DIR,
            'checkpoints/scene_test_2048_bs3_hor_sigma_001'
        )

        # ---- ROS IO ----
        self.sub_pc = self.create_subscription(
            PointCloud2, self.pc_sub_topic, self.pc_cb, 10
        )
        self.pub_pose = self.create_publisher(PoseStamped, self.pose_pub_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_pub_topic, 10)
        self.pub_best_marker = self.create_publisher(Marker, self.best_marker_pub_topic, 10)

        # ---- Initialize Contact-GraspNet ----
        self.get_logger().info("Loading Contact-GraspNet to GPU...")

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
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
            self.sess, saver, self.ckpt_dir, mode='test'
        )
        self.get_logger().info("Contact-GraspNet Initialized! Waiting for PointCloud...")

    def pc_cb(self, msg: PointCloud2):
        t0 = time.time()

        # 1. PointCloud2 -> Numpy Array (N, 3)
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_list = list(pc_data)

        if len(pc_list) < 50:
            self.get_logger().warn("Point cloud is too sparse. Skipping.")
            return

        pc_np = np.array([[p[0], p[1], p[2]] for p in pc_list], dtype=np.float32)

        # 2. Contact-GraspNet inference (네 번째 인자인 'widths'를 받도록 수정)
        pred_grasps_cam, scores, contact_pts, metadata = self.grasp_estimator.predict_scene_grasps(
            self.sess,
            pc_full=pc_np,
            pc_segments={},
            local_regions=False,
            filter_grasps=False,
            forward_passes=1
        )

        if -1 not in pred_grasps_cam or len(pred_grasps_cam[-1]) == 0:
            self.get_logger().warn("No grasp poses found!")
            return

        original_grasps = pred_grasps_cam[-1]
        original_scores = scores[-1]
        # 모델의 예측 width (단위: Meter)
        original_widths = metadata[-1] 

        # ---------------------------------------------------------
        # Shelf & Gripper Width filtering
        # ---------------------------------------------------------
        filtered_grasps = []
        filtered_scores = []
        filtered_widths = [] # 필터링된 너비 저장용

        # 물리적 그리퍼 한계 설정 (예: FFW-SG2 그리퍼 최대 8cm 벌어짐)
        MAX_GRIPPER_WIDTH = 0.08 
        MIN_GRIPPER_WIDTH = 0.01

        min_y = np.min(pc_np[:, 1])
        max_y = np.max(pc_np[:, 1])
        obj_height = max_y - min_y
        top_threshold = min_y + (obj_height * 0.3)

        for g, s, w in zip(original_grasps, original_scores, original_widths):
            pos_y = g[1, 3]
            approach_vec = g[0:3, 2]

            # Filter 1: Height (기존 동일)
            if pos_y < top_threshold:
                continue

            # Filter 2: Approach angle (기존 동일)
            if abs(approach_vec[1]) > 0.6:
                continue

            # Filter 3: Gripper Width (물리적으로 불가능한 너비 제외)
            # 과자 봉지 정면을 잡으려 하면 w가 MAX_GRIPPER_WIDTH를 초과하게 됨
            if w > MAX_GRIPPER_WIDTH or w < MIN_GRIPPER_WIDTH:
                continue

            filtered_grasps.append(g)
            filtered_scores.append(s)
            filtered_widths.append(w)

        if len(filtered_grasps) == 0:
            self.get_logger().warn("All grasps filtered! Fallback to original grasps.")
            filtered_grasps = original_grasps
            filtered_scores = original_scores
            filtered_widths = original_widths # Fallback 시에도 대응
        else:
            filtered_grasps = np.array(filtered_grasps)
            filtered_scores = np.array(filtered_scores)
            filtered_widths = np.array(filtered_widths) # 이 줄을 추가하세요!
            self.get_logger().info(
                f"Filtered grasps: {len(filtered_grasps)} / {len(original_grasps)} remaining."
            )

        # ---------------------------------------------------------
        # Find best grasp
        # ---------------------------------------------------------
        best_idx = np.argmax(filtered_scores)
        best_grasp_mat = filtered_grasps[best_idx]
        best_score = filtered_scores[best_idx]

        # 위치와 방향(쿼터니언) 추출
        grasp_pos = best_grasp_mat[0:3, 3]
        rot_mat = best_grasp_mat[0:3, 0:3]
        q = R.from_matrix(rot_mat).as_quat() # [x, y, z, w]

        pc_center = np.mean(pc_np, axis=0)
        dist_diff = np.linalg.norm(pc_center - grasp_pos)

        self.publish_grasp_pose(best_grasp_mat, msg.header)
        self.publish_best_grasp_marker(best_grasp_mat, msg.header)
        self.publish_grasp_markers(filtered_grasps, filtered_scores, msg.header)

        dt = time.time() - t0
        
        # --- 터미널 로그 출력 업데이트 ---
        self.get_logger().info(f"--- Inference Report ---")
        self.get_logger().info(f"Time: {dt:.3f}s | Best Score: {best_score:.2f}")
        # 추가: 예측된 그리퍼 너비 출력 (cm 단위로 변환해서 보면 직관적입니다)
        self.get_logger().info(f"Predicted Width: {filtered_widths[best_idx]*100:.2f} cm")
        self.get_logger().info(f"Position: x={grasp_pos[0]:.4f}, y={grasp_pos[1]:.4f}, z={grasp_pos[2]:.4f}")
        # 위치 정보 (x, y, z)
        self.get_logger().info(f"Position: x={grasp_pos[0]:.4f}, y={grasp_pos[1]:.4f}, z={grasp_pos[2]:.4f}")
        # 방향 정보 (Quaternion: x, y, z, w)
        self.get_logger().info(f"Orientation (Quat): x={q[0]:.4f}, y={q[1]:.4f}, z={q[2]:.4f}, w={q[3]:.4f}")
        self.get_logger().info(f"Distance (PC Center to Grasp): {dist_diff:.4f}m")

    def publish_grasp_pose(self, grasp_mat, header):
        """Raw grasp pose for control"""
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
        self.pub_pose.publish(pose_msg)

    def publish_best_grasp_marker(self, grasp_mat, header):
        """Best grasp marker for RViz debugging"""
        marker = Marker()
        marker.header = header
        marker.ns = "best_grasp"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        tx, ty, tz = grasp_mat[0:3, 3]
        rot_mat = grasp_mat[0:3, 0:3]
        q = R.from_matrix(rot_mat).as_quat()

        marker.pose.position.x = float(tx)
        marker.pose.position.y = float(ty)
        marker.pose.position.z = float(tz)

        marker.pose.orientation.x = float(q[0])
        marker.pose.orientation.y = float(q[1])
        marker.pose.orientation.z = float(q[2])
        marker.pose.orientation.w = float(q[3])

        marker.scale.x = 0.12
        marker.scale.y = 0.02
        marker.scale.z = 0.02

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.pub_best_marker.publish(marker)

    def publish_grasp_markers(self, grasps, scores, header):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        top_indices = np.argsort(scores)[::-1][:20]

        for i, idx in enumerate(top_indices):
            g_mat = grasps[idx]
            score = scores[idx]

            correction = R.from_euler('y', -90, degrees=True)
            original_rot = R.from_matrix(g_mat[0:3, 0:3])
            q = (original_rot * correction).as_quat()

            marker = Marker()
            marker.header = header
            marker.ns = "grasp_candidates"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose.position.x = float(g_mat[0, 3])
            marker.pose.position.y = float(g_mat[1, 3])
            marker.pose.position.z = float(g_mat[2, 3])

            marker.pose.orientation.x = float(q[0])
            marker.pose.orientation.y = float(q[1])
            marker.pose.orientation.z = float(q[2])
            marker.pose.orientation.w = float(q[3])

            marker.scale.x = 0.08
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            marker.color.a = 0.8
            marker.color.r = float(1.0 - score)
            marker.color.g = float(score)
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.pub_markers.publish(marker_array)


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


if __name__ == '__main__':
    main()