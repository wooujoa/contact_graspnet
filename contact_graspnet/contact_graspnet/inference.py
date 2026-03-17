#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import glob
import cv2
import time

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

import config_utils
from data import load_available_input_data
from contact_grasp_estimator import GraspEstimator

from visualization_utils import show_image, overlay_grasps_2d_on_rgb


def inference(
    global_config,
    checkpoint_dir,
    input_paths,
    K=None,
    local_regions=True,
    skip_border_objects=False,
    filter_grasps=True,
    segmap_id=None,
    z_range=[0.2, 1.8],
    forward_passes=1,
    save_png=True,
    show_fig=True,
    topk_per_object=10,
    score_thresh=None,
    ignore_segmap=False,
):
    """
    Predict grasps and save a 2D overlay image (no Mayavi required).
    """

    # Build model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Saver
    saver = tf.train.Saver(save_relative_paths=True)

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    os.makedirs('results', exist_ok=True)

    # input_paths is expected to be a single glob string (recommended)
    matched = glob.glob(input_paths)
    if not matched:
        print('No files found:', input_paths)
        return

    for p in matched:
        print('Loading', p)

        # IMPORTANT: do not mutate global flags across files
        local_regions_i = local_regions
        filter_grasps_i = filter_grasps

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)

        if ignore_segmap:
            segmap = None

        # If no segmap -> fall back to scene-mode
        if segmap is None:
            if local_regions_i or filter_grasps_i:
                print("[WARN] segmap is None -> disabling local_regions/filter_grasps (scene-mode).")
            local_regions_i = False
            filter_grasps_i = False
            pc_segments = {}

        # Build point cloud if needed
        if pc_full is None:
            print('Converting depth to point cloud(s)...')

            t0 = time.perf_counter()

            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
                depth,
                cam_K,
                segmap=None if ignore_segmap else segmap,   # <- 너 코드 흐름대로면 이미 segmap=None일 수도 있음
                rgb=rgb,
                skip_border_objects=skip_border_objects,
                z_range=z_range
            )

            dt = time.perf_counter() - t0

            mode = "scene(no-seg)" if (segmap is None) else "segmented(with-seg)"
            print(f"[TIME] depth+K -> pc ({mode}): {dt:.4f} s | pc_full: {pc_full.shape} | segments: {len(pc_segments) if isinstance(pc_segments, dict) else 'N/A'}")
        else:
            print("[TIME] pc_full already provided by input file -> skip reconstruction")

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
            sess,
            pc_full,
            pc_segments=pc_segments,
            local_regions=local_regions_i,
            filter_grasps=filter_grasps_i,
            forward_passes=forward_passes
        )

        # ---- Debug prints ----
        print("pred_grasps_cam type:", type(pred_grasps_cam))
        print("scores type:", type(scores))

        if isinstance(pred_grasps_cam, dict):
            ks = list(pred_grasps_cam.keys())
            print("pred_grasps_cam keys:", ks[:10], "... total:", len(ks))
            if len(ks) > 0:
                k0 = ks[0]
                print("pred_grasps_cam[k0] shape:", np.asarray(pred_grasps_cam[k0]).shape)

        if isinstance(scores, dict):
            ks = list(scores.keys())
            print("scores keys:", ks[:10], "... total:", len(ks))
            if len(ks) > 0:
                k0 = ks[0]
                print("scores[k0] shape:", np.asarray(scores[k0]).shape)

        # Save npz (dict 저장됨: 로드할 때 allow_pickle 필요)
        out_npz = os.path.join(
            "results",
            "predictions_{}".format(os.path.basename(p.replace('png', 'npz').replace('npy', 'npz')))
        )
        np.savez(out_npz, pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)
        print("Saved:", out_npz)

        # 2D overlay 이미지 저장/표시
        if cam_K is None:
            print("cam_K is None -> skip 2D overlay (need intrinsics)")
            continue

        vis = overlay_grasps_2d_on_rgb(
            rgb=rgb,
            segmap=segmap,  # None이면 그냥 rgb 위에만 그리도록 overlay 함수가 처리해야 함
            pred_grasps_cam=pred_grasps_cam,
            scores=scores,
            K=cam_K,
            topk_per_object=topk_per_object,
            score_thresh=score_thresh,
            gripper_width=0.08,
            thickness=2
        )

        if save_png:
            stem = os.path.splitext(os.path.basename(p))[0]
            out_png = os.path.join("results", f"grasps_{stem}.png")
            # cv2는 BGR이라 변환
            cv2.imwrite(out_png, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print("Saved grasp overlay:", out_png)

        if show_fig:
            # segmap이 있으면 같이 확인
            show_image(rgb, segmap, title=f"RGB+Seg: {os.path.basename(p)}")

            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(vis)
            plt.title(f"2D Grasp overlay: {os.path.basename(p)}")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001')
    parser.add_argument('--np_path', default='test_data/*.npy',
                        help='IMPORTANT: wrap glob with quotes, e.g. "test_data/*.npy"')
    parser.add_argument('--png_path', default='')
    parser.add_argument('--K', default=None, help='Flat K: "[fx,0,cx,0,fy,cy,0,0,1]"')
    parser.add_argument('--z_range', default=[0.2, 1.8])
    parser.add_argument('--local_regions', action='store_true', default=False)
    parser.add_argument('--filter_grasps', action='store_true', default=False)
    parser.add_argument('--skip_border_objects', action='store_true', default=False)
    parser.add_argument('--forward_passes', type=int, default=1)
    parser.add_argument('--segmap_id', type=int, default=0)
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[])

    # visualization options
    parser.add_argument('--topk', type=int, default=10, help='object당 표시할 grasp 개수')
    parser.add_argument('--score_thresh', type=float, default=None, help='이 값 미만 score는 표시 안 함')
    parser.add_argument('--no_save_png', action='store_true', help='overlay png 저장 안 함')
    parser.add_argument('--no_show', action='store_true', help='matplotlib 창 띄우지 않음')

    # segmap control
    parser.add_argument('--ignore_segmap', action='store_true',
                        help='segmap이 있어도 무시하고 scene-mode로 실행 (bbox/seg 없이 전체 pc에서 추론)')

    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(
        FLAGS.ckpt_dir,
        batch_size=FLAGS.forward_passes,
        arg_configs=FLAGS.arg_configs
    )

    print(str(global_config))
    print('pid:', os.getpid())

    input_paths = FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path

    inference(
        global_config,
        FLAGS.ckpt_dir,
        input_paths,
        z_range=eval(str(FLAGS.z_range)),
        K=FLAGS.K,
        local_regions=FLAGS.local_regions,
        filter_grasps=FLAGS.filter_grasps,
        segmap_id=FLAGS.segmap_id,
        forward_passes=FLAGS.forward_passes,
        skip_border_objects=FLAGS.skip_border_objects,
        save_png=(not FLAGS.no_save_png),
        show_fig=(not FLAGS.no_show),
        topk_per_object=FLAGS.topk,
        score_thresh=FLAGS.score_thresh,
        ignore_segmap=FLAGS.ignore_segmap,
    )
