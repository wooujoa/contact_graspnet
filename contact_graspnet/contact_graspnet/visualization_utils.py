import numpy as np

# Mayavi는 있으면 쓰고, 없으면 None으로 두고 2D fallback으로 처리
try:
    import mayavi.mlab as mlab
except Exception:
    mlab = None

import matplotlib.pyplot as plt
import cv2

import mesh_utils


# -----------------------------
# Matplotlib: seg overlay 확인용
# -----------------------------
def show_image(rgb, segmap, title="RGB+Seg"):
    """
    Overlay rgb image with segmentation and show via matplotlib.
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()
    # 어떤 backend에서는 window 속성이 없을 수 있음
    try:
        figManager.window.showMaximized()
    except Exception:
        pass

    plt.ion()
    plt.show()

    if rgb is not None:
        plt.imshow(rgb)

    if segmap is not None:
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)

    plt.title(title)
    plt.axis("off")
    plt.draw()
    plt.pause(0.001)


# ------------------------------------
# 2D Grasp overlay (Mayavi 없이도 가능)
# ------------------------------------
def _get_gripper_line_points(gripper_opening: float):
    """
    그리퍼 와이어프레임(7포인트)을 gripper local 좌표계에서 생성.
    opening(벌림 폭)은 x축 방향으로 적용.
    """
    gripper = mesh_utils.create_gripper('panda')
    cp = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()

    mid = 0.5 * (cp[1, :] + cp[2, :])
    line = np.array([
        np.zeros((3,)),  # 0
        mid,             # 1
        cp[1],           # 2
        cp[3],           # 3
        cp[1],           # 4
        cp[2],           # 5
        cp[4],           # 6
    ], dtype=np.float32)

    # opening 적용
    line2 = line.copy()
    line2[2:, 0] = np.sign(line[2:, 0]) * (gripper_opening / 2.0)
    return line2  # (7,3)


def _transform_points(pts: np.ndarray, T: np.ndarray):
    """
    pts: (N,3), T: (4,4)
    -> pts_cam: (N,3) where pts_cam = pts*R^T + t
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return pts @ R.T + t[None, :]


def _project_cam_to_pixel(pts_cam: np.ndarray, K: np.ndarray):
    """
    pts_cam: (N,3), K: (3,3)
    -> pix: (N,2), valid mask
    """
    z = pts_cam[:, 2]
    valid = z > 1e-6
    x = pts_cam[:, 0] / (z + 1e-9)
    y = pts_cam[:, 1] / (z + 1e-9)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    pix = np.stack([u, v], axis=1)
    return pix, valid


def _segmap_to_color(segmap):
    """
    rgb가 없을 때 배경용으로 segmap을 컬러 이미지로 변환.
    """
    seg = segmap.astype(np.float32)
    seg_norm = (seg - seg.min()) / (seg.max() - seg.min() + 1e-9)
    seg_u8 = (seg_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(seg_u8, cv2.COLORMAP_JET)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def overlay_grasps_2d_on_rgb(
    rgb: np.ndarray,
    segmap: np.ndarray,
    pred_grasps_cam,
    scores,
    K: np.ndarray,
    topk_per_object: int = 10,
    score_thresh: float = None,
    gripper_width: float = 0.08,
    thickness: int = 2,
):
    """
    Mayavi 없이 grasp를 2D로 투영해서 RGB 위에 선으로 그려 반환.
    - pred_grasps_cam / scores 가 dict이면 object별 topk를 그림
    - ndarray면 전체에서 topk를 그림
    """

    if rgb is None:
        if segmap is None:
            raise ValueError("rgb도 segmap도 None이라 2D overlay 배경을 만들 수 없습니다.")
        rgb = _segmap_to_color(segmap)

    if K is None:
        raise ValueError("K(cam intrinsics)가 None이라 2D 투영을 할 수 없습니다.")

    img = rgb.copy()
    H, W = img.shape[:2]

    # 7포인트를 잇는 선 연결
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    def draw_one_grasp(T, s, color_bgr):
        pts_local = _get_gripper_line_points(gripper_width)  # opening을 gripper_width로 대체
        pts_cam = _transform_points(pts_local, T)
        pix, valid = _project_cam_to_pixel(pts_cam, K)

        if not np.all(valid):
            return

        pts2 = pix.astype(np.int32)

        # 화면 밖이면 대충 스킵(너무 aggressive하면 보이는 것도 날아가서 buffer 줌)
        if (pts2[:, 0] < -200).all() or (pts2[:, 0] > W + 200).all() or (pts2[:, 1] < -200).all() or (pts2[:, 1] > H + 200).all():
            return

        for a, b in edges:
            x1, y1 = pts2[a]
            x2, y2 = pts2[b]
            cv2.line(img, (x1, y1), (x2, y2), color_bgr, thickness, lineType=cv2.LINE_AA)

    # dict 케이스: object별로
    if isinstance(pred_grasps_cam, dict):
        keys = list(pred_grasps_cam.keys())

        for i, k in enumerate(keys):
            G = np.asarray(pred_grasps_cam[k])
            S = np.asarray(scores[k]) if isinstance(scores, dict) else None

            if G.size == 0:
                continue
            if S is None or S.size == 0:
                # scores가 dict가 아니면 그냥 순서대로 일부만
                order = np.arange(min(topk_per_object, len(G)))
            else:
                order = np.argsort(S)[::-1]
                if score_thresh is not None:
                    order = [idx for idx in order if S[idx] >= score_thresh]
                order = order[:topk_per_object]

            # object별 색 단순 변화
            if len(keys) <= 1:
                color_bgr = (0, 255, 0)
            else:
                # i에 따라 색 변화
                r = int(50 + 205 * (i / max(1, len(keys) - 1)))
                b = int(50 + 205 * ((len(keys) - 1 - i) / max(1, len(keys) - 1)))
                color_bgr = (b, 255, r)

            for idx in order:
                draw_one_grasp(G[idx], float(S[idx]) if S is not None else 0.0, color_bgr)

        return img

    # ndarray 케이스: 전체
    G = np.asarray(pred_grasps_cam)
    S = np.asarray(scores) if scores is not None else None

    if S is not None and S.size == len(G):
        order = np.argsort(S)[::-1]
        if score_thresh is not None:
            order = [idx for idx in order if S[idx] >= score_thresh]
        order = order[:topk_per_object]
    else:
        order = np.arange(min(topk_per_object, len(G)))

    for idx in order:
        draw_one_grasp(G[idx], float(S[idx]) if S is not None else 0.0, (0, 255, 0))

    return img


# ------------------------------------
# 기존 visualize_grasps: Mayavi 있으면 3D, 없으면 2D로 fallback
# ------------------------------------
def visualize_grasps(
    full_pc,
    pred_grasps_cam,
    scores,
    plot_opencv_cam=False,
    pc_colors=None,
    gripper_openings=None,
    gripper_width=0.08,
    rgb=None,
    segmap=None,
    K=None,
    save_path=None,
    topk_per_object=10,
    score_thresh=None,
):
    """
    - Mayavi 있으면 3D로 띄움 (원래 코드 흐름 유지)
    - Mayavi 없으면 2D overlay(이미지)로 대체해서 보여주고/저장함
    """

    # Mayavi 없으면 2D fallback
    if mlab is None:
        if rgb is None and segmap is None:
            raise RuntimeError("Mayavi가 없고, 2D overlay를 위한 rgb/segmap도 없습니다.")
        if K is None:
            raise RuntimeError("Mayavi가 없고, 2D overlay를 위한 K도 없습니다.")

        vis = overlay_grasps_2d_on_rgb(
            rgb=rgb,
            segmap=segmap,
            pred_grasps_cam=pred_grasps_cam,
            scores=scores,
            K=K,
            topk_per_object=topk_per_object,
            score_thresh=score_thresh,
            gripper_width=gripper_width,
            thickness=2
        )

        if save_path is not None:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        plt.figure()
        plt.imshow(vis)
        plt.title("Grasp 2D overlay (fallback)")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)
        return

    # ---- 여기부터는 Mayavi 3D (원하면 그대로 두고, 안 쓰면 inference에서 호출 안 하면 됨) ----
    print('Visualizing...takes time (Mayavi)')
    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)

    # Point cloud
    if full_pc is not None and len(full_pc) > 0:
        if pc_colors is None:
            mlab.points3d(full_pc[:, 0], full_pc[:, 1], full_pc[:, 2], color=(0.3, 0.3, 0.3),
                          scale_factor=0.0018, mode='2dsquare')
        else:
            mlab.points3d(full_pc[:, 0], full_pc[:, 1], full_pc[:, 2], scale_factor=0.0018, mode='2dsquare')

    # Grasps (간단히만: dict일 때 object별로)
    def draw_wire(T, color=(0, 1, 0)):
        pts_local = _get_gripper_line_points(gripper_width)
        pts_cam = _transform_points(pts_local, T)
        for a, b in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]:
            p1 = pts_cam[a]
            p2 = pts_cam[b]
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=color, tube_radius=0.0008, opacity=1.0)

    if isinstance(pred_grasps_cam, dict):
        keys = list(pred_grasps_cam.keys())
        for i, k in enumerate(keys):
            G = np.asarray(pred_grasps_cam[k])
            S = np.asarray(scores[k]) if isinstance(scores, dict) else None
            if G.size == 0:
                continue
            if S is not None and S.size == len(G):
                order = np.argsort(S)[::-1][:topk_per_object]
            else:
                order = np.arange(min(topk_per_object, len(G)))
            col = (0, 1, 0) if len(keys) == 1 else (i / max(1, len(keys) - 1), 1, 0)
            for idx in order:
                draw_wire(G[idx], color=col)
    else:
        G = np.asarray(pred_grasps_cam)
        for idx in range(min(topk_per_object, len(G))):
            draw_wire(G[idx], color=(0, 1, 0))

    mlab.show()
