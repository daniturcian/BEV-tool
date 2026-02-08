import os
import math
import numpy as np
import cv2


# -----------------------------
# KITTI helpers
# -----------------------------
def load_kitti_velo_bin(bin_path: str) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]  # x,y,z

def parse_kitti_calib_file(path: str) -> dict:
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            val = val.strip()
            parts = val.split()
            try:
                nums = np.array([float(x) for x in parts], dtype=np.float64)
                data[key] = nums
            except ValueError:
                data[key] = val
    return data

def camera_origin_in_velo(R_v2c: np.ndarray, T_v2c: np.ndarray) -> np.ndarray:
    # p_cam = R p_velo + T ; camera origin in cam is [0,0,0]
    # => 0 = R*C_velo + T => C_velo = -R^T T
    return (-R_v2c.T @ T_v2c).reshape(3,)

# -----------------------------
# RANSAC plane fit
# -----------------------------
def ransac_plane(points: np.ndarray, n_iter=200, dist_thresh=0.12, seed=0):
    """
    Plane: ax+by+cz+d=0 with ||[a,b,c]||=1
    Returns (a,b,c,d), inlier_mask
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if N < 3:
        raise ValueError("Not enough points for plane fit.")

    best_inliers = None
    best_count = 0
    best_plane = None

    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            continue
        n = n / norm_n
        d = -float(np.dot(n, p1))

        dist = np.abs(points @ n + d)  # because n normalized
        inliers = dist < dist_thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = (n[0], n[1], n[2], d)

    if best_plane is None:
        raise RuntimeError("RANSAC failed.")

    # refine with SVD on inliers
    P = points[best_inliers]
    centroid = P.mean(axis=0)
    Q = P - centroid
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n = n / np.linalg.norm(n)
    d = -float(np.dot(n, centroid))

    # normalize sign for consistency (optional)
    # keep n roughly pointing "up" in velodyne by enforcing c>0 (z component)
    if n[2] < 0:
        n = -n
        d = -d

    return (float(n[0]), float(n[1]), float(n[2]), float(d)), best_inliers

def estimate_height_pitch_roll_from_lidar(points_velo: np.ndarray,
                                         R_v2c: np.ndarray,
                                         T_v2c: np.ndarray,
                                         roi=(3, 30, -10, 10, -3, 1),
                                         ransac_iters=250,
                                         dist_thresh=0.12):
    """
    Estimate ground plane in velodyne and compute camera height + pitch/roll.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = roi
    m = (
        (points_velo[:, 0] >= x_min) & (points_velo[:, 0] <= x_max) &
        (points_velo[:, 1] >= y_min) & (points_velo[:, 1] <= y_max) &
        (points_velo[:, 2] >= z_min) & (points_velo[:, 2] <= z_max)
    )
    P = points_velo[m]
    if P.shape[0] < 800:
        raise RuntimeError(f"Too few ROI points for plane fit: {P.shape[0]}")

    a, b, c, d = ransac_plane(P, n_iter=ransac_iters, dist_thresh=dist_thresh)[0]
    n_velo = np.array([a, b, c], dtype=np.float64)  # already unit
    d_velo = float(d)

    # camera center in velodyne coordinates
    C_velo = camera_origin_in_velo(R_v2c, T_v2c)

    # signed distance camera to plane
    dist_signed = float(n_velo @ C_velo + d_velo)
    height = abs(dist_signed)

    # normal in camera coords
    n_cam = (R_v2c @ n_velo.reshape(3, 1)).reshape(3,)
    n_cam = n_cam / np.linalg.norm(n_cam)

    # KITTI cam coords (convention): x right, y down, z forward
    # for a flat road, normal should be roughly [0, -1, 0] in cam coords
    pitch = math.atan2(-n_cam[2], -n_cam[1])
    roll  = math.atan2( n_cam[0], -n_cam[1])

    return height, pitch, roll, (a, b, c, d)

# -----------------------------
# BEV remap map precomputation
# -----------------------------
def rpy_to_R(roll, pitch, yaw=0.0):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def transform_plane(R, T, n, d):
    n2 = R @ n
    d2 = float(d - n2.T @ T.reshape(3,))
    # normalize
    s = np.linalg.norm(n2)
    n2 = n2 / s
    d2 = d2 / s
    return n2, d2

def precompute_bev_maps_from_plane(K, n_cam, d_cam,
                                   x_min=0.0, x_max=30.0,
                                   y_min=-10.0, y_max=10.0,
                                   ppm=20):
    n = n_cam / np.linalg.norm(n_cam)

    # define BEV axes on the plane using camera axes projected to plane
    z_axis = np.array([0.0, 0.0, 1.0])  # forward in KITTI cam
    x_axis = np.array([1.0, 0.0, 0.0])  # right in KITTI cam

    fwd = z_axis - (z_axis @ n) * n
    right = x_axis - (x_axis @ n) * n
    fwd /= np.linalg.norm(fwd)
    right /= np.linalg.norm(right)
    left = -right

    # closest point on plane to camera origin
    X0 = -d_cam * n

    bev_w = int(round((y_max - y_min) * ppm))
    bev_h = int(round((x_max - x_min) * ppm))

    xs = np.linspace(x_max, x_min, bev_h)  # far -> top
    ys = np.linspace(y_min, y_max, bev_w)

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = (X0.reshape(1,3)
           + X.reshape(-1,1) * fwd.reshape(1,3)
           + Y.reshape(-1,1) * left.reshape(1,3))

    z = pts[:, 2]
    valid = z > 1e-6

    u = np.full((pts.shape[0],), -1.0, dtype=np.float32)
    v = np.full((pts.shape[0],), -1.0, dtype=np.float32)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u[valid] = (fx * (pts[valid,0] / pts[valid,2]) + cx).astype(np.float32)
    v[valid] = (fy * (pts[valid,1] / pts[valid,2]) + cy).astype(np.float32)

    map_x = u.reshape(bev_h, bev_w)
    map_y = v.reshape(bev_h, bev_w)
    return map_x, map_y


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # ADAPTEAZA doar ROOT la proiectul tau
    ROOT = "dataset/2011_09_26/2011_09_26_drive_0002_sync"

    cam_to_cam = os.path.join(ROOT, "calib_cam_to_cam.txt")
    velo_to_cam = os.path.join(ROOT, "calib_velo_to_cam.txt")

    img_dir  = os.path.join(ROOT, "image_02", "data")
    velo_dir = os.path.join(ROOT, "velodyne_points", "data")
    out_dir  = os.path.join(ROOT, "bev_image_02")
    os.makedirs(out_dir, exist_ok=True)

    # 1 calib
    camcal = parse_kitti_calib_file(cam_to_cam)
    velocal = parse_kitti_calib_file(velo_to_cam)

    K = camcal["K_02"].reshape(3, 3)
    D = camcal["D_02"].reshape(-1)

    R_v2c = velocal["R"].reshape(3, 3)
    T_v2c = velocal["T"].reshape(3, 1)

    # 2 liste frame-uri (presupunem aceleasi nume)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")])
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir}")

    # KITTI lidar files are .bin with same stem
    def to_bin_name(img_name: str) -> str:
        stem = os.path.splitext(img_name)[0]
        return stem + ".bin"

    # 3 estimeaza height/pitch/roll din primele N frame-uri (median)
    N_EST = min(50, len(img_files))  # poti creste la 200 pentru mai stabil
    heights, pitches, rolls = [], [], []

    roi = (3, 30, -10, 10, -3, 1)  # x_min,x_max,y_min,y_max,z_min,z_max

    for i in range(N_EST):
        bin_path = os.path.join(velo_dir, to_bin_name(img_files[i]))
        if not os.path.exists(bin_path):
            continue
        pts = load_kitti_velo_bin(bin_path)
        try:
            h, p, r, _plane = estimate_height_pitch_roll_from_lidar(
                pts, R_v2c, T_v2c, roi=roi, ransac_iters=250, dist_thresh=0.12
            )
            heights.append(h); pitches.append(p); rolls.append(r)
            (a,b,c,d) = _plane
        except Exception:
            continue
    
    n_v = np.array([a,b,c], dtype=np.float64)
    d_v = float(d)

    # velo -> cam0 (din calib_velo_to_cam.txt)
    n_c0, d_c0 = transform_plane(R_v2c, T_v2c, n_v, d_v)

    # cam0 -> cam2 (din calib_cam_to_cam.txt: R_02, T_02)
    R_02 = camcal["R_02"].reshape(3,3)
    T_02 = camcal["T_02"].reshape(3,1)
    n_c2, d_c2 = transform_plane(R_02, T_02, n_c0, d_c0)
    if len(heights) < 10:
        raise RuntimeError("Could not estimate stable ground plane. Try adjusting ROI/dist_thresh.")

    h_med = float(np.median(heights))
    p_med = float(np.median(pitches))
    r_med = float(np.median(rolls))

    print(f"[EST] camera height (m): {h_med:.3f}")
    print(f"[EST] pitch (deg): {p_med * 180/math.pi:.3f}")
    print(f"[EST] roll  (deg): {r_med * 180/math.pi:.3f}")

    # 4 precompute undistort maps + BEV maps
    # (optional) stabilize K with optimal new camera matrix
    # K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (1242, 375), 0)
    # here we keep original K for simplicity
    K_new = K.copy()

    # Precompute undistort map (for image size from first frame)
    first = cv2.imread(os.path.join(img_dir, img_files[0]))
    if first is None:
        raise RuntimeError("Failed to read first image.")
    H0, W0 = first.shape[:2]

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_new, (W0, H0), cv2.CV_32FC1)

    # BEV settings (poti schimba)
    x_min, x_max = 0.0, 30.0
    y_min, y_max = -10.0, 10.0
    ppm = 20                

    bev_map_x, bev_map_y = precompute_bev_maps_from_plane(K_new, n_c2, d_c2,
                                                      x_min=0, x_max=30,
                                                      y_min=-10, y_max=10,
                                                      ppm=20)
    print("map_x range:", np.nanmin(bev_map_x), np.nanmax(bev_map_x))
    print("map_y range:", np.nanmin(bev_map_y), np.nanmax(bev_map_y))
    # run all frames
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)            
        if img is None:
            continue

        # undistort
        img_u = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        H0, W0 = img_u.shape[:2]
        inside = (bev_map_x >= 0) & (bev_map_x < W0) & (bev_map_y >= 0) & (bev_map_y < H0)
        print("inside ratio:", inside.mean())

        # BEV
        bev = cv2.remap(img_u, bev_map_x, bev_map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow('img1', bev)
        
        cv2.waitKey(0)

        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, bev)

    print("Done. Saved BEV to:", out_dir)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
