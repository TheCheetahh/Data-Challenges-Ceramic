import numpy as np
from svgpathtools import svg2paths2, Path
from scipy.spatial import cKDTree
import io
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

# --------------------------------------------------
# ICP caches (cleared explicitly at safe boundaries)
# --------------------------------------------------
_ORDER_CACHE = {}
_SIGNED_WIDTH_CACHE = {}

def _clear_icp_caches():
    _ORDER_CACHE.clear()
    _SIGNED_WIDTH_CACHE.clear()

def prepare_icp_geometry_from_svg_string(svg_string, n_points):
    """
    Parse SVG string → sampled, centered outline points + avg width
    """
    paths, _, _ = svg2paths2(io.StringIO(svg_string))
    if not paths:
        raise ValueError("No paths in SVG")

    merged = Path(*paths)
    length = merged.length()
    if length == 0:
        raise ValueError("Zero-length SVG path")

    ts = np.linspace(0, length, n_points)
    pts = np.array([merged.point(merged.ilength(t)) for t in ts])
    pts = np.column_stack([pts.real, pts.imag])

    # center
    pts -= pts.mean(axis=0)

    avg_width = compute_average_width(pts)
    if avg_width is None or avg_width == 0:
        raise ValueError("Invalid average width")

    # width normalization
    pts /= avg_width

    return pts, avg_width

def compute_centerline(pts, slice_frac=0.5, tol=0.03):
    mean = pts.mean(axis=0)
    X = pts - mean

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    main_dir = Vt[0]
    width_dir = Vt[1]

    s = X @ main_dir
    w = X @ width_dir

    s0 = s.min() + slice_frac * (s.max() - s.min())

    band = np.abs(s - s0) < tol * (s.max() - s.min())
    if np.sum(band) < 5:
        return None

    w_slice = w[band]

    w_min = w_slice.min()
    w_max = w_slice.max()

    p1 = mean + s0 * main_dir + w_min * width_dir
    p2 = mean + s0 * main_dir + w_max * width_dir

    return np.vstack([p1, p2])

def compute_average_width(pts):
    seg = compute_centerline(pts)
    if seg is not None:
        width = float(np.linalg.norm(seg[1] - seg[0]))
        if width > 0:
            return width

    # --- fallback: PCA-based global width ---
    X = pts - pts.mean(axis=0)
    _, S, _ = np.linalg.svd(X, full_matrices=False)

    # second singular value ≈ half-width scale
    fallback_width = 2.0 * S[1] / np.sqrt(len(pts))

    if fallback_width > 0:
        return float(fallback_width)

    return None


def rail_aware_correspondences(src, dst):
    """
    Compute ICP correspondences where:
    - left src points only match left dst points
    - right src points only match right dst points
    """
    src_w = signed_width_coordinate(src)
    dst_w = signed_width_coordinate(dst)

    src_pos = src[src_w >= 0]
    src_neg = src[src_w < 0]

    dst_pos = dst[dst_w >= 0]
    dst_neg = dst[dst_w < 0]

    matched_src = []
    matched_dst = []

    if len(src_pos) > 0 and len(dst_pos) > 0:
        tree_p = cKDTree(dst_pos)
        _, idx_p = tree_p.query(src_pos)
        matched_src.append(src_pos)
        matched_dst.append(dst_pos[idx_p])

    if len(src_neg) > 0 and len(dst_neg) > 0:
        tree_n = cKDTree(dst_neg)
        _, idx_n = tree_n.query(src_neg)
        matched_src.append(src_neg)
        matched_dst.append(dst_neg[idx_n])

    if not matched_src:
        return None, None

    return (
        np.vstack(matched_src),
        np.vstack(matched_dst)
    )


def run_icp(source_pts, target_pts,
            iters=30,
            max_total_deg=2.0,
            max_scale_step=0.002):

    src = source_pts.copy()
    dst = target_pts.copy()

    # initial top-point alignment
    src_top = src[np.argmin(src[:, 1])]
    dst_top = dst[np.argmin(dst[:, 1])]
    src += (dst_top - src_top)

    total_angle = 0.0
    best_err = np.inf

    for _ in range(iters):
        matched_src, matched_dst = rail_aware_correspondences(src, dst)
        if matched_src is None:
            break

        mu_src = matched_src.mean(0)
        mu_dst = matched_dst.mean(0)

        X = matched_src - mu_src
        Y = matched_dst - mu_dst

        U, S, Vt = np.linalg.svd(X.T @ Y)
        R = Vt.T @ U.T

        scale = np.sum(S) / np.sum(X ** 2) if np.sum(X ** 2) > 0 else 1.0
        scale = np.clip(scale, 1 - max_scale_step, 1 + max_scale_step)

        angle = np.arctan2(R[1, 0], R[0, 0])
        angle_deg = np.degrees(angle)

        remaining = max_total_deg - abs(total_angle)
        if remaining <= 0:
            break

        angle_deg = np.clip(angle_deg, -remaining, remaining)
        angle = np.radians(angle_deg)

        c, s = np.cos(angle), np.sin(angle)
        R_limited = np.array([[c, -s], [s, c]])

        total_angle += angle_deg

        t = mu_dst - scale * (R_limited @ mu_src)
        src = scale * (R_limited @ src.T).T + t

        err = np.mean(np.linalg.norm(matched_src - matched_dst, axis=1))
        if err < best_err - 1e-6:
            best_err = err
        else:
            break

    return float(best_err), src

def signed_width_coordinate(pts):
    key = (pts.shape[0], pts.tobytes())
    if key in _SIGNED_WIDTH_CACHE:
        return _SIGNED_WIDTH_CACHE[key]

    mean = pts.mean(axis=0)
    X = pts - mean

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    main_dir = Vt[0]
    width_dir = np.array([-main_dir[1], main_dir[0]])

    w = X @ width_dir
    _SIGNED_WIDTH_CACHE[key] = w
    return w


def count_rail_clusters_1d(pts, gap_ratio=0.15, min_cluster_size=20):
    """
    Count clusters along width direction without sklearn.

    Strategy:
    - Project to signed width coordinate
    - Sort values
    - Detect large gaps
    - Count resulting clusters
    """

    if len(pts) < min_cluster_size:
        return 0

    w = signed_width_coordinate(pts)
    w_sorted = np.sort(w)

    # Compute consecutive gaps
    gaps = np.diff(w_sorted)

    if len(gaps) == 0:
        return 0

    # Threshold relative to overall width span
    total_span = w_sorted[-1] - w_sorted[0]
    if total_span <= 1e-8:
        return 0

    gap_threshold = gap_ratio * total_span

    # Find where large separations occur
    split_indices = np.where(gaps > gap_threshold)[0]

    # Number of clusters = number of splits + 1
    cluster_count = len(split_indices) + 1

    return cluster_count

def adjust_bbox_to_include_split_rails(reference_pts, bbox_min, bbox_max,
                                       gap_ratio=0.15, min_cluster_size=20):
    """
    If the bbox splits a rail into two parts (causing >2 rail clusters),
    expand the bbox to fully include the split rail(s).
    """

    def inside_bbox(pts):
        return np.all((pts >= bbox_min) & (pts <= bbox_max), axis=1)

    ref_inside = reference_pts[inside_bbox(reference_pts)]

    # If rails are already acceptable, do nothing
    rail_count = count_rail_clusters_1d(
        ref_inside,
        gap_ratio=gap_ratio,
        min_cluster_size=min_cluster_size
    )
    if rail_count <= 2:
        return bbox_min, bbox_max

    # Project all points to width coordinate
    w_all = signed_width_coordinate(reference_pts)

    # Cluster by width using the same 1D logic
    order = np.argsort(w_all)
    w_sorted = w_all[order]
    pts_sorted = reference_pts[order]

    gaps = np.diff(w_sorted)
    total_span = w_sorted[-1] - w_sorted[0]
    gap_threshold = gap_ratio * total_span

    split_idxs = np.where(gaps > gap_threshold)[0]
    cluster_bounds = np.concatenate(([0], split_idxs + 1, [len(w_sorted)]))

    # For each rail cluster, check if bbox cuts it
    for i in range(len(cluster_bounds) - 1):
        a, b = cluster_bounds[i], cluster_bounds[i + 1]
        rail_pts = pts_sorted[a:b]

        inside = inside_bbox(rail_pts)
        if np.any(inside) and not np.all(inside):
            # Rail is split → expand bbox to include full rail
            bbox_min = np.minimum(bbox_min, rail_pts.min(axis=0))
            bbox_max = np.maximum(bbox_max, rail_pts.max(axis=0))

    return bbox_min, bbox_max

def find_icp_matches(
    target_pts,
    reference_dict,
    icp_params,
    top_k=20
):
    """
    reference_dict: { sample_id: reference_pts }
    """
    _clear_icp_caches()
    results = []

    for ref_id, ref_pts in reference_dict.items():
        err, aligned = run_icp(
            target_pts,
            ref_pts,
            iters=icp_params["iters"],
            max_total_deg=icp_params["max_total_deg"],
            max_scale_step=icp_params["max_scale_step"]
        )
        score, bbox = icp_score(ref_pts, aligned, ref_id=ref_id)
        results.append((ref_id, score, aligned, bbox))

    results.sort(key=lambda x: x[1])
    return [
        {
            "id": rid,
            "distance": float(score),
            "aligned_target": aligned.tolist(),
            "bbox": bbox
        }
        for rid, score, aligned, bbox in results[:top_k]
    ]

def discrete_curvature(pts):
    """
    Signed discrete curvature.
    Positive = bending one way
    Negative = bending the other way
    """

    if pts is None or len(pts) < 3:
        return None

    p_prev = pts[:-2]
    p_mid  = pts[1:-1]
    p_next = pts[2:]

    v1 = p_mid - p_prev
    v2 = p_next - p_mid

    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)

    # 2D cross product (scalar)
    cross = v1_norm[:, 0] * v2_norm[:, 1] - v1_norm[:, 1] * v2_norm[:, 0]

    # signed curvature proxy
    curvature = cross

    return curvature

def discrete_direction(pts):
    """
    Tangent direction angles (radians) along ordered rail points.
    Returned length = len(pts) - 1
    """
    if pts is None or len(pts) < 2:
        return None

    diffs = np.diff(pts, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    return angles

def mean_angular_error(a, b):
    """
    Mean absolute wrapped angular difference (radians).
    """
    delta = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.mean(np.abs(delta))

def extract_tip_region(pts, percent=0.15):
    """
    Returns the top X% of points (tip region).
    Assumes smaller Y = higher in image (after invert_yaxis logic).
    """
    if len(pts) < 10:
        return None

    y = pts[:, 1]
    cutoff = np.percentile(y, percent * 100)

    tip = pts[y <= cutoff]

    if len(tip) < 5:
        return None

    # sort along main direction for stability
    order = np.argsort(tip[:, 1])
    return tip[order]

def cut_bbox_by_line(points, p0, p1, keep_side=1):
    """
    Returns boolean mask:
    True = point is on the kept side of the line p0→p1
    keep_side = +1 or -1
    """
    # Line normal (2D cross product sign)
    v = p1 - p0
    w = points - p0
    cross = v[0] * w[:, 1] - v[1] * w[:, 0]
    return keep_side * cross >= 0

def order_points_by_arclength(pts, k=6):
    key = (pts.shape[0], pts.tobytes())
    if key in _ORDER_CACHE:
        return _ORDER_CACHE[key]

    from scipy.spatial import cKDTree
    import heapq

    n = len(pts)
    tree = cKDTree(pts)
    neighbors = tree.query(pts, k=k+1)[1][:, 1:]

    graph = [[] for _ in range(n)]
    for i in range(n):
        pi = pts[i]
        for j in neighbors[i]:
            if j > i:
                d = np.linalg.norm(pi - pts[j])
                graph[i].append((j, d))
                graph[j].append((i, d))

    def bfs_farthest(start):
        dist = np.full(n, -1.0)
        prev = np.full(n, -1, dtype=int)
        q = deque([start])
        dist[start] = 0.0

        while q:
            u = q.popleft()
            for v, w in graph[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    q.append(v)

        end = np.argmax(dist)
        return end, prev

    a, _ = bfs_farthest(0)
    b, prev = bfs_farthest(a)

    path = []
    cur = b
    while cur != -1:
        path.append(cur)
        cur = prev[cur]


    ordered = pts[path]
    _ORDER_CACHE[key] = ordered
    return ordered

def bbox_polygon_clipped_by_line(bbox_min, bbox_max, p0, p1):
    """
    Returns a polygon (Nx2 array) representing the bbox
    clipped by the rail line p0→p1.
    The kept side is chosen automatically.
    """

    rect = np.array([
        [bbox_min[0], bbox_min[1]],
        [bbox_max[0], bbox_min[1]],
        [bbox_max[0], bbox_max[1]],
        [bbox_min[0], bbox_max[1]],
    ])

    # --- line normal test ---
    v = p1 - p0

    def signed_side(pt):
        w = pt - p0
        return v[0] * w[1] - v[1] * w[0]

    # Decide which side to keep using bbox center
    center = 0.5 * (bbox_min + bbox_max)
    keep_positive = signed_side(center) >= 0

    def inside(pt):
        return (signed_side(pt) >= 0) == keep_positive

    clipped = []

    for i in range(len(rect)):
        a = rect[i]
        b = rect[(i + 1) % len(rect)]

        a_in = inside(a)
        b_in = inside(b)

        if a_in:
            clipped.append(a)

        if a_in ^ b_in:
            da = a - p0
            db = b - p0

            denom = (
                v[1] * (b[0] - a[0]) -
                v[0] * (b[1] - a[1])
            )

            if abs(denom) > 1e-12:
                t = (v[0] * da[1] - v[1] * da[0]) / denom
                inter = a + t * (b - a)
                clipped.append(inter)

    return np.array(clipped)


def icp_score(reference_pts,
              aligned_target_pts,
              ref_id=None):
    """
    ICP score using:
    - Bounding box around entire aligned target
    - Reference cropped to that box
    - Rail-wise curvature comparison

    Always returns:
        (score, bbox)   or   (np.inf, None)
    """
    if False:
        print("\n--- ICP SCORE DEBUG ---")
        print("Template:", ref_id)
    # --------------------------------------------------
    # Basic safety checks
    # --------------------------------------------------
    if len(reference_pts) < 20 or len(aligned_target_pts) < 20:
        return np.inf, None

    # --------------------------------------------------
    # 1) Bounding box around aligned target,
    #     extended to reference left-most and top-most
    # --------------------------------------------------

    # Target extents
    tgt_min = aligned_target_pts.min(axis=0)
    tgt_max = aligned_target_pts.max(axis=0)

    # Reference extents
    ref_min = reference_pts.min(axis=0)
    ref_max = reference_pts.max(axis=0)

    # Start from target bbox
    bbox_min = tgt_min.copy()
    bbox_max = tgt_max.copy()

    # Force bbox to include reference left-most (X) and top-most (Y)
    bbox_min[0] = min(bbox_min[0], ref_min[0])  # left-most
    bbox_min[1] = min(bbox_min[1], ref_min[1])  # top-most

    # Optional expansion (only outward)
    expand_factor = 1.5

    width = bbox_max[0] - bbox_min[0]
    height = bbox_max[1] - bbox_min[1]

    expand_x = (expand_factor - 1.0) * width / 2.0
    expand_y = (expand_factor - 1.0) * height / 2.0

    bbox_min -= np.array([expand_x, expand_y])
    bbox_max += np.array([expand_x, expand_y])

    # --- Adjust bbox if it splits a rail ---
    bbox_min, bbox_max = adjust_bbox_to_include_split_rails(
        reference_pts,
        bbox_min,
        bbox_max
    )

    # --- Curvature-safe rail endpoint detection ---
    ordered = order_points_by_arclength(aligned_target_pts)

    n = len(ordered)
    k = max(5, int(0.15 * n))   # tail region near the end

    line20_pts, _ = make_points_on_target_rail(aligned_target_pts)
    line_p0 = line20_pts[0]
    line_p1 = line20_pts[-1]

    bbox_poly = bbox_polygon_clipped_by_line(
        bbox_min,
        bbox_max,
        line_p0,
        line_p1
    )

    from matplotlib.path import Path as Pathh
    bbox_path = Pathh(bbox_poly)

    def inside_bbox(points):
        return bbox_path.contains_points(points)

    ref_box = reference_pts[inside_bbox(reference_pts)]
    tgt_box = aligned_target_pts  # full target

    # --------------------------------------------------
    # Reject if reference contains multiple rails in bbox
    # --------------------------------------------------
    ref_rail_count = count_rail_clusters_1d(
        ref_box,
        gap_ratio=0.15,
        min_cluster_size=20
    )

    if ref_rail_count != 1:
        print(f"[{ref_id}] Reference rails in bbox:", ref_rail_count)
        # Reference bbox contains multiple (or zero) rails
        return np.inf, None

    if len(ref_box) < 20:
        print(f"[{ref_id}] Reject: ref_box too small:", len(ref_box))
        return np.inf, None

    # --------------------------------------------------
    # 2) Single-rail assumption inside bbox
    # --------------------------------------------------
    tgt_rail = tgt_box
    ref_rail = ref_box

    tgt_line_pts, _ = make_points_on_target_rail(tgt_rail, n_points=100)
    ref_line_pts, _ = make_points_on_reference_rail(
        reference_pts,
        bbox_poly,
        n_points=100
    )

    if tgt_line_pts is None or ref_line_pts is None:
        return np.inf, None
    # --------------------------------------------------
    # 3) Curvature + direction rail similarity score
    # --------------------------------------------------

    # --- curvature ---
    curv_tgt = discrete_curvature(tgt_line_pts)
    curv_ref = discrete_curvature(ref_line_pts)

    if curv_tgt is None or curv_ref is None:
        return np.inf, None

    m = min(len(curv_tgt), len(curv_ref))
    curv_tgt = curv_tgt[:m]
    curv_ref = curv_ref[:m]

    curvature_error = np.mean(np.abs(curv_tgt - curv_ref))

    # --- direction ---
    dir_tgt = discrete_direction(tgt_line_pts)
    dir_ref = discrete_direction(ref_line_pts)

    if dir_tgt is None or dir_ref is None:
        return np.inf, None

    k = min(len(dir_tgt), len(dir_ref))
    dir_tgt = dir_tgt[:k]
    dir_ref = dir_ref[:k]

    direction_error = mean_angular_error(dir_tgt, dir_ref)

    # --- weighted combination ---
    W_CURV = 1.0
    W_DIR  = 0.1   # start conservative; increase if curvature dominates too much

    score = W_CURV * curvature_error + W_DIR * direction_error
    return float(score), bbox_poly


def clip_line_to_bbox(p0, p1, bbox_min, bbox_max):
    """
    Liang–Barsky line clipping.
    Returns (q0, q1) or None if no intersection.
    """
    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [
        x0 - bbox_min[0],
        bbox_max[0] - x0,
        y0 - bbox_min[1],
        bbox_max[1] - y0
    ]

    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)

    if u1 > u2:
        return None

    q0 = np.array([x0 + u1 * dx, y0 + u1 * dy])
    q1 = np.array([x0 + u2 * dx, y0 + u2 * dy])
    return q0, q1

def make_points_on_target_rail(target_pts, n_points=100):
    """
    Sample points along a single continuous rail
    using geodesic ordering
    """
    ordered = order_points_by_arclength(target_pts)

    if len(ordered) < n_points:
        return None, None

    idx = np.linspace(0, len(ordered) - 1, n_points).astype(int)
    line_pts = ordered[idx]

    return line_pts, np.arange(len(line_pts))

def make_points_on_reference_rail(reference_pts, bbox_poly, n_points=100):
    """
    Sample n_points along the reference rail,
    restricted to points inside bbox_poly.
    """

    from matplotlib.path import Path as Pathh

    bbox_path = Pathh(bbox_poly)
    inside = bbox_path.contains_points(reference_pts)
    ref_inside = reference_pts[inside]

    if len(ref_inside) < n_points:
        return None, None

    ordered = order_points_by_arclength(ref_inside)

    idx = np.linspace(0, len(ordered) - 1, n_points).astype(int)
    line_pts = ordered[idx]

    return line_pts, idx

def extend_line(p0, p1, scale=1000.0):
    """
    Extend line p0→p1 in both directions by a large factor.
    """
    v = p1 - p0
    v /= (np.linalg.norm(v) + 1e-12)

    q0 = p0 - scale * v
    q1 = p0 + scale * v
    return q0, q1


def plot_icp_overlap(
    target_pts,
    aligned_target_pts,
    reference_pts,
    bbox=None
):
    """
    Shows:
    - Full reference (light gray)
    - Reference inside bbox split into left/right rails
    - Full aligned target split into rails
    - Bounding box
    - Average width lines
    """
    _clear_icp_caches()
    fig, ax = plt.subplots(figsize=(6, 6))
    # --- Precompute 20-point rail line once ---
    line20_pts, _ = make_points_on_target_rail(aligned_target_pts, n_points=20)

    line100_tgt, _ = make_points_on_target_rail(
        aligned_target_pts,
        n_points=100
    )

    line100_ref, _ = (
        make_points_on_reference_rail(reference_pts, bbox, n_points=100)
        if bbox is not None else (None, None)
    )
    if line20_pts is None:
        return None

    # --------------------------------------------------
    # 2) Determine bbox mask
    # --------------------------------------------------
    from matplotlib.path import Path as Pathh

    if bbox is not None:
        bbox_poly = bbox
        bbox_path = Pathh(bbox_poly)
        ref_mask = bbox_path.contains_points(reference_pts)
    else:
        ref_mask = np.ones(len(reference_pts), dtype=bool)
    ref_box = reference_pts[ref_mask]

    # --------------------------------------------------
    # 3) Plot full reference faint
    # --------------------------------------------------
    ax.scatter(
        reference_pts[:, 0],
        reference_pts[:, 1],
        s=6, color="lightgray",
        label="Reference (full)"
    )

    # --------------------------------------------------
    # 4) Highlight reference rails INSIDE bbox
    # --------------------------------------------------
    if bbox is not None:

        ax.scatter(
            ref_box[:, 0], ref_box[:, 1],
            s=10, color="blue",
            label="Reference (used, single rail)"
        )

    """if line100_ref is not None:
        ax.scatter(
            line100_ref[:, 0],
            line100_ref[:, 1],
            s=28,
            facecolors="none",
            edgecolors="green",
            linewidths=1.5,
            label="Reference rail (100 pts)"
        )
        step = 20  # label every 20th point
        for i in range(0, len(line100_ref), step):
            p = line100_ref[i]
            ax.text(
                p[0], p[1],
                str(i),
                fontsize=10,
                color="green",
                ha="center",
                va="center",
                zorder=20,
                clip_on=False,
                bbox=dict(
                    facecolor="white",
                    edgecolor="green",
                    linewidth=0.5,
                    alpha=0.7,
                    pad=0.5
                )
            )"""
    # --------------------------------------------------
    # 5) Split and plot target rails
    # --------------------------------------------------
    ax.scatter(
        aligned_target_pts[:, 0],
        aligned_target_pts[:, 1],
        s=8, color="orange",
        label="Target (single rail)"
    )
    """if line100_tgt is not None:
        ax.scatter(
            line100_tgt[:, 0],
            line100_tgt[:, 1],
            s=28,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            label="Target rail (100 pts)"
        )"""
    # --------------------------------------------------
    # 6) Draw bounding box
    # --------------------------------------------------
    if bbox is not None:
        poly = np.vstack([bbox_poly, bbox_poly[0]])
        ax.plot(
            poly[:, 0],
            poly[:, 1],
            color="purple",
            linewidth=2,
            linestyle="--",
            dashes=(6, 4),
            label="Bounding Box"
        )

    # --------------------------------------------------
    # 7) Average width lines
    # --------------------------------------------------
    ref_seg = compute_centerline(reference_pts)
    tgt_seg = compute_centerline(aligned_target_pts)

    if ref_seg is not None:
        ax.plot(ref_seg[:, 0], ref_seg[:, 1],
                color="green", linewidth=3,
                label="Reference avg width")

    if tgt_seg is not None:
        ax.plot(tgt_seg[:, 0], tgt_seg[:, 1],
                color="black", linewidth=3,
                label="Target avg width")

    """step = 20  # label every 20th point on 100-pt target rail
    for i in range(0, len(line100_tgt), step):
        p = line100_tgt[i]
        ax.text(
            p[0], p[1],
            str(i),
            fontsize=10,
            color="red",
            zorder=21,
            clip_on=False,
            bbox=dict(
                facecolor="white",
                edgecolor="red",
                linewidth=0.5,
                alpha=0.8,
                pad=0.4
            )
        )"""

    # --------------------------------------------------
    # 8) Final formatting
    # --------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(loc="best")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

def precompute_icp_for_all_references(db_handler, n_points):
    """
    Precompute and store ICP geometry for all reference SVGs.
    """

    db_handler.use_collection("svg_template_types")

    for doc in db_handler.collection.find(
        {"cleaned_svg": {"$exists": True}},
        {"sample_id": 1, "cleaned_svg": 1, "icp_data": 1}
    ):
        if "icp_data" in doc:
            settings = doc["icp_data"].get("settings", {})
            if settings.get("n_points") == n_points:
                continue

        try:
            svg_string = doc.get("cropped_svg", doc["cleaned_svg"])

            pts, avg_width = prepare_icp_geometry_from_svg_string(
                svg_string,
                n_points
            )

            icp_data = {
                "outline_points": pts.tolist(),
                "avg_width": avg_width,
                "settings": {
                    "n_points": n_points,
                    "centering": "mean",
                    "width_normalization": True
                }
            }

            db_handler.collection.update_one(
                {"sample_id": doc["sample_id"]},
                {"$set": {"icp_data": icp_data}}
            )

            print(f"[ICP] prepared reference {doc['sample_id']}")

        except Exception as e:
            print(f"[ICP] FAILED reference {doc['sample_id']}: {e}")


def ensure_icp_geometry(doc, db_handler, n_points):
    """
    function to get info for icp method
    """
    if doc is None:
        raise ValueError("ensure_icp_geometry called with doc=None")

    if doc.get("icp_data"):
        settings = doc["icp_data"].get("settings", {})
        if settings.get("n_points") == n_points:
            return doc["icp_data"]
    
    svg_string = doc.get("cropped_svg", doc["cleaned_svg"])

    pts, avg_width = prepare_icp_geometry_from_svg_string(
        svg_string,
        n_points
    )

    icp_data = {
        "outline_points": pts.tolist(),
        "avg_width": avg_width,
        "settings": {
            "n_points": n_points,
            "centering": "mean",
            "width_normalization": True
        }
    }

    db_handler.collection.update_one(
        {"sample_id": doc["sample_id"]},
        {"$set": {"icp_data": icp_data}}
    )

    return icp_data


def find_icp_closest_matches(analysis_config, top_k=20):
    db_handler = analysis_config["db_handler"]
    sample_id = analysis_config["sample_id"]

    n_target = analysis_config.get("icp_n_target", 300)
    n_ref = analysis_config.get("icp_n_reference", 500)

    icp_params = {
        "iters": analysis_config.get("icp_iters", 30),
        "max_total_deg": analysis_config.get("icp_max_deg", 2.0),
        "max_scale_step": analysis_config.get("icp_max_scale", 0.2),
        "top_percent": analysis_config.get("icp_top_percent", 0.2)
    }

    db_handler.use_collection("svg_raw")
    doc = db_handler.collection.find_one({"sample_id": sample_id})

    icp_data = ensure_icp_geometry(doc, db_handler, n_target)
    target_pts = np.array(icp_data["outline_points"])

    db_handler.use_collection("svg_template_types")

    precompute_icp_for_all_references(db_handler, n_ref)

    refs = {}
    for ref_doc in db_handler.collection.find({"icp_data": {"$exists": True}}):
        refs[ref_doc["sample_id"]] = np.array(
            ref_doc["icp_data"]["outline_points"]
        )

    matches = find_icp_matches(
        target_pts,
        refs,
        icp_params,
        top_k
    )

    db_handler.use_collection("svg_raw")
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {
            "icp_matches": [
                {"id": m["id"], "distance": m["distance"]}
                for m in matches
            ],
            "icp_matches_valid": True,
            "icp_settings": icp_params
        }}
    )
    return matches

def generate_icp_overlap_image(db_handler, sample_id, template_id, analysis_config):
    n_target = analysis_config.get("icp_n_target", 300)
    n_ref = analysis_config.get("icp_n_reference", 500)

    # Load target
    db_handler.use_collection("svg_raw")
    target_doc = db_handler.collection.find_one({"sample_id": sample_id})
    target_icp = ensure_icp_geometry(target_doc, db_handler, n_target)
    target_pts = np.array(target_icp["outline_points"])

    # Load reference
    db_handler.use_collection("svg_template_types")
    ref_doc = db_handler.collection.find_one({"sample_id": template_id})
    ref_icp = ensure_icp_geometry(ref_doc, db_handler, n_ref)
    ref_pts = np.array(ref_icp["outline_points"])

    # Run ICP again (cheap at top-1 scale)
    err, aligned = run_icp(
        target_pts,
        ref_pts,
        iters=analysis_config.get("icp_iters", 30),
        max_total_deg=analysis_config.get("icp_max_deg", 2.0),
        max_scale_step=analysis_config.get("icp_max_scale", 0.2)
    )

    score, bbox = icp_score(ref_pts, aligned, ref_id=template_id)

    if not np.isfinite(score):
        return None

    return plot_icp_overlap(
        target_pts,
        aligned,
        ref_pts,
        bbox=bbox
    )