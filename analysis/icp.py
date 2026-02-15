import numpy as np
from svgpathtools import svg2paths2, Path
from scipy.spatial import cKDTree
import io
import matplotlib.pyplot as plt
from PIL import Image
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
    """
    Signed distance of points from the shape centerline.
    Used to separate the two outline rails.
    """
    mean = pts.mean(axis=0)
    X = pts - mean

    # PCA to get main (length) direction
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    main_dir = Vt[0]

    # perpendicular = width direction
    width_dir = np.array([-main_dir[1], main_dir[0]])

    return X @ width_dir

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


def top_percent_nn_matches(reference_pts, aligned_target_pts, top_percent):
    """
    Returns:
    - top reference points
    - their nearest neighbors in the aligned target
    """
    y = reference_pts[:, 1]
    cutoff = np.percentile(y, top_percent * 100)
    ref_top = reference_pts[y <= cutoff]

    if len(ref_top) == 0:
        return None, None

    tree = cKDTree(aligned_target_pts)
    _, idx = tree.query(ref_top)
    tgt_nn = aligned_target_pts[idx]

    return ref_top, tgt_nn

def same_height_distance(src_pts, dst_pts):
    """
    For each src point, find the dst point with closest Y value
    and return horizontal (X) distance.
    """
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return None

    dst_y = dst_pts[:, 1]

    dists = []

    for p in src_pts:
        y = p[1]
        idx = np.argmin(np.abs(dst_y - y))
        dx = np.linalg.norm(p - dst_pts[idx])
        dists.append(dx)

    return np.array(dists)


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
    """
    Robust ordering of points along a single rail using
    graph geodesic longest path (no PCA, no zigzag).
    """
    from scipy.spatial import cKDTree
    import heapq

    n = len(pts)
    tree = cKDTree(pts)

    # --- build adjacency list ---
    neighbors = tree.query(pts, k=k+1)[1][:, 1:]

    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            d = np.linalg.norm(pts[i] - pts[j])
            graph[i].append((j, d))
            graph[j].append((i, d))

    # --- Dijkstra ---
    def dijkstra(start):
        dist = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)
        dist[start] = 0.0
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in graph[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    # --- find graph diameter ---
    d0, _ = dijkstra(0)
    a = np.argmax(d0)

    d1, prev = dijkstra(a)
    b = np.argmax(d1)

    # --- recover path ---
    path = []
    cur = b
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return pts[path]

def extract_single_rail(pts, side="auto"):
    """
    Returns points belonging to ONE outline rail.
    side: "left", "right", or "auto"
    """
    w = signed_width_coordinate(pts)

    if side == "left":
        return pts[w < 0]
    if side == "right":
        return pts[w >= 0]

    # auto: pick denser side
    return pts[w < 0] if np.sum(w < 0) >= np.sum(w >= 0) else pts[w >= 0]


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
    - Local spatial consistency

    Always returns:
        (score, bbox)   or   (np.inf, None)
    """
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

    line20_pts, _ = make_20_line_points_on_target_rail(aligned_target_pts)
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

    if len(ref_box) < 20:
        print(f"[{ref_id}] Reject: ref_box too small:", len(ref_box))
        return np.inf, None

    # --------------------------------------------------
    # 2) Single-rail assumption inside bbox
    # --------------------------------------------------
    ref_rail = ref_box
    tgt_rail = tgt_box
    line20_pts, line20_ids = make_20_line_points_on_target_rail(tgt_rail)
    # --------------------------------------------------
    # 3) Curvature comparison per rail
    # --------------------------------------------------
    def curvature_error(a, b):
        if len(a) < 10 or len(b) < 10:
            print("Reject: too few points", len(a), len(b))
            return np.inf

        # ---- Direction check ----
        dir_a = a[-1] - a[0]
        dir_b = b[-1] - b[0]

        dir_a /= (np.linalg.norm(dir_a) + 1e-8)
        dir_b /= (np.linalg.norm(dir_b) + 1e-8)

        dot = np.dot(dir_a, dir_b)

        curv_a = discrete_curvature(a)
        curv_b = discrete_curvature(b)

        if curv_a is None or curv_b is None:
            print("Reject: curvature None")
            return np.inf

        n = min(len(curv_a), len(curv_b))
        if n < 10:
            return np.inf

        # ---- Compare forward ----
        curv_b_forward = curv_b[:n]
        curv_b_backward = curv_b[:n][::-1]

        err_forward = np.mean((curv_a[:n] - curv_b_forward) ** 2)
        err_backward = np.mean((curv_a[:n] - curv_b_backward) ** 2)

        best_err = min(err_forward, err_backward)

        # ---- Penalize opposite curvature sign ----
        sign_corr = np.mean(np.sign(curv_a[:n]) * np.sign(curv_b_forward))

        if sign_corr < -0.3:
            print(f"[{ref_id}] Reject: sign_corr =", sign_corr)
            return np.inf

        return best_err


    curvature_term = curvature_error(ref_rail, tgt_rail)

    if not np.isfinite(curvature_term):
        return np.inf, None

    # --------------------------------------------------
    # 3.5) TIP ALIGNMENT TERM (critical)
    # --------------------------------------------------

    ref_tip = extract_tip_region(ref_box, percent=0.15)
    tgt_tip = extract_tip_region(tgt_box, percent=0.15)

    if ref_tip is None or tgt_tip is None:
        return np.inf, None

    # nearest neighbor matching within tip region
    tree_tip = cKDTree(ref_tip)
    tip_dists, _ = tree_tip.query(tgt_tip)

    tip_spatial = np.mean(tip_dists)

    # curvature at tip (strong structural check)
    curv_ref_tip = discrete_curvature(ref_tip)
    curv_tgt_tip = discrete_curvature(tgt_tip)

    if curv_ref_tip is None or curv_tgt_tip is None:
        return np.inf, None

    n_tip = min(len(curv_ref_tip), len(curv_tgt_tip))

    if n_tip < 5:
        return np.inf, None

    tip_curv_err = np.mean(
        (curv_ref_tip[:n_tip] - curv_tgt_tip[:n_tip]) ** 2
    )

    tip_term = tip_spatial + 3.0 * tip_curv_err

    # --------------------------------------------------
    # 4) Spatial term (local consistency)
    # --------------------------------------------------
    tree_ref = cKDTree(ref_box)
    dists, _ = tree_ref.query(tgt_box)
    spatial_term = np.mean(dists)

    # --------------------------------------------------
    # Final weighted score
    # --------------------------------------------------
    w_spatial = 1.0
    w_curv = 2.0
    w_tip = 4.0

    score = (
        w_spatial * spatial_term +
        w_curv * curvature_term +
        w_tip * tip_term
    )
    return float(score), bbox_poly

def same_height_matches(src_pts, dst_pts):
    """
    For each src point, return the dst point with closest Y.
    """
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return None

    dst_y = dst_pts[:, 1]
    matches = []

    for p in src_pts:
        idx = np.argmin(np.abs(dst_y - p[1]))
        matches.append(dst_pts[idx])

    return np.array(matches)

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

def make_20_line_points_on_target_rail(target_pts, n_points=20):
    """
    Sample points along a single continuous rail
    using geodesic ordering (no PCA, no zigzag).
    """
    ordered = order_points_by_arclength(target_pts)

    if len(ordered) < n_points:
        return None, None

    idx = np.linspace(0, len(ordered) - 1, n_points).astype(int)
    line_pts = ordered[idx]

    return line_pts, np.arange(len(line_pts))

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

    fig, ax = plt.subplots(figsize=(6, 6))
    # --- Precompute 20-point rail line once ---
    line20_pts, _ = make_20_line_points_on_target_rail(aligned_target_pts)
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
    # --------------------------------------------------
    # 5) Split and plot target rails
    # --------------------------------------------------
    ax.scatter(
        aligned_target_pts[:, 0],
        aligned_target_pts[:, 1],
        s=8, color="orange",
        label="Target (single rail)"
    )

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

    ax.plot(
        line20_pts[:, 0],
        line20_pts[:, 1],
        color="red",
        linewidth=2,
        label="20-point rail line"
    )

    for i, p in enumerate(line20_pts):
        ax.text(p[0], p[1], str(i), fontsize=8, color="red")

    # --- Brown rail extended to bbox ---
    p_ext0, p_ext1 = extend_line(
        line20_pts[0],
        line20_pts[-1]
    )
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