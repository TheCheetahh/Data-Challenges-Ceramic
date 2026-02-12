import numpy as np
from svgpathtools import svg2paths2, Path
from scipy.spatial import cKDTree
import io
import matplotlib.pyplot as plt
from PIL import Image
from analysis.analyze_curvature import curvature_from_points
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

def split_by_rail(pts, ref_pts_for_sign):
    """
    Split pts into left/right rails using signed width
    computed from ref_pts_for_sign PCA frame.
    """
    w = signed_width_coordinate(ref_pts_for_sign)
    sign = np.sign(w.mean()) if np.any(w != 0) else 1.0

    pts_w = signed_width_coordinate(pts)

    left = pts[pts_w * sign < 0]
    right = pts[pts_w * sign >= 0]

    return left, right


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
        score = icp_score(ref_pts, aligned)
        results.append((ref_id, score, aligned))

    results.sort(key=lambda x: x[1])
    return [
        {
            "id": rid,
            "distance": float(score),
            "aligned_target": aligned.tolist()
        }
        for rid, score, aligned in results[:top_k]
    ]

def discrete_curvature(pts):
    """
    Fast discrete curvature estimate.
    Returns curvature magnitude for interior points.
    """

    if pts is None or len(pts) < 3:
        return None

    # consecutive triplets
    p_prev = pts[:-2]
    p_mid  = pts[1:-1]
    p_next = pts[2:]

    v1 = p_mid - p_prev
    v2 = p_next - p_mid

    # normalize to avoid scale bias
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)

    # curvature proxy = change in direction
    curvature = np.linalg.norm(v2_norm - v1_norm, axis=1)

    return curvature


def icp_score(reference_pts,
              aligned_target_pts,
              overlap_dist_ratio=0.05):
    """
    Structural ICP score.

    Combines:
    - symmetric spatial distance
    - tangent direction consistency
    - longitudinal width profile similarity
    - symmetric coverage penalty
    """

    if len(reference_pts) < 5 or len(aligned_target_pts) < 5:
        return np.inf

    # -----------------------------
    # 1) Symmetric spatial distance
    # -----------------------------
    tree_ref = cKDTree(reference_pts)
    d_fwd, idx_fwd = tree_ref.query(aligned_target_pts)

    tree_tgt = cKDTree(aligned_target_pts)
    d_rev, _ = tree_tgt.query(reference_pts)

    spatial_error = 0.5 * (np.mean(d_fwd) + np.mean(d_rev))

    # -----------------------------
    # 2) Symmetric coverage
    # -----------------------------
    ref_min = reference_pts.min(axis=0)
    ref_max = reference_pts.max(axis=0)
    diag = np.linalg.norm(ref_max - ref_min)

    overlap_thresh = overlap_dist_ratio * diag

    coverage_fwd = np.mean(d_fwd < overlap_thresh)
    coverage_rev = np.mean(d_rev < overlap_thresh)

    coverage_ratio = min(coverage_fwd, coverage_rev)

    if coverage_ratio < 0.1:
        return np.inf

    coverage_penalty = 5.0 * (1.0 - coverage_ratio) ** 2

    # -----------------------------
    # 3) Tangent direction error
    # -----------------------------
    matched_tgt = aligned_target_pts[d_fwd < overlap_thresh]
    matched_ref = reference_pts[idx_fwd[d_fwd < overlap_thresh]]

    tangent_error = 0.0

    if len(matched_tgt) > 10:
        tgt_vecs = matched_tgt[1:] - matched_tgt[:-1]
        ref_vecs = matched_ref[1:] - matched_ref[:-1]

        tgt_vecs /= np.linalg.norm(tgt_vecs, axis=1, keepdims=True) + 1e-8
        ref_vecs /= np.linalg.norm(ref_vecs, axis=1, keepdims=True) + 1e-8

        min_len = min(len(tgt_vecs), len(ref_vecs))

        if min_len > 5:
            dots = np.sum(
                tgt_vecs[:min_len] * ref_vecs[:min_len],
                axis=1
            )
            dots = np.clip(dots, -1.0, 1.0)
            tangent_error = np.mean(np.arccos(dots))

    # -----------------------------
    # 4) Width profile error
    # -----------------------------
    ref_mean = reference_pts.mean(axis=0)
    X_ref = reference_pts - ref_mean

    _, _, Vt = np.linalg.svd(X_ref, full_matrices=False)
    main_dir = Vt[0]
    width_dir = np.array([-main_dir[1], main_dir[0]])

    ref_s = (reference_pts - ref_mean) @ main_dir
    tgt_s = (aligned_target_pts - ref_mean) @ main_dir

    ref_w = (reference_pts - ref_mean) @ width_dir
    tgt_w = (aligned_target_pts - ref_mean) @ width_dir

    bins = np.linspace(ref_s.min(), ref_s.max(), 20)

    width_errors = []

    for i in range(len(bins) - 1):
        mask_ref = (ref_s >= bins[i]) & (ref_s < bins[i+1])
        mask_tgt = (tgt_s >= bins[i]) & (tgt_s < bins[i+1])

        if np.sum(mask_ref) > 5 and np.sum(mask_tgt) > 5:
            w_ref = np.max(ref_w[mask_ref]) - np.min(ref_w[mask_ref])
            w_tgt = np.max(tgt_w[mask_tgt]) - np.min(tgt_w[mask_tgt])
            width_errors.append(abs(w_ref - w_tgt))

    width_profile_error = np.mean(width_errors) if width_errors else 0.0

    # -----------------------------
    # Final weighted score
    # -----------------------------
    w_spatial = 1.0
    w_tangent = 1.0
    w_width   = 1.5

    score = (
        w_spatial * spatial_error
        + w_tangent * tangent_error
        + w_width * width_profile_error
        + coverage_penalty
    )

    return float(score)


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

def plot_icp_overlap(
    target_pts,
    aligned_target_pts,
    reference_pts,
    top_percent=0.2
):
    """
    Blue  = reference
    Orange = target (aligned)
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # --- scatter outlines ---
    ax.scatter(
        reference_pts[:, 0],
        reference_pts[:, 1],
        s=6,
        color="blue",
        label="Reference"
    )

    ax.scatter(
        aligned_target_pts[:, 0],
        aligned_target_pts[:, 1],
        s=6,
        color="orange",
        label="Target (aligned)"
    )
    
    # --- average width segments ---
    ref_seg = compute_centerline(reference_pts)
    tgt_seg = compute_centerline(aligned_target_pts)

    if ref_seg is not None:
        ax.plot(
            ref_seg[:, 0],
            ref_seg[:, 1],
            color="green",
            linewidth=3,
            label="Reference avg width"
        )

    if tgt_seg is not None:
        ax.plot(
            tgt_seg[:, 0],
            tgt_seg[:, 1],
            color="red",
            linewidth=3,
            label="Target avg width"
        )

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
        "max_scale_step": analysis_config.get("icp_max_scale", 0.02),
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
