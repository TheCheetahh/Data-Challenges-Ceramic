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
    if seg is None:
        return None
    return float(np.linalg.norm(seg[1] - seg[0]))

def run_icp(source_pts, target_pts,
            iters=30,
            max_total_deg=2.0,
            max_scale_step=0.02):

    src = source_pts.copy()
    dst = target_pts.copy()

    # initial top-point alignment
    src_top = src[np.argmin(src[:, 1])]
    dst_top = dst[np.argmin(dst[:, 1])]
    src += (dst_top - src_top)

    total_angle = 0.0
    best_err = np.inf

    for _ in range(iters):
        tree = cKDTree(dst)
        _, idx = tree.query(src)
        matched = dst[idx]

        mu_src = src.mean(0)
        mu_dst = matched.mean(0)

        X = src - mu_src
        Y = matched - mu_dst

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

        err = np.mean(np.linalg.norm(src - matched, axis=1))
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

"""def icp_score(reference_pts, aligned_target_pts, top_percent=0.2):
    Top-percent nearest-neighbor error (lower is better)
    y = reference_pts[:, 1]
    cutoff = np.percentile(y, top_percent * 100)

    ref_top = reference_pts[y <= cutoff]
    tree = cKDTree(aligned_target_pts)
    dists, _ = tree.query(ref_top)

    return float(np.mean(dists))"""

def icp_score(reference_pts, aligned_target_pts, top_percent=0.1):
    """
    Asymmetric symmetric ICP score:

    - reference → target : top `top_percent` of reference
    - target → reference : entire target
    - outline-aware (left/right rails)
    """

    def outline_distance(src_pts, dst_pts, use_top):
        # --- select points ---
        if use_top:
            y = src_pts[:, 1]
            cutoff = np.percentile(y, top_percent * 100)
            src = src_pts[y <= cutoff]
        else:
            src = src_pts

        if len(src) == 0 or len(dst_pts) == 0:
            return None

        # --- signed width ---
        src_w = signed_width_coordinate(src)
        dst_w = signed_width_coordinate(dst_pts)

        # --- split rails ---
        src_pos = src[src_w >= 0]
        src_neg = src[src_w < 0]

        dst_pos = dst_pts[dst_w >= 0]
        dst_neg = dst_pts[dst_w < 0]

        dists = []

        if len(src_pos) > 0 and len(dst_pos) > 0:
            tree_p = cKDTree(dst_pos)
            d_p, _ = tree_p.query(src_pos)
            dists.append(d_p)

        if len(src_neg) > 0 and len(dst_neg) > 0:
            tree_n = cKDTree(dst_neg)
            d_n, _ = tree_n.query(src_neg)
            dists.append(d_n)

        if not dists:
            return None

        return float(np.percentile(np.concatenate(dists), 95))

    # --- directional distances ---
    ref_to_tgt = outline_distance(
        reference_pts,
        aligned_target_pts,
        use_top=True
    )

    tgt_to_ref = outline_distance(
        aligned_target_pts,
        reference_pts,
        use_top=False
    )

    if ref_to_tgt is None and tgt_to_ref is None:
        return float("inf")

    if ref_to_tgt is None:
        base_error = tgt_to_ref
    elif tgt_to_ref is None:
        base_error = ref_to_tgt
    else:
        base_error = 0.5 * (ref_to_tgt + tgt_to_ref)

    # --- asymmetry penalty (global, unchanged) ---
    ref_w = signed_width_coordinate(reference_pts)
    tgt_w = signed_width_coordinate(aligned_target_pts)

    if np.any(ref_w < 0) and np.any(ref_w >= 0):
        ref_left = np.mean(np.abs(ref_w[ref_w < 0]))
        ref_right = np.mean(np.abs(ref_w[ref_w >= 0]))
        ref_asym = abs(ref_left - ref_right)
    else:
        ref_asym = 0.0

    if np.any(tgt_w < 0) and np.any(tgt_w >= 0):
        tgt_left = np.mean(np.abs(tgt_w[tgt_w < 0]))
        tgt_right = np.mean(np.abs(tgt_w[tgt_w >= 0]))
        tgt_asym = abs(tgt_left - tgt_right)
    else:
        tgt_asym = 0.0

    asym_penalty = abs(ref_asym - tgt_asym)

    alpha = 2.0
    return base_error + alpha * asym_penalty

def find_icp_matches(
    target_pts,
    reference_dict,
    icp_params,
    top_k=5
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
        score = icp_score(ref_pts, aligned, icp_params["top_percent"])
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


def plot_icp_overlap(target_pts, aligned_target_pts, reference_pts):
    """
    Blue = reference
    Orange = target (aligned)
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(reference_pts[:, 0], reference_pts[:, 1],
               s=6, color="blue", label="Reference")

    ax.scatter(aligned_target_pts[:, 0], aligned_target_pts[:, 1],
               s=6, color="orange", label="Target (aligned)")

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
            pts, avg_width = prepare_icp_geometry_from_svg_string(
                doc["cleaned_svg"],
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
    if "icp_data" in doc:
        settings = doc["icp_data"].get("settings", {})
        if settings.get("n_points") == n_points:
            return doc["icp_data"]
    
    pts, avg_width = prepare_icp_geometry_from_svg_string(
        doc["cleaned_svg"],
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


def find_icp_closest_matches(analysis_config, top_k=5):
    db_handler = analysis_config["db_handler"]
    sample_id = analysis_config["sample_id"]

    n_target = analysis_config.get("icp_n_target", 100)
    n_ref = analysis_config.get("icp_n_reference", 300)

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
