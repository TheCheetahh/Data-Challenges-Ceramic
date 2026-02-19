import io
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path, curvature_from_points
from analysis.calculation.apply_metric import apply_metric
from analysis.calculation.distance_methods import euclidean_distance, cosine_similarity_distance, correlation_distance, \
    dtw_distance, integral_difference
from database_handler import MongoDBHandler
from analysis.icp import ensure_icp_geometry, run_icp, icp_score

def compute_curvature_for_all_items(analysis_config):
    """
    Compute curvature for ALL documents that have cleaned_svg (skip if already stored).
    For each document that does not have it call compute_and_store_curvature

    :param analysis_config:
    :return:
    """

    # set vars from analysis_config
    db_handler = analysis_config.get("db_handler")
    distance_type_dataset = analysis_config.get("distance_type_dataset")
    smooth_method = analysis_config.get("smooth_method")
    smooth_factor = analysis_config.get("smooth_factor")
    smooth_window = analysis_config.get("smooth_window")
    n_samples = analysis_config.get("n_samples")

    # set collection of db_handler
    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    # get all items
    docs = db_handler.collection.find({}, {"sample_id": 1, "cleaned_svg": 1, "curvature_data": 1, "outdated_curvature": 1})

    # init counter vars
    processed = 0
    skipped = 0
    errors = 0

    # iterate over all items
    for doc in docs:
        current_sample_id = doc.get("sample_id")
        if not current_sample_id:
            continue

        # load curvature data and smoothing settings of the doc
        stored_data = doc.get("curvature_data")
        stored_settings = stored_data.get("settings", {}) if stored_data else {}

        # Skip if curvature is up-to-date (same settings AND not marked as outdated)
        if (
                not doc.get("outdated_curvature", False) and
                stored_settings.get("smooth_method") == smooth_method and
                float(stored_settings.get("smooth_factor", 0)) == float(smooth_factor) and
                int(stored_settings.get("smooth_window", 0)) == int(smooth_window) and
                int(stored_settings.get("n_samples", 0)) == int(n_samples)
        ):
            skipped += 1
            continue

        # Compute and overwrite stored curvature data if necessary
        status = compute_curvature_for_one_item(analysis_config, current_sample_id)

        if status.startswith("❌"):
            errors += 1
        else:
            processed += 1

    # returns status message
    return f"✅ Recomputed: {processed}, ⏭️ Skipped (same settings): {skipped}, ❌ Errors: {errors}"


def compute_curvature_for_one_item(analysis_config, current_sample_id):
    """
    Computes and stores curvature, direction, arc-length,
    and lip anchor indices for a single sample.
    Uses cropped_svg if available, otherwise falls back to cleaned_svg.

    :param analysis_config:
    return
    """

    # set vars from analysis_config
    db_handler = analysis_config.get("db_handler")
    distance_type_dataset = analysis_config.get("distance_type_dataset")
    smooth_method = analysis_config.get("smooth_method")
    smooth_factor = analysis_config.get("smooth_factor")
    smooth_window = analysis_config.get("smooth_window")
    n_samples = analysis_config.get("n_samples")

    # create db handler
    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    # get the doc of the sample_id
    doc = db_handler.collection.find_one({"sample_id": current_sample_id})
    if doc is None:
        return f"❌ No sample found with sample_id: {current_sample_id}"

    # --- Parse SVG path ---
    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")

        # Use cropped_svg if available, otherwise use cleaned_svg
        svg_content = doc.get("cropped_svg") or doc.get("cleaned_svg")

        if not svg_content:
            return f"❌ No SVG data found (neither cropped_svg nor cleaned_svg) for sample_id: {current_sample_id}"

        svg_file_like = io.StringIO(svg_content)
        print(
            f"Using {'cropped_svg' if 'cropped_svg' in doc and doc.get('cropped_svg') else 'cleaned_svg'} for sample_id: {current_sample_id}")

    else:
        db_handler.use_collection("svg_template_types")
        svg_file_like = io.StringIO(doc["raw_content"])

    paths, _ = svg2paths(svg_file_like)
    if len(paths) == 0:
        return f"❌ No valid path found in SVG."

    path = paths[0]

    # Sample points
    ts = np.linspace(0, 1, n_samples)
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    # Normalize & smooth
    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    # Compute curvature
    curvature = curvature_from_points(points)

    # Arc length (normalized)
    arc_lengths = np.concatenate(
        ([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    arc_lengths /= arc_lengths[-1]

    # Direction (angle to x-axis)
    diffs = np.diff(points, axis=0)
    directions = np.arctan2(diffs[:, 1], diffs[:, 0])
    directions = np.concatenate(([directions[0]], directions))

    # 🆕 Lip anchor detection
    lip_idx_angle = find_lip_index_by_angle(directions)
    lip_idx_curvature = find_lip_index_by_curvature(curvature)

    lip_arc_angle = (
        float(arc_lengths[lip_idx_angle])
        if lip_idx_angle is not None else None
    )
    lip_arc_curvature = (
        float(arc_lengths[lip_idx_curvature])
        if lip_idx_curvature is not None else None
    )

    # Store in DB
    db_handler.collection.update_one(
        {"sample_id": current_sample_id},
        {"$set": {
            "closest_matches_valid": False,
            "outdated_curvature": False,
            "curvature_data": {
                "arc_lengths": arc_lengths.tolist(),
                "curvature": curvature.tolist(),
                "directions": directions.tolist(),

                # 🆕 Lip anchors
                "lip_anchor": {
                    "angle": {
                        "index": int(lip_idx_angle) if lip_idx_angle is not None else None,
                        "arc_length": lip_arc_angle
                    },
                    "curvature": {
                        "index": int(lip_idx_curvature) if lip_idx_curvature is not None else None,
                        "arc_length": lip_arc_curvature
                    }
                },
                "settings": {
                    "smooth_method": smooth_method,
                    "smooth_factor": smooth_factor,
                    "smooth_window": smooth_window,
                    "n_samples": n_samples
                }
            }
        }}
    )

    return f"✅ Curvature + lip anchors stored for sample_id {current_sample_id}"


def generate_all_plots(analysis_config):
    """
    Compute curvature if missing or settings changed, otherwise load from DB.
    Also generates debug plots with lip anchor markers.
    """

    # set vars from analysis_config
    db_handler = analysis_config.get("db_handler")
    sample_id = analysis_config.get("sample_id")
    distance_type_dataset = analysis_config.get("distance_type_dataset")
    smooth_method = analysis_config.get("smooth_method")
    smooth_factor = analysis_config.get("smooth_factor")
    smooth_window = analysis_config.get("smooth_window")
    n_samples = analysis_config.get("n_samples")
    distance_value_dataset = analysis_config.get("distance_value_dataset")

    # set db handler
    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    # get document of the sample id
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "cleaned_svg" not in doc:
        return None, None, None, f"❌ No cleaned SVG found for sample_id {sample_id}"

    # figure out if data is already computed
    recompute = True
    if "curvature_data" in doc and "settings" in doc["curvature_data"]:
        stored = doc["curvature_data"]["settings"]
        if (
                stored.get("smooth_method") == smooth_method and
                float(stored.get("smooth_factor", 0)) == float(smooth_factor) and
                int(stored.get("smooth_window", 0)) == int(smooth_window) and
                int(stored.get("n_samples", 0)) == int(n_samples)
        ):
            recompute = False

    # compute data if needed
    if recompute:
        print("Debug: recompute of plots")
        compute_curvature_for_one_item(analysis_config, sample_id)
        doc = db_handler.collection.find_one({"sample_id": sample_id})

    curvature_data = doc["curvature_data"]
    arc_lengths = np.array(curvature_data["arc_lengths"])
    curvature = np.array(curvature_data["curvature"])
    directions = np.array(curvature_data["directions"])

    lip_anchor = curvature_data.get("lip_anchor", {})
    lip_angle_arc = lip_anchor.get("angle", {}).get("arc_length")
    lip_curv_arc = lip_anchor.get("curvature", {}).get("arc_length")

    status_msg = f"✅ Loaded stored curvature for sample_id {sample_id}"

    # Reconstruct points for color map
    # Use cropped_svg if available, otherwise use cleaned_svg
    svg_file_like = io.StringIO(doc.get("cropped_svg") or doc.get("cleaned_svg"))

    paths, _ = svg2paths(svg_file_like)
    path = paths[0]

    ts = np.linspace(0, 1, len(curvature))
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    stored_settings = curvature_data.get("settings", {})
    smooth_method = stored_settings.get("smooth_method", smooth_method)
    smooth_factor = stored_settings.get("smooth_factor", smooth_factor)
    smooth_window = stored_settings.get("smooth_window", smooth_window)

    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    # ============================================================
    # Curvature line plot (with lip markers)
    # ============================================================
    buf1 = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", linestyle="--")

    plt.plot(arc_lengths, curvature, color="black", label="Curvature")

    if lip_angle_arc is not None:
        plt.axvline(lip_angle_arc, color="blue", linestyle="--", linewidth=2,
                    label="Lip (angle)")

    if lip_curv_arc is not None:
        plt.axvline(lip_curv_arc, color="red", linestyle="--", linewidth=2,
                    label="Lip (curvature)")

    plt.title(f"Curvature along normalized arc length (sample {sample_id})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Curvature κ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf1, format="png")
    plt.close()
    buf1.seek(0)
    curvature_plot_img = Image.open(buf1)

    # ============================================================
    # Curvature color map (geometry view)
    # ============================================================
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(
        vmin=-np.max(np.abs(curvature)),
        vmax=np.max(np.abs(curvature)) * 0.8
    )

    buf2 = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    lc.set_array(curvature)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    ax.invert_yaxis()
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Curvature Color Map")
    plt.colorbar(lc, ax=ax, label="Curvature κ")
    plt.tight_layout()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    curvature_color_img = Image.open(buf2)

    # ============================================================
    # Direction plot (with lip markers)
    # ============================================================
    directions = np.unwrap(directions)
    directions_deg = np.degrees(directions)

    buf3 = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.plot(arc_lengths, directions_deg, color="blue", label="Direction")

    if lip_angle_arc is not None:
        plt.axvline(lip_angle_arc, color="blue", linestyle="--", linewidth=2,
                    label="Lip (angle)")

    if lip_curv_arc is not None:
        plt.axvline(lip_curv_arc, color="red", linestyle="--", linewidth=2,
                    label="Lip (curvature)")

    plt.title(f"Direction along normalized arc length (sample {sample_id})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Angle to x-axis [°]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf3, format="png")
    plt.close()
    buf3.seek(0)
    angle_plot_img = Image.open(buf3)

    return curvature_plot_img, curvature_color_img, angle_plot_img, status_msg


def find_enhanced_closest_curvature(analysis_config):
    """
    Calculate close samples, save results to DB, and return the top result.
    """
    db_handler = analysis_config.get("db_handler")
    
    sample_id = analysis_config.get("sample_id")
    distance_value_dataset = analysis_config.get("distance_value_dataset")
    distance_calculation = analysis_config.get("distance_calculation")
    top_k = analysis_config.get("top_k")
    
    db_handler.use_collection("svg_raw")
    # Load sample curvature data
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "curvature_data" not in doc:
        return None, None, f"No curvature data for sample_id {sample_id}"

    sample_curvature = np.array(doc["curvature_data"]["curvature"])
    sample_direction = np.array(doc["curvature_data"]["directions"])

    db_handler.use_collection("svg_template_types")

    # Compute distances against all other samples
    curve_len = len(sample_curvature)
    crop = int(curve_len * 0.10)

    distances = []
    # since the collection used is svg_template_types this is templates
    for template_doc in db_handler.collection.find(
        {"sample_id": {"$ne": sample_id}},
        {"sample_id": 1, "curvature_data": 1}
    ):
        template_id = template_doc["sample_id"]
        curv_data = template_doc.get("curvature_data")
        if not curv_data:
            continue

        other_curv = np.array(curv_data["curvature"])
        other_dir = np.array(curv_data["directions"])

        # Guard shared by several distance types
        if curve_len <= 2 * crop:
            dist = None
        elif distance_value_dataset == "only curvature":
            dist = float(apply_metric(sample_curvature, other_curv, distance_calculation))

        elif distance_value_dataset == "lip_aligned_angle":
            n_samples = analysis_config.get("n_samples")
            if sample_direction is None or len(sample_direction) < 20:
                dist = None
            else:
                direction_deg = ((np.degrees(sample_direction) + 180) % 360) - 180
                shard_candidates = find_all_lip_index_by_angle(sample_direction)
                if len(shard_candidates) == 0:
                    dist = None
                else:
                    tmpl_db = MongoDBHandler("svg_data")
                    tmpl_db.use_collection("svg_template_types")
                    template_doc = tmpl_db.collection.find_one({"sample_id": template_id})
                    if template_doc is None or "raw_content" not in template_doc:
                        dist = None
                    else:
                        paths, _ = svg2paths(io.StringIO(template_doc["raw_content"]))
                        if not paths:
                            dist = None
                        else:
                            path_template = paths[0]
                            n_shard = len(sample_direction)
                            min_distance = np.inf
                            left, right = n_samples, 20000

                            while left < right:
                                mid1 = left + (right - left) // 3
                                mid2 = left + 2 * (right - left) // 3
                                dist1, *_ = compute_distance_for_resample(
                                    path_template, shard_candidates, direction_deg, n_shard, mid1)
                                dist2, *_ = compute_distance_for_resample(
                                    path_template, shard_candidates, direction_deg, n_shard, mid2)

                                if dist1 < dist2:
                                    right = mid2 - 1
                                else:
                                    left = mid1 + 1

                                min_distance = min(min_distance, dist1, dist2)

                            dist = float(min_distance) if np.isfinite(min_distance) else None

        elif distance_value_dataset == "ICP":
            n_target = analysis_config.get("icp_n_target", 300)
            n_ref = analysis_config.get("icp_n_reference", 500)
            icp_params = {
                "iters":          analysis_config.get("icp_iters", 30),
                "max_total_deg":  analysis_config.get("icp_max_deg", 2.0),
                "max_scale_step": analysis_config.get("icp_max_scale", 0.2),
                "top_percent":    analysis_config.get("icp_top_percent", 0.2),
            }
            try:
                db_handler.use_collection("svg_raw")
                target_doc = db_handler.collection.find_one({"sample_id": sample_id})
                if target_doc is None:
                    raise ValueError("Target document not found")
                target_pts = np.array(ensure_icp_geometry(target_doc, db_handler, n_target)["outline_points"])
            except Exception as e:
                analysis_config.setdefault("icp_skipped_targets", []).append({"id": template_id, "reason": str(e)})
                dist = float("inf")
            else:
                try:
                    db_handler.use_collection("svg_template_types")
                    ref_doc = db_handler.collection.find_one({"sample_id": template_id})
                    if ref_doc is None:
                        raise ValueError("Ref not found")
                    ref_pts = np.array(ensure_icp_geometry(ref_doc, db_handler, n_ref)["outline_points"])
                    err, aligned = run_icp(target_pts, ref_pts, **icp_params)
                    if not np.isfinite(err):
                        raise ValueError("non-finite ICP error")
                    score, _ = icp_score(ref_pts, aligned, ref_id=template_id)
                    if not np.isfinite(score):
                        raise ValueError("non-finite ICP score")
                    dist = float(score)
                except Exception as e:
                    analysis_config.setdefault("icp_skipped_targets", []).append({"id": template_id, "reason": str(e)})
                    dist = float("inf")

        elif distance_value_dataset == "cropped curvature":
            dist = float(apply_metric(sample_curvature[crop:-crop], other_curv[crop:-crop], distance_calculation))
        elif distance_value_dataset == "only angle":
            dist = float(apply_metric(sample_direction, other_dir, distance_calculation))
        elif distance_value_dataset == "cropped angle":
            dist = float(apply_metric(sample_direction[crop:-crop], other_dir[crop:-crop], distance_calculation))
        elif distance_value_dataset == "cropped curvature and angle":
            dist = float(
                apply_metric(sample_curvature[crop:-crop], other_curv[crop:-crop], distance_calculation) +
                apply_metric(sample_direction[crop:-crop], other_dir[crop:-crop], distance_calculation)
            )
        else:
            # Fallback: full curvature + full angle
            dist = float(
                apply_metric(sample_curvature, other_curv, distance_calculation) +
                apply_metric(sample_direction, other_dir, distance_calculation)
            )

        if dist is not None:
            distances.append((template_id, dist))

    if not distances:
        return None, None, "No comparable samples found."

    distances.sort(key=lambda x: x[1])
    top_matches = []
    for sid, dist in distances:
        if not math.isfinite(dist) or (top_k is not None and len(top_matches) >= top_k):
            break
        top_matches.append({"id": sid, "distance": float(dist)})

    # Save results
    db_handler.use_collection("svg_raw")
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {
            "closest_matches": top_matches,
            "full_closest_matches": top_matches,
            "closest_matches_valid": True
        }}
    )

    closest = top_matches[0]
    msg = f"Closest sample to {sample_id} is {closest['id']} (distance={closest['distance']:.6f})"
    return closest["id"], closest["distance"], msg


def find_all_lip_index_by_angle(directions, angle_tolerance_deg=5.0, edge_margin=0.05):
    if len(directions) == 0:
        return []

    angles_deg = np.degrees(directions)
    angles_deg = ((angles_deg + 180) % 360) - 180

    n = len(angles_deg)
    start = int(edge_margin * n)
    end = int((1.0 - edge_margin) * n)
    idx_range = np.arange(start, end)

    abs_dev = np.abs(angles_deg[idx_range])
    candidate_mask = abs_dev <= angle_tolerance_deg
    candidate_indices = idx_range[candidate_mask]

    if len(candidate_indices) == 0:
        # fallback: pick the closest to zero
        fallback_idx = idx_range[np.argmin(abs_dev)]
        candidate_indices = [fallback_idx]

    # remove consecutive duplicates
    candidates_filtered = [candidate_indices[0]]
    for idx in candidate_indices[1:]:
        if idx - candidates_filtered[-1] > 1:
            candidates_filtered.append(idx)

    return candidates_filtered



def find_lip_index_by_curvature(curvature, edge_margin=0.05):
    """
    Find index of maximum absolute curvature,
    preferring candidates near the middle of the path.

    :param curvature: np.ndarray of curvature values
    :param edge_margin: fraction of path length to ignore at both ends
    :return: index of lip candidate
    """

    if len(curvature) == 0:
        return None

    n = len(curvature)
    mid_idx = n // 2

    # Ignore edges
    start = int(edge_margin * n)
    end = int((1.0 - edge_margin) * n)

    idx_range = np.arange(start, end)
    abs_curv = np.abs(curvature[idx_range])

    max_val = np.max(abs_curv)

    # All indices with maximum curvature
    candidate_indices = idx_range[abs_curv == max_val]

    if len(candidate_indices) > 0:
        # Choose the one closest to the middle
        lip_idx = candidate_indices[np.argmin(np.abs(candidate_indices - mid_idx))]
        return int(lip_idx)

    # Fallback (should not happen)
    return int(idx_range[np.argmax(abs_curv)])


def angular_difference(a, b):
    delta = (a - b + np.pi) % (2 * np.pi) - np.pi
    return delta


# -------------------------------------------------------
# Plotting function
# -------------------------------------------------------
def plot_lip_alignment(shard_deg, template_deg, zero_idx_shard, sample_id, template_id):
    import matplotlib.pyplot as plt
    import os

    # Ensure the folder exists
    folder = "debug_plots"
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(12, 4))
    n_points = len(shard_deg)
    x = np.arange(n_points)

    plt.plot(x, shard_deg, label=f"Sample {sample_id}", color="blue")
    plt.plot(x, template_deg, label=f"Template {template_id}", color="red")

    # Mark zero-crossing index on sample
    plt.axvline(zero_idx_shard, color="green", linestyle="--", label="Zero Crossing (sample)")

    plt.title(f"Lip-Aligned Angle Comparison\nSample {sample_id} vs Template {template_id}")
    plt.xlabel("Point index")
    plt.ylabel("Angle [deg]")
    plt.legend()
    plt.grid(True)

    filepath = os.path.join(folder, f"lip_alignment_{sample_id}_{template_id}.png")
    plt.savefig(filepath)
    plt.close()

    print(f"Saved lip alignment plot to {filepath}")


def find_lip_index_by_angle(directions, angle_tolerance_deg=5.0, edge_margin=0.05):
    """
    Find index where direction is closest to horizontal (0°),
    preferring candidates near the middle of the path.

    :param directions: np.ndarray of angles in radians
    :param angle_tolerance_deg: tolerance around 0° to consider as horizontal
    :param edge_margin: fraction of path length to ignore at both ends
    :return: index of lip candidate (single int)
    """
    if len(directions) == 0:
        return None

    # Convert to degrees and normalize to [-180, 180]
    angles_deg = np.degrees(np.unwrap(directions))
    angles_deg = (angles_deg + 180) % 360 - 180

    n = len(angles_deg)
    mid_idx = n // 2

    # Ignore edges
    start = int(edge_margin * n)
    end = int((1.0 - edge_margin) * n)
    idx_range = np.arange(start, end)

    # Absolute deviation from horizontal
    abs_dev = np.abs(angles_deg[idx_range])

    # Find candidates within tolerance
    candidate_mask = abs_dev <= angle_tolerance_deg
    candidate_indices = idx_range[candidate_mask]

    if len(candidate_indices) > 0:
        # Choose the one closest to the middle
        lip_idx = candidate_indices[np.argmin(np.abs(candidate_indices - mid_idx))]
        return int(lip_idx)

    # Fallback: global minimum deviation (still middle-biased)
    fallback_idx = idx_range[np.argmin(abs_dev)]
    return int(fallback_idx)


def compute_distance_for_resample(path_template, shard_candidates, direction_deg, n_shard, n_resample):
    ts = np.linspace(0, 1, n_resample)
    points = np.array([path_template.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))
    points = normalize_path(points, smooth_method="savgol", smooth_factor=0.02, smooth_window=15)

    diffs = np.diff(points, axis=0)
    dir_template = np.arctan2(diffs[:, 1], diffs[:, 0])
    dir_template = np.concatenate(([dir_template[0]], dir_template))
    dir_template_deg = np.degrees(dir_template)
    dir_template_deg = ((dir_template_deg + 180) % 360) - 180

    template_candidates = find_all_lip_index_by_angle(dir_template)
    # print(len(template_candidates))
    if len(template_candidates) == 0:
        return np.inf, None, None, None

    local_min = np.inf
    local_best_crop = None
    local_shard_candidate = None
    local_template_candidate = None

    for shard_idx in shard_candidates:
        for template_idx in template_candidates:
            shift = shard_idx - template_idx
            dir_aligned_deg = np.roll(dir_template_deg, shift)
            dir_aligned_crop = dir_aligned_deg[:n_shard]
            diffs_angle = direction_deg[:len(dir_aligned_crop)] - dir_aligned_crop
            distance_val = np.mean(diffs_angle ** 2)
            if distance_val < local_min:
                local_min = distance_val
                local_best_crop = dir_aligned_crop.copy()
                local_shard_candidate = shard_idx
                local_template_candidate = template_idx

    return local_min, local_best_crop, local_shard_candidate, local_template_candidate