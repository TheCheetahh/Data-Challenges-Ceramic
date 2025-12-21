import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path, curvature_from_points
from analysis.distance_methods import euclidean_distance, cosine_similarity_distance, correlation_distance, \
    dtw_distance, integral_difference
from database_handler import MongoDBHandler


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
    docs = db_handler.collection.find({}, {"sample_id": 1, "cleaned_svg": 1, "curvature_data": 1})

    # init counter vars
    processed = 0
    skipped = 0
    errors = 0

    # iterate over all items
    for doc in docs:
        current_sanple_id = doc.get("sample_id")
        if not current_sanple_id:
            continue

        # load curvature data and smoothing settings of the doc
        stored_data = doc.get("curvature_data")
        stored_settings = stored_data.get("settings", {}) if stored_data else {}

        # if the loaded curvature data has the same settings as the current settings skip this doc
        if (
                stored_settings.get("smooth_method") == smooth_method and
                float(stored_settings.get("smooth_factor", 0)) == float(smooth_factor) and
                int(stored_settings.get("smooth_window", 0)) == int(smooth_window) and
                int(stored_settings.get("n_samples", 0)) == int(n_samples)
        ):
            skipped += 1
            continue

        # Compute and overwrite stored curvature data if necessary
        print("Debug: compute_curvature_for_one_item")
        status = compute_curvature_for_one_item(analysis_config, current_sanple_id)

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
    if "cleaned_svg" not in doc:
        return f"❌ Field 'cleaned_svg' not found in document for sample_id: {current_sample_id}"

    # --- Parse SVG path ---
    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
        svg_file_like = io.StringIO(doc["cleaned_svg"])
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
    svg_file_like = io.StringIO(doc["cleaned_svg"])
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


def find_enhanced_closest_curvature(analysis_config, top_k=5):
    """
    calculate close samples and save to db. return the top result

    :param analysis_config:
    :param top_k:
    :return:
    """

    # set vars from analysis_config
    db_handler = analysis_config.get("db_handler")
    sample_id = analysis_config.get("sample_id")
    distance_type_dataset = analysis_config.get("distance_type_dataset")

    # create db handler
    db_handler.use_collection("svg_raw")

    # get the sample from db and its curvature and direction (angle)
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "curvature_data" not in doc:
        return None, None, f"No curvature data for sample_id {sample_id}"

    curvature = np.array(doc["curvature_data"]["curvature"])
    direction = np.array(doc["curvature_data"]["directions"])

    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    # compute distances
    top_matches = calculate_distances(analysis_config, curvature, direction, top_k)

    if not top_matches:
        return None, None, "No comparable samples found."

    db_handler.use_collection("svg_raw")

    # save results
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {"closest_matches": top_matches,
                  "closest_matches_valid": True}
         }
    )

    closest = top_matches[0]
    msg = f"Closest sample to {sample_id} is {closest['id']} (distance={closest['distance']:.6f})"

    return closest["id"], closest["distance"], msg


def calculate_distances(analysis_config, curvature, direction, top_k=5):
    """

    :param direction:
    :param curvature:
    :param analysis_config:
    :param top_k:
    :return:
    """

    # set vars from analysis_config
    db_handler = analysis_config.get("db_handler")
    sample_id = analysis_config.get("sample_id")

    # setup distances list
    distances = []

    # iterate all other samples
    for other_doc in db_handler.collection.find({"sample_id": {"$ne": sample_id}},
                                                {"sample_id": 1, "curvature_data": 1}):
        oid = other_doc["sample_id"]
        curv_data = other_doc.get("curvature_data")
        if not curv_data:
            continue

        # get other samples data
        other_curv = np.array(curv_data["curvature"])
        other_dir = np.array(curv_data["directions"])

        # get distance list from select_arrays (already calculated)
        dist = get_distance(analysis_config, oid, curvature, other_curv,
                            direction, other_dir)

        distances.append((oid, dist))

    # if no results
    if not distances:
        return []

    # sort + top-k results
    distances.sort(key=lambda x: x[1])
    top_matches = [{"id": sid, "distance": float(dist)} for sid, dist in distances[:top_k]]

    return top_matches


def get_distance(analysis_config, oid, curvature, other_curv,
                 direction, other_dir):
    """
    Compute distance between shard and template using different methods.
    """

    # set vars from analysis_config
    sample_id = analysis_config.get("sample_id")
    distance_value_dataset = analysis_config.get("distance_value_dataset")
    distance_calculation = analysis_config.get("distance_calculation")

    # get amount of elements in the cropped 10%
    curve_len = len(curvature)
    crop = int(curve_len * 0.10)
    if curve_len <= 2 * crop:
        return None

    # dataset selection
    if distance_value_dataset == "only curvature":

        return float(sum([apply_metric(curvature, other_curv, distance_calculation)]))

    elif distance_value_dataset == "lip_aligned_angle":

        # validate arrays
        if direction is None:
            return None
        n_shard = len(direction)
        if n_shard < 20:
            return None

        # Convert sample directions to degrees in [-180, 180]
        direction_deg = np.degrees(direction)
        direction_deg = ((direction_deg + 180) % 360) - 180

        # Find candidates zero-crossings for the shard
        shard_candidates = find_all_lip_index_by_angle(direction)
        if len(shard_candidates) == 0:
            return None

        # Load template SVG from DB
        db_handler = MongoDBHandler("svg_data")
        db_handler.use_collection("svg_template_types")
        template_doc = db_handler.collection.find_one({"sample_id": oid})
        if template_doc is None or "raw_content" not in template_doc:
            return None
        raw_template_svg = io.StringIO(template_doc["raw_content"])
        paths, _ = svg2paths(raw_template_svg)
        if len(paths) == 0:
            return None
        path_template = paths[0]

        # Iterative resampling
        min_distance = np.inf
        best_dir_aligned_crop = None
        best_shard_candidate = None
        best_template_candidate = None

        n_resample = 2000
        while n_resample <= 15000:

            # Sample points along template SVG
            ts = np.linspace(0, 1, n_resample)
            points = np.array([path_template.point(t) for t in ts])
            points = np.column_stack((points.real, points.imag))

            # Smooth like in compute_curvature_for_one_sample
            points = normalize_path(points, smooth_method="savgol", smooth_factor=0.02, smooth_window=15)

            # Compute template directions in degrees [-180, 180]
            diffs = np.diff(points, axis=0)
            dir_template = np.arctan2(diffs[:, 1], diffs[:, 0])
            dir_template = np.concatenate(([dir_template[0]], dir_template))
            dir_template_deg = np.degrees(dir_template)
            dir_template_deg = ((dir_template_deg + 180) % 360) - 180

            # Find candidate zero-crossings for the template
            template_candidates = find_all_lip_index_by_angle(dir_template)

            if len(template_candidates) == 0:
                n_resample += 100
                continue

            # Iterate over all candidate pairs
            for shard_idx in shard_candidates:

                for template_idx in template_candidates:

                    # Align template by candidate zero-crossing
                    shift = shard_idx - template_idx
                    dir_aligned_deg = np.roll(dir_template_deg, shift)

                    # Truncate to shard length
                    dir_aligned_crop = dir_aligned_deg[:n_shard]

                    # Angular distance (squared)
                    diffs_angle = direction_deg[:len(dir_aligned_crop)] - dir_aligned_crop
                    distance = np.mean(diffs_angle ** 2)

                    # Track minimum distance
                    if distance < min_distance:
                        min_distance = distance
                        best_dir_aligned_crop = dir_aligned_crop.copy()
                        best_shard_candidate = shard_idx
                        best_template_candidate = template_idx

            n_resample += 100

        if not np.isfinite(min_distance):
            return None

        # Plot only the template with minimum distance
        if best_dir_aligned_crop is not None:
            # plot_lip_alignment(direction_deg, best_dir_aligned_crop, best_shard_candidate, sample_id, oid)
            print("Debug: theory calc")
        return float(min_distance)


    elif distance_value_dataset == "lip_aligned_curvature":
        return None
    elif distance_value_dataset == "cropped curvature":
        return float(sum([apply_metric(curvature[crop:-crop], other_curv[crop:-crop], distance_calculation)]))
    elif distance_value_dataset == "only angle":
        return float(sum([apply_metric(direction, other_dir, distance_calculation)]))
    elif distance_value_dataset == "cropped angle":
        return float(sum([apply_metric(direction[crop:-crop], other_dir[crop:-crop], distance_calculation)]))
    elif distance_value_dataset == "cropped curvature and angle":
        return float(sum([
            apply_metric(curvature[crop:-crop], other_curv[crop:-crop], distance_calculation),
            apply_metric(direction[crop:-crop], other_dir[crop:-crop], distance_calculation)
        ]))

    print("Calculating distance...")

    # default fallback: full curvature + full angle
    return float(sum([
        apply_metric(curvature, other_curv, distance_calculation),
        apply_metric(direction, other_dir, distance_calculation)
    ]))


def apply_metric(a, b, distance_calculation):
    """

    :param a:
    :param b:
    :param distance_calculation:
    :return:
    """

    if distance_calculation == "Euclidean Distance":
        return euclidean_distance(a, b)

    elif distance_calculation == "Cosine Similarity":
        return cosine_similarity_distance(a, b)

    elif distance_calculation == "Correlation Distance":
        return correlation_distance(a, b)

    elif distance_calculation == "dynamic time warping":
        return dtw_distance(a, b)

    elif distance_calculation == "integral difference":
        return integral_difference(a, b)

    else:
        raise ValueError(f"Unknown distance_calculation: {distance_calculation}")


def find_all_lip_index_by_angle(directions, angle_tolerance_deg=5.0, edge_margin=0.05):
    """
    Find all candidate indices where direction is close to horizontal (0°),
    preferring candidates near the middle of the path.

    Returns a list of indices, one per 0° crossing.

    :param directions: np.ndarray of angles in radians
    :param angle_tolerance_deg: tolerance around 0° to consider horizontal
    :param edge_margin: fraction of path length to ignore at both ends
    :return: list of candidate indices
    """
    if len(directions) == 0:
        return []

    # Convert to degrees and normalize to [-180, 180]
    angles_deg = np.degrees(directions)
    angles_deg = ((angles_deg + 180) % 360) - 180

    n = len(angles_deg)
    start = int(edge_margin * n)
    end = int((1.0 - edge_margin) * n)
    idx_range = np.arange(start, end)

    # Absolute deviation from horizontal
    abs_dev = np.abs(angles_deg[idx_range])

    # Candidate points within tolerance
    candidate_mask = abs_dev <= angle_tolerance_deg
    candidate_indices = idx_range[candidate_mask]

    # Filter duplicates per crossing
    # Only keep one candidate per consecutive block
    candidates_filtered = []
    if len(candidate_indices) > 0:
        prev_idx = candidate_indices[0]
        candidates_filtered.append(prev_idx)
        for idx in candidate_indices[1:]:
            if idx - prev_idx > 1:
                candidates_filtered.append(idx)
            prev_idx = idx

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
