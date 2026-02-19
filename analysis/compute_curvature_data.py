import io
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path, curvature_from_points

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

    :param current_sample_id:
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


def get_closest_matches_list(analysis_config):
    """
    calculate close samples and save to db. return the top result

    :param analysis_config:
    :return:
    """

    # set vars from analysis_config
    sample_id = analysis_config.get("sample_id")
    top_k = analysis_config.get("top_k")
    distance_value_dataset = analysis_config.get("distance_value_dataset")
    db_handler = analysis_config.get("db_handler")
    db_handler.use_collection("svg_template_types")

    # compute distances
    # setup distances list
    distances = []
    # iterate all templates, fill distances[] with results
    for template_doc in db_handler.collection.find({"sample_id": {"$ne": sample_id}},
                                                   {"sample_id": 1, "curvature_data": 1}):
        template_id = template_doc["sample_id"]

        # dataset selection
        # Marco code placeholder
        if distance_value_dataset == "Keypoints":
            distances.append((template_id, None))
            continue
        elif distance_value_dataset == "lip_aligned_angle":

            db_handler.use_collection("svg_raw")

            # get the sample from db and its curvature and direction (angle)
            doc = db_handler.collection.find_one({"sample_id": sample_id})
            if not doc or "curvature_data" not in doc:
                return None, None, f"No curvature data for sample_id {sample_id}"

            sample_curvature = np.array(doc["curvature_data"]["curvature"])
            sample_direction = np.array(doc["curvature_data"]["directions"])

            db_handler.use_collection("svg_template_types")

            curv_data = template_doc.get("curvature_data")
            if not curv_data:
                continue

            # get other samples data
            template_curvature = np.array(curv_data["curvature"])
            template_direction = np.array(curv_data["directions"])

            # get amount of elements in the cropped 10%
            curve_len = len(sample_curvature)
            crop = int(curve_len * 0.10)
            if curve_len <= 2 * crop:
                distances.append((template_id, None))
                continue

            n_samples = analysis_config.get("n_samples")
            # validate arrays
            if sample_direction is None:
                distances.append((template_id, None))
                continue
            n_shard = len(sample_direction)
            if n_shard < 20:
                distances.append((template_id, None))
                continue

            # Convert sample directions to degrees in [-180, 180]
            direction_deg = np.degrees(sample_direction)
            direction_deg = ((direction_deg + 180) % 360) - 180

            # Find candidates zero-crossings for the shard
            shard_candidates = find_all_lip_index_by_angle(sample_direction)
            if len(shard_candidates) == 0:
                print("No shard candidates found")
                distances.append((template_id, None))
                continue

            # Load template SVG from DB
            db_handler = MongoDBHandler("svg_data")
            db_handler.use_collection("svg_template_types")
            template_doc = db_handler.collection.find_one({"sample_id": template_id})
            if template_doc is None or "raw_content" not in template_doc:
                distances.append((template_id, None))
                continue
            raw_template_svg = io.StringIO(template_doc["raw_content"])
            paths, _ = svg2paths(raw_template_svg)
            if len(paths) == 0:
                distances.append((template_id, None))
                continue
            path_template = paths[0]

            # Ternary search over n_resample
            min_distance = np.inf
            best_dir_aligned_crop = None
            best_shard_candidate = None
            best_template_candidate = None

            left = n_samples  # start from usual n_samples
            right = 20000  # maximum resample

            while left < right:

                mid1 = left + (right - left) // 3
                mid2 = left + 2 * (right - left) // 3
                dist1, crop1, shard1, temp1 = compute_distance_for_resample(
                    path_template, shard_candidates, direction_deg, n_shard, mid1
                )
                dist2, crop2, shard2, temp2 = compute_distance_for_resample(
                    path_template, shard_candidates, direction_deg, n_shard, mid2
                )

                # Shrink search interval
                if dist1 < dist2:
                    right = mid2 - 1
                else:
                    left = mid1 + 1

                # Track overall minimum
                if dist1 < min_distance:
                    min_distance = dist1
                    best_dir_aligned_crop = crop1
                    best_shard_candidate = shard1
                    best_template_candidate = temp1
                if dist2 < min_distance:
                    min_distance = dist2
                    best_dir_aligned_crop = crop2
                    best_shard_candidate = shard2
                    best_template_candidate = temp2

            if not np.isfinite(min_distance):
                distances.append((template_id, None))
                continue

            if best_dir_aligned_crop is not None:
                # print("Debug: theory calc")
                pass
            distances.append((template_id, float(min_distance)))
            continue
        elif distance_value_dataset == "ICP":
            db_handler = analysis_config["db_handler"]

            n_target = analysis_config.get("icp_n_target", 300)
            n_ref = analysis_config.get("icp_n_reference", 500)

            icp_params = {
                "iters": analysis_config.get("icp_iters", 30),
                "max_total_deg": analysis_config.get("icp_max_deg", 2.0),
                "max_scale_step": analysis_config.get("icp_max_scale", 0.2),
                "top_percent": analysis_config.get("icp_top_percent", 0.2)
            }

            # --------------------------------------------------
            # Load target geometry (FAIL HERE = target invalid)
            # --------------------------------------------------
            try:
                db_handler.use_collection("svg_raw")
                target_doc = db_handler.collection.find_one({"sample_id": sample_id})
                if target_doc is None:
                    raise ValueError("Target document not found")

                target_icp = ensure_icp_geometry(target_doc, db_handler, n_target)
                target_pts = np.array(target_icp["outline_points"])
            except Exception as e:
                # Target is unsuitable for ICP → all distances = inf
                skipped = analysis_config.setdefault("icp_skipped_targets", [])
                skipped.append({
                    "id": template_id,
                    "reason": str(e)
                })
                distances.append((template_id, float("inf")))
                continue

            # --------------------------------------------------
            # Load reference geometry (per-template failures OK)
            # --------------------------------------------------
            try:
                db_handler.use_collection("svg_template_types")
                ref_doc = db_handler.collection.find_one({"sample_id": template_id})
                if ref_doc is None:
                    distances.append((template_id, float("inf")))
                    continue

                ref_icp = ensure_icp_geometry(ref_doc, db_handler, n_ref)
                ref_pts = np.array(ref_icp["outline_points"])
            except Exception:
                distances.append((template_id, float("inf")))
                continue

            # --------------------------------------------------
            # Run ICP + score
            # --------------------------------------------------
            try:
                err, aligned = run_icp(
                    target_pts,
                    ref_pts,
                    iters=icp_params["iters"],
                    max_total_deg=icp_params["max_total_deg"],
                    max_scale_step=icp_params["max_scale_step"]
                )

                if not np.isfinite(err):
                    distances.append((template_id, float("inf")))
                    continue

                score, _ = icp_score(ref_pts, aligned, ref_id=template_id)
                if not np.isfinite(score):
                    skipped = analysis_config.setdefault("icp_skipped_targets", [])
                    skipped.append({
                        "id": template_id,
                        "reason": "non-finite ICP score"
                    })
                    distances.append((template_id, float("inf")))
                    continue

                distances.append((template_id, float(score)))
                continue

            except Exception:
                distances.append((template_id, float("inf")))
                continue
        else:
            print("invalid distance_value_dataset")
            distances.append((template_id, None))
            continue

    # sort
    distances.sort(key=lambda x: x[1])
    # populate top results. Leave out inf and anything beyond top_k
    top_matches = []
    for temp_id, dist in distances:
        if not math.isfinite(dist) or (top_k is not None and len(top_matches) >= top_k):
            break
        top_matches.append({"id": temp_id, "distance": float(dist)})

    if not top_matches:
        return None, None, "No comparable samples found."

    db_handler.use_collection("svg_raw")
    # save results
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {"closest_matches": top_matches,
                  "full_closest_matches": top_matches,
                  "closest_matches_valid": True}
         }
    )

    closest = top_matches[0]
    msg = f"Closest sample to {sample_id} is {closest['id']} (distance={closest['distance']:.6f})"

    print(closest["id"], " ", closest["distance"])

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