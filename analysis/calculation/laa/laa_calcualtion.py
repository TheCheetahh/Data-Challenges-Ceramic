import io

import numpy as np
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path

from database_handler import MongoDBHandler


def laa_calculation(analysis_config, template_doc, template_id):

    # set vars from analysis_config
    sample_id = analysis_config.get("sample_id")
    top_k = analysis_config.get("top_k")
    distance_value_dataset = analysis_config.get("distance_value_dataset")
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_template_types")

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
        return None

    # get other samples data
    template_curvature = np.array(curv_data["curvature"])
    template_direction = np.array(curv_data["directions"])

    # get amount of elements in the cropped 10%
    curve_len = len(sample_curvature)
    crop = int(curve_len * 0.10)
    if curve_len <= 2 * crop:
        return None

    n_samples = analysis_config.get("n_samples")
    # validate arrays
    if sample_direction is None:
        return None
    n_shard = len(sample_direction)
    if n_shard < 20:
        return None

    # Convert sample directions to degrees in [-180, 180]
    direction_deg = np.degrees(sample_direction)
    direction_deg = ((direction_deg + 180) % 360) - 180

    # Find candidates zero-crossings for the shard
    shard_candidates = find_all_lip_index_by_angle(sample_direction)
    if len(shard_candidates) == 0:
        print("No shard candidates found")
        return None

    # Load template SVG from DB
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_template_types")
    template_doc = db_handler.collection.find_one({"sample_id": template_id})
    if template_doc is None or "raw_content" not in template_doc:
        return None
    raw_template_svg = io.StringIO(template_doc["raw_content"])
    paths, _ = svg2paths(raw_template_svg)
    if len(paths) == 0:
        return None
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
        return None

    if best_dir_aligned_crop is not None:
        # print("Debug: theory calc")
        pass
    return float(min_distance)


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