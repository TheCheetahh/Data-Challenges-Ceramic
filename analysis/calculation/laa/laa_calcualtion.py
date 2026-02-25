import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path

from database_handler import MongoDBHandler

DEBUG_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "debug_plots")


def save_direction_plot(sample_id, template_id, direction_deg, dir_aligned_crop, best_shard_idx, best_template_idx):
    os.makedirs(DEBUG_PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(range(len(direction_deg)), direction_deg, color="red", linewidth=1.2, label=f"Sample ({sample_id})")
    ax.plot(range(len(dir_aligned_crop)), dir_aligned_crop, color="blue", linewidth=1.2, label=f"Template ({template_id})")

    ax.axvline(x=best_shard_idx, color="green", linestyle="--", linewidth=1.0, alpha=0.7, label=f"Sample 0° point (idx {best_shard_idx})")

    ax.set_xlabel("Point index")
    ax.set_ylabel("Direction (degrees)")
    ax.set_title(f"Direction comparison: {sample_id} vs {template_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-185, 185)

    filename = f"{sample_id}_vs_{template_id}.png".replace("/", "_").replace(" ", "_")
    filepath = os.path.join(DEBUG_PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=100)
    plt.close(fig)
    print(f"Saved debug plot: {filepath}")


def laa_calculation(analysis_config, template_doc, template_id):

    sample_id = analysis_config.get("sample_id")
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "curvature_data" not in doc:
        return None
    sample_curvature = np.array(doc["curvature_data"]["curvature"])
    sample_direction = np.array(doc["curvature_data"]["directions"])

    n_sample_points = analysis_config.get("n_samples")
    if sample_direction is None:
        return None
    sample_direction_len = len(sample_direction)

    direction_deg = np.degrees(sample_direction)
    direction_deg = ((direction_deg + 180) % 360) - 180

    shard_candidates = find_all_lip_index_by_angle(sample_direction)
    if len(shard_candidates) == 0:
        print("No shard candidates found")
        return None

    raw_template_svg = io.StringIO(template_doc["raw_content"])
    paths, _ = svg2paths(raw_template_svg)
    if len(paths) == 0:
        return None
    path_template = paths[0]

    min_distance = np.inf
    best_n_resample = None
    best_shard_idx = None
    best_template_idx = None
    best_aligned_crop = None

    left = n_sample_points
    right = 40000

    while right - left > 1:
        mid1 = left + (right - left) // 3
        mid2 = left + 2 * (right - left) // 3

        dist1, crop1, shard1, temp1 = compute_distance_for_resample(
            path_template, shard_candidates, direction_deg, sample_direction_len, mid1
        )
        dist2, crop2, shard2, temp2 = compute_distance_for_resample(
            path_template, shard_candidates, direction_deg, sample_direction_len, mid2
        )

        if dist1 < dist2:
            right = mid2 - 1
            if dist1 < min_distance:
                min_distance = dist1
                best_n_resample = mid1
                best_shard_idx = shard1
                best_template_idx = temp1
                best_aligned_crop = crop1
        else:
            left = mid1 + 1
            if dist2 < min_distance:
                min_distance = dist2
                best_n_resample = mid2
                best_shard_idx = shard2
                best_template_idx = temp2
                best_aligned_crop = crop2

    if not np.isfinite(min_distance):
        return None

    # Save direction comparison plot
    if best_aligned_crop is not None:
        save_direction_plot(sample_id, template_id, direction_deg, best_aligned_crop, best_shard_idx, best_template_idx)

    # Save visualization data to database
    db_handler.use_collection("svg_raw")
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$push": {
            "laa_overlap_data": (template_id, int(best_n_resample), int(best_shard_idx), int(best_template_idx))}}
    )

    return float(min_distance)


def find_all_lip_index_by_angle(directions, angle_tolerance_deg=3.0, edge_margin=0.05):
    if len(directions) == 0:
        return []

    angles_deg = np.degrees(directions)
    angles_deg = ((angles_deg + 180) % 360) - 180

    n = len(angles_deg)
    start = int(edge_margin * n)
    end = int((1.0 - edge_margin) * n)
    idx_range = np.arange(start, end)

    abs_dev = np.abs(angles_deg[idx_range])

    from scipy.signal import find_peaks

    min_distance = max(1, int(0.01 * n))

    peaks, _ = find_peaks(-abs_dev, distance=min_distance)

    candidates = [idx_range[p] for p in peaks if abs_dev[p] <= angle_tolerance_deg]

    if len(candidates) == 0:
        fallback_idx = idx_range[np.argmin(abs_dev)]
        return [fallback_idx]

    return candidates


def compute_distance_for_resample(path_template, shard_candidates, direction_deg, n_shard, n_resample):

    ts = np.linspace(0, 1, n_resample)
    points = np.array([path_template.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    smooth_window = max(5, int(0.005 * n_resample))
    points = normalize_path(points, smooth_method="savgol", smooth_factor=2, smooth_window=smooth_window)

    diffs = np.diff(points, axis=0)
    dir_template = np.arctan2(diffs[:, 1], diffs[:, 0])
    dir_template = np.concatenate(([dir_template[0]], dir_template))
    dir_template_deg = np.degrees(dir_template)
    dir_template_deg = ((dir_template_deg + 180) % 360) - 180

    template_candidates = find_all_lip_index_by_angle(dir_template)
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
            diffs_angle = ((diffs_angle + 180) % 360) - 180
            distance_val = np.mean(diffs_angle ** 2)
            if distance_val < local_min:
                local_min = distance_val
                local_best_crop = dir_aligned_crop.copy()
                local_shard_candidate = shard_idx
                local_template_candidate = template_idx

    return local_min, local_best_crop, local_shard_candidate, local_template_candidate