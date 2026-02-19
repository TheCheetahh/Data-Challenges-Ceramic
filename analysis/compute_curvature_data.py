import io
import math

import numpy as np
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path, curvature_from_points
from analysis.calculation.laa.laa_calcualtion import laa_calculation

from analysis.calculation.ipc.icp import ensure_icp_geometry, run_icp, icp_score

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
            # instead of none it should call the function that returns the distance
            distances.append((template_id, None))

        elif distance_value_dataset == "lip_aligned_angle":
            distances.append((template_id, laa_calculation(analysis_config, template_doc, template_id)))

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
    distances = [x for x in distances if x[1] is not None]
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
