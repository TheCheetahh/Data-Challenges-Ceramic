import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from analysis.calculation.ipc.icp import ensure_icp_geometry, run_icp, icp_score
from analysis.calculation.laa.laa_calcualtion import laa_calculation
from database_handler import MongoDBHandler

def _compute_distance_process(template_doc, safe_config):
    template_id = template_doc["sample_id"]
    start_ts = time.perf_counter()

    try:
        distance_value_dataset = safe_config["distance_value_dataset"]

        # Keypoints
        if distance_value_dataset == "Keypoints":
            return template_id, None

        # Lip aligned angle
        elif distance_value_dataset == "lip_aligned_angle":
            local_config = dict(safe_config)
            local_config["db_handler"] = MongoDBHandler("svg_data")

            return template_id, laa_calculation(
                local_config, template_doc, template_id
            )

        # ICP
        elif distance_value_dataset == "ICP":
            db_handler = MongoDBHandler("svg_data")

            sample_id = safe_config["sample_id"]
            n_target = safe_config.get("icp_n_target", 300)
            n_ref = safe_config.get("icp_n_reference", 500)

            icp_params = safe_config["icp_params"]

            # sample
            try:
                db_handler.use_collection("svg_raw")
                target_doc = db_handler.collection.find_one(
                    {"sample_id": sample_id}
                )
                if target_doc is None:
                    return template_id, float("inf")

                target_icp = ensure_icp_geometry(
                    target_doc, db_handler, n_target
                )
                target_pts = np.array(target_icp["outline_points"])
            except Exception:
                return template_id, float("inf")

            # template
            try:
                db_handler.use_collection("svg_template_types")
                ref_doc = db_handler.collection.find_one(
                    {"sample_id": template_id}
                )
                if ref_doc is None:
                    return template_id, float("inf")

                ref_icp = ensure_icp_geometry(
                    ref_doc, db_handler, n_ref
                )
                ref_pts = np.array(ref_icp["outline_points"])
            except Exception:
                return template_id, float("inf")

            # run icp
            try:
                err, aligned = run_icp(
                    target_pts,
                    ref_pts,
                    iters=icp_params["iters"],
                    max_total_deg=icp_params["max_total_deg"],
                    max_scale_step=icp_params["max_scale_step"]
                )

                if not np.isfinite(err):
                    return template_id, float("inf")

                score, _ = icp_score(
                    ref_pts, aligned, ref_id=template_id
                )

                if not np.isfinite(score):
                    return template_id, float("inf")

                return template_id, float(score)

            except Exception:
                return template_id, float("inf")

        else:
            return template_id, None

    finally:
        end_ts = time.perf_counter()
        """print(
            f"[PID {os.getpid()}] template={template_id} "
            f"time={(end_ts - start_ts):.3f}s"
        )"""

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
    
    templates = list(
        db_handler.collection.find(
            {"sample_id": {"$ne": sample_id}},
            {"sample_id": 1, "curvature_data": 1}
        )
    )
    # Using safe_config instead as parallelization doesn't work with same db handler
    safe_config = {
        "sample_id": analysis_config["sample_id"],
        "distance_value_dataset": analysis_config["distance_value_dataset"],
        "n_samples": analysis_config.get("n_samples"),

        # ICP parameters
        "icp_n_target": analysis_config.get("icp_n_target", 300),
        "icp_n_reference": analysis_config.get("icp_n_reference", 500),
        "icp_params": {
            "iters": analysis_config.get("icp_iters", 30),
            "max_total_deg": analysis_config.get("icp_max_deg", 2.0),
            "max_scale_step": analysis_config.get("icp_max_scale", 0.2),
            "top_percent": analysis_config.get("icp_top_percent", 0.2),
        },
    }

    if distance_value_dataset == "lip_aligned_angle":
        # parallel
        with ProcessPoolExecutor(
            max_workers=min(os.cpu_count(), 4)
        ) as executor:
            futures = [
                executor.submit(
                    _compute_distance_process, template_doc, safe_config
                )
                for template_doc in templates
            ]

            for future in as_completed(futures):
                distances.append(future.result())

    else:
        # sequential
        for template_doc in templates:
            distances.append(
                _compute_distance_process(template_doc, safe_config)
            )

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
