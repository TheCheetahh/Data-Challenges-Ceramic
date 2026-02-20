import math

import numpy as np

from analysis.calculation.ipc.icp import ensure_icp_geometry, run_icp, icp_score
from analysis.calculation.laa.laa_calcualtion import laa_calculation


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
