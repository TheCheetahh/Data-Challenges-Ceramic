import math

import numpy as np

from analysis.calculation.icp.icp import compute_icp_distance
from analysis.calculation.laa.laa_calcualtion import laa_calculation
from analysis.calculation.keypoint.orb import orb_distance
from analysis.calculation.keypoint.disk import disk_distance
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


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
    batch_mode = analysis_config.get("batch_mode", False)

    template_docs = list(db_handler.collection.find(
        {"sample_id": {"$ne": sample_id}},
        {"sample_id": 1, "curvature_data": 1, "raw_content": 1, "cleaned_svg": 1}
    ))

    template_ids = [doc["sample_id"] for doc in template_docs]
    distances = []

    # compute distances
    # laa is here because this is parallel
    distances = []
    if distance_value_dataset == "lip_aligned_angle":

        # Clear old overlap data before recalculating
        db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$unset": {"laa_overlap_data": ""}}
        )

        db_handler = analysis_config.pop("db_handler")

        if batch_mode:
            distances = [
                compute_distance(doc, analysis_config)
                for doc in template_docs
            ]
        else:
            with ProcessPoolExecutor() as executor:
                distances = list(executor.map(
                    compute_distance,
                    template_docs,
                    [analysis_config] * len(template_docs)
                ))

        analysis_config["db_handler"] = db_handler

    elif distance_value_dataset == "ICP":
        if batch_mode:
            dists = [
                compute_icp_distance(db_handler, sample_id, tid, analysis_config)
                for tid in template_ids
            ]
        else:
            with ThreadPoolExecutor() as executor:
                dists = list(executor.map(
                    compute_icp_distance,
                    [db_handler] * len(template_ids),
                    [sample_id] * len(template_ids),
                    template_ids,
                    [analysis_config] * len(template_ids),
                ))

        distances = list(zip(template_ids, dists))

    # Closest matches with Orb
    elif distance_value_dataset == "Orb":
        for template_doc in template_docs:
            template_id = template_doc["sample_id"]
            distances.append((template_id, orb_distance(analysis_config, template_doc, template_id)))

    elif distance_value_dataset == "DISK":
        for template_doc in template_docs:
            template_id = template_doc["sample_id"]
            distances.append((template_id, disk_distance(analysis_config, template_doc, template_id)))

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



def compute_distance(template_doc, analysis_config):
    template_id = template_doc["sample_id"]
    return template_id, laa_calculation(analysis_config, template_doc, template_id)

