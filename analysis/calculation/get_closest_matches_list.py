import math

import numpy as np

from analysis.calculation.icp.icp import compute_icp_distance
from analysis.calculation.laa.laa_calcualtion import laa_calculation
from concurrent.futures import ProcessPoolExecutor


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
    # laa is here because this is parallel
    distances = []
    if distance_value_dataset == "lip_aligned_angle":

        template_docs = list(db_handler.collection.find(
            {"sample_id": {"$ne": sample_id}},
            {"sample_id": 1, "curvature_data": 1}
        ))

        db_handler = analysis_config.pop("db_handler")

        with ProcessPoolExecutor() as executor:
            distances = list(executor.map(
                compute_distance,
                template_docs,
                [analysis_config] * len(template_docs)
            ))

        analysis_config["db_handler"] = db_handler

    else:
        # non parallel
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

            # cannot be called because this is the old sequential one, but maybe we will need it one day
            elif distance_value_dataset == "lip_aligned_angle":
                distances.append((template_id, laa_calculation(analysis_config, template_doc, template_id)))

            elif distance_value_dataset == "ICP":
                dist = compute_icp_distance(
                    analysis_config["db_handler"],
                    sample_id,
                    template_id,
                    analysis_config
                )
                distances.append((template_id, dist))
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



def compute_distance(template_doc, analysis_config):
    template_id = template_doc["sample_id"]
    return template_id, laa_calculation(analysis_config, template_doc, template_id)

