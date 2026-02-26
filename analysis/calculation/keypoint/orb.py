import cv2
import numpy as np
import os
import argparse
import cairosvg
from svgpathtools import svg2paths
import xml.etree.ElementTree as ET

from analysis.calculation.keypoint.convert_svg_to_opencv import load_svg

# -------------------------
# Show image
# For debugging as a tool to see what the algorithm sees
# -------------------------
def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# -------------------------
# ORB feature extractor
# -------------------------
def extract_features(img):
    
    orb = cv2.ORB_create(nfeatures=2000)
    
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


# -------------------------
# Show ORB keypoints
# For debugging as a tool to see what features of the image orb uses for keypoints
# -------------------------
def show_keypoints(img, keypoints, title="Keypoints"):
    vis = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------
# Match ORB features
# Calulation of distance and nomalization
# -------------------------
def match_features(des_sample, des_template, distance_threshold=50):
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if des_sample is None or des_template is None:
        return 1.0

    # Calculating hamming distance between descriptors
    matches = bf.match(des_sample, des_template)

    if not matches:
        return 1.0

    # Keeping only matches closer than the threshold (might need some finetuning still)
    good = [m for m in matches if m.distance < distance_threshold]


    # Return list of close matches
    return good


# -------------------------
# RANSAC verification
# -------------------------
def ransac_filter(kp_q, kp_t, matches, thresh=3.0):

    if len(matches) < 4:
        return []

    src = []
    dst = []

    for m in matches:
        i = m.queryIdx
        j = m.trainIdx

        src.append(kp_q[i].pt)
        dst.append(kp_t[j].pt)

    src = np.float32(src)
    dst = np.float32(dst)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)

    if mask is None:
        return []

    inliers = []

    for i, m in enumerate(mask.ravel()):
        if m:
            inliers.append(matches[i])

    return inliers


# -------------------------
# Calculate matching score for the Sample and one given Template SVG
# Should safe Orb keypoints, descriptors and List of closes matches in MongoDB for huge speedup
# and graph generation
# -------------------------
def orb_distance(analysis_config, template_doc, template_id):

    # Extracting essentials from the config list and setting sample image for matching
    db_handler = analysis_config.get("db_handler")
    db_handler.use_collection("svg_raw")
    sample_id = analysis_config.get("sample_id")
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    sample_svg = doc.get("cropped_svg") or doc.get("cleaned_svg")
    sample = load_svg(sample_svg)

    # Setting template image
    template_svg = template_doc.get("cleaned_svg")
    if template_svg is None:
        print(f"Skipping template {template_id}: no SVG found")
        return None
    template = load_svg(template_svg)

    # Extracting Keypoints and descriptors from the sample and template
    kp_s, des_s = extract_features(sample)
    kp_t, des_t = extract_features(template)

    # Get matches and check for geometric consistancy using RANSAC
    matches = match_features(des_s, des_t, distance_threshold=50) 
    inliers = ransac_filter(kp_s, kp_t, matches)

    # Normalization (important)
    normalized_score = len(inliers) / max(1, len(matches))
    distance = 1.0 - normalized_score

    return float(distance)

