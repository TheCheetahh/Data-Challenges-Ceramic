import cv2
import numpy as np
import os
import argparse
import cairosvg
from svgpathtools import svg2paths
import xml.etree.ElementTree as ET





# -------------------------
# Load SVG → OpenCV image (no preprocessing)
# Given that Orb does not work directly on the SVG vector representation it needs to be converted
# -------------------------
def load_svg(svg_input, size=(256, 256)):
    """
    Load SVG from:
        - ET.Element
        - ET.ElementTree
        - SVG string
    and convert to OpenCV image (Grayscale).
    """

    # Convert input to SVG string
    if isinstance(svg_input, ET.Element):
        svg_string = ET.tostring(svg_input, encoding="unicode")

    elif isinstance(svg_input, ET.ElementTree):
        svg_string = ET.tostring(svg_input.getroot(), encoding="unicode")

    elif isinstance(svg_input, str):
        svg_string = svg_input

    else:
        raise TypeError("svg_input must be Element, ElementTree, or SVG string")

    # Render SVG → PNG bytes (in memory)
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=size[0],
        output_height=size[1],
        background_color="white"
    )

    # Convert PNG bytes → OpenCV image
    png_data = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(png_data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not render SVG.")

    return img


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
        print("descriptor None")
        return 1.0

    # Calculating hamming distance between descriptors
    matches = bf.match(des_sample, des_template)

    if not matches:
        print("not matches")
        return 1.0

    # Keeping only matches closer than the threshold (might need some finetuning still)
    good = [m for m in matches if m.distance < distance_threshold]

    # Normalize 
    normalization_factor = min(len(des_sample), len(des_template))

    if normalization_factor == 0:
        print("normalization_factor")
        return 1.0

    # (1.0 = perfect match, 0.0 = no match at all) get switched at return
    normalized_score = len(good) / normalization_factor

    # Return 1 - normalized score because it is sorted by lowest distance.
    return float(1 - normalized_score)


# -------------------------
# Calculate matching score for the Sample and one given Template SVG
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

    # Get score
    score = match_features(des_s, des_t, distance_threshold=50)

    return score

