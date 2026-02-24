import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import cv2
import numpy as np
import torch
torch.set_num_threads(1) #Avoid conflicts
import xml.etree.ElementTree as ET
import kornia
import kornia.feature as KF
import cairosvg
import matplotlib.pyplot as plt

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


# -------------------------
# Load SVG → OpenCV image (no preprocessing)
# Given that DISK does not work directly on the SVG vector representation it needs to be converted
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
# Torch image prep
# -------------------------
def to_tensor(img):
    t = torch.from_numpy(img).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0)
    t = t.repeat(1, 3, 1, 1)
    return t.to(device)


# -------------------------
# DISK extractor
# -------------------------
disk = KF.DISK.from_pretrained("depth").to(device).eval()


# -------------------------
# Feature extraction + score filtering
# -------------------------
def extract_features(img, max_kp=800):
    with torch.no_grad():
        t = to_tensor(img)
        feats = disk(t)[0]

    kp = feats.keypoints
    des = feats.descriptors

    if des is None or len(des) == 0:
        return None, None

    # Use descriptor norm as strength proxy
    strength = torch.norm(des, dim=1)

    # Keep strongest K
    if len(strength) > max_kp:
        idx = torch.argsort(strength, descending=True)[:max_kp]
        kp = kp[idx]
        des = des[idx]

    return kp, des



# -------------------------
# Draw keypoints
# -------------------------
def draw_keypoints(img, kp):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if kp is None:
        return vis

    pts = kp.cpu().numpy()

    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    return vis


# -------------------------
# Lowe Ratio Matching
# -------------------------
def match_lowe(des_q, des_t, ratio=0.95):
    if des_q is None or des_t is None:
        return []

    dists = torch.cdist(des_q, des_t, p=2)

    vals, idx = torch.topk(dists, k=2, largest=False)

    good = vals[:, 0] < ratio * vals[:, 1]

    matches = []

    for i in range(len(good)):
        if good[i]:
            matches.append((i, idx[i, 0].item()))

    return matches


# -------------------------
# RANSAC verification
# -------------------------
def ransac_filter(kp_q, kp_t, matches, thresh=3.0):

    if len(matches) < 4:
        return []

    src = []
    dst = []

    for i, j in matches:
        src.append(kp_q[i].cpu().numpy())
        dst.append(kp_t[j].cpu().numpy())

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
# Draw match lines
# -------------------------
def draw_matches(img1, kp1, img2, kp2, matches):

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for i, j in matches:
        x1, y1 = kp1[i].cpu().numpy()
        x2, y2 = kp2[j].cpu().numpy()

        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w1), int(y2))

        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(canvas, pt1, 3, (0, 0, 255), -1)
        cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)

    return canvas


def disk_distance(analysis_config, template_doc, template_id):
    
    db_handler = analysis_config.get("db_handler")

    # Load sample SVG and cache keypoints and descriptors
    if "disk_sample_features" not in analysis_config:

        db_handler.use_collection("svg_raw")
        sample_id = analysis_config.get("sample_id")
        doc = db_handler.collection.find_one({"sample_id": sample_id})

        sample_svg = doc.get("cropped_svg") or doc.get("cleaned_svg")
        sample = load_svg(sample_svg)

        kp_s, des_s = extract_features(sample)

        analysis_config["disk_sample_features"] = (kp_s, des_s)

    kp_s, des_s = analysis_config["disk_sample_features"]


    # Load template SVG and extract keypoints and descriptors
    template_svg = template_doc.get("cleaned_svg")

    if template_svg is None:
        print(f"Skipping template {template_id}: no SVG found")
        return None
    template = load_svg(template_svg)

    kp_t, des_t = extract_features(template)

    if des_s is None or des_t is None:
        print("Sample des None")
        return 1.0


    # Matching
    #matches = match_lowe(des_s, des_t)
    dists = torch.cdist(des_s, des_t)
    min_vals, min_idx = torch.min(dists, dim=1)
    matches = [(i, min_idx[i].item()) for i in range(len(min_vals))]
    
    inliers = ransac_filter(kp_s, kp_t, matches)

    # Normalization (important)
    normalized_score = len(inliers) / max(1, len(matches))
    distance = 1.0 - normalized_score
    
    # Return distance (like ORB)
    return float(distance)