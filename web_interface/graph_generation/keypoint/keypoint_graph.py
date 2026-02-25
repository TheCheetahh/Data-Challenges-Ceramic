import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import torch

from database_handler import MongoDBHandler

# ORB
from analysis.calculation.keypoint.orb import (
    load_svg as load_svg_orb,
    extract_features as orb_extract,
    match_features as orb_match,
    ransac_filter as orb_ransac
)

# DISK
from analysis.calculation.keypoint.disk import (
    load_svg as load_svg_disk,
    extract_features as disk_extract,
    ransac_filter as disk_ransac
)


# -------------------------------------------------------
# Helper: convert OpenCV image to RGB for matplotlib
# -------------------------------------------------------
def cv_to_rgb(img_gray):
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


# =======================================================
# 1️⃣ KEYPOINT VISUALIZATION
# =======================================================
def generate_keypoint_plot(target_id, target_type="sample", method="ORB"):

    db_handler = MongoDBHandler("svg_data")

    if target_type == "sample":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    doc = db_handler.collection.find_one({"sample_id": target_id})
    if not doc:
        return None, f"❌ {target_type} {target_id} not found"

    svg = doc.get("cropped_svg") or doc.get("cleaned_svg")
    if not svg:
        return None, "❌ No SVG found"

    # Load + extract
    if method == "ORB":
        img = load_svg_orb(svg)
        kp, _ = orb_extract(img)
        points = np.array([k.pt for k in kp]) if kp else np.empty((0, 2))

    elif method == "DISK":
        img = load_svg_disk(svg)
        kp, _ = disk_extract(img)
        points = kp.cpu().numpy() if kp is not None else np.empty((0, 2))

    else:
        return None, "❌ Unknown method"

    # Plot
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv_to_rgb(img))
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], s=10)
    ax.set_title(f"{method} Keypoints — {target_id}")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf), f"✅ {method} keypoints generated for {target_id}"


# =======================================================
# 2️⃣ MATCH VISUALIZATION
# =======================================================
def generate_match_visualization(analysis_config, template_id, method="ORB"):

    db_handler = analysis_config.get("db_handler")
    sample_id = analysis_config.get("sample_id")

    # -------------------------
    # Load sample
    # -------------------------
    db_handler.use_collection("svg_raw")
    sample_doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not sample_doc:
        return None, "❌ Sample not found"

    sample_svg = sample_doc.get("cropped_svg") or sample_doc.get("cleaned_svg")

    # -------------------------
    # Load template
    # -------------------------
    db_handler.use_collection("svg_template_types")
    template_doc = db_handler.collection.find_one({"sample_id": template_id})
    if not template_doc:
        return None, "❌ Template not found"

    template_svg = template_doc.get("cleaned_svg")

    # ====================================================
    # ORB
    # ====================================================
    if method == "ORB":

        img1 = load_svg_orb(sample_svg)
        img2 = load_svg_orb(template_svg)

        kp1, des1 = orb_extract(img1)
        kp2, des2 = orb_extract(img2)

        matches = orb_match(des1, des2)
        inliers = orb_ransac(kp1, kp2, matches)

        # Build OpenCV DMatch list for drawing
        draw_matches = inliers if inliers else []

        vis = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            draw_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    # ====================================================
    # DISK
    # ====================================================
    elif method == "DISK":

        img1 = load_svg_disk(sample_svg)
        img2 = load_svg_disk(template_svg)

        kp1, des1 = disk_extract(img1)
        kp2, des2 = disk_extract(img2)

        if des1 is None or des2 is None:
            return None, "❌ No descriptors"

        dists = torch.cdist(des1, des2)
        min_vals, min_idx = torch.min(dists, dim=1)
        matches = [(i, min_idx[i].item()) for i in range(len(min_vals))]

        inliers = disk_ransac(kp1, kp2, matches)

        # Create canvas manually (like your disk debug)
        h1, w1 = img1.shape
        h2, w2 = img2.shape

        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        vis[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        for i, j in inliers:
            x1, y1 = kp1[i].cpu().numpy()
            x2, y2 = kp2[j].cpu().numpy()

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2 + w1), int(y2))

            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    else:
        return None, "❌ Unknown method"

    # ----------------------------------------------------
    # Convert to PIL image
    # ----------------------------------------------------
    buf = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{method} Matches — {sample_id} vs {template_id}")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Image.open(buf), f"✅ {method} match visualization generated"



def generate_match_visualization_overlay(analysis_config, template_id, method="ORB"):

    db_handler = analysis_config.get("db_handler")
    sample_id = analysis_config.get("sample_id")

    # -------------------------
    # Load sample
    # -------------------------
    db_handler.use_collection("svg_raw")
    sample_doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not sample_doc:
        return None, "❌ Sample not found"

    sample_svg = sample_doc.get("cropped_svg") or sample_doc.get("cleaned_svg")

    # -------------------------
    # Load template
    # -------------------------
    db_handler.use_collection("svg_template_types")
    template_doc = db_handler.collection.find_one({"sample_id": template_id})
    if not template_doc:
        return None, "❌ Template not found"

    template_svg = template_doc.get("cleaned_svg")

    # ====================================================
    # ORB
    # ====================================================
    if method == "ORB":

        img1 = load_svg_orb(sample_svg)
        img2 = load_svg_orb(template_svg)

        kp1, des1 = orb_extract(img1)
        kp2, des2 = orb_extract(img2)

        matches = orb_match(des1, des2)
        inliers = orb_ransac(kp1, kp2, matches)

        total_matches = len(matches)
        inlier_count = len(inliers)

        draw_matches = inliers if inliers else []

        vis = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            draw_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    # ====================================================
    # DISK
    # ====================================================
    elif method == "DISK":

        img1 = load_svg_disk(sample_svg)
        img2 = load_svg_disk(template_svg)

        kp1, des1 = disk_extract(img1)
        kp2, des2 = disk_extract(img2)

        if des1 is None or des2 is None:
            return None, "❌ No descriptors"

        dists = torch.cdist(des1, des2)
        min_vals, min_idx = torch.min(dists, dim=1)
        matches = [(i, min_idx[i].item()) for i in range(len(min_vals))]

        inliers = disk_ransac(kp1, kp2, matches)

        total_matches = len(matches)
        inlier_count = len(inliers)

        h1, w1 = img1.shape
        h2, w2 = img2.shape

        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        vis[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        for i, j in inliers:
            x1, y1 = kp1[i].cpu().numpy()
            x2, y2 = kp2[j].cpu().numpy()

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2 + w1), int(y2))

            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    else:
        return None, "❌ Unknown method"

    # ====================================================
    # 🔥 MATCH SCORE OVERLAY
    # ====================================================

    inlier_ratio = inlier_count / max(1, total_matches)
    distance = 1.0 - inlier_ratio

    overlay_text = [
        f"Sample: {sample_id}",
        f"Template: {template_id}",
        f"Total Matches: {total_matches}",
        f"Inliers: {inlier_count}",
        f"Inlier Ratio: {inlier_ratio:.4f}",
        f"Distance: {distance:.4f}"
    ]

    y0 = 30
    for i, line in enumerate(overlay_text):
        cv2.putText(
            vis,
            line,
            (20, y0 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # ----------------------------------------------------
    # Convert to PIL image
    # ----------------------------------------------------
    buf = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Image.open(buf), (
        f"✅ {method} match visualization | "
        f"Inlier ratio={inlier_ratio:.4f} | Distance={distance:.4f}"
    )