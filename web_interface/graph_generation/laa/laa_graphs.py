import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path
from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import remove_svg_fill, format_svg_for_display


def laa_get_template_svg(target_id):

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_template_types")

    closest_svg_content, closest_error = db_handler.get_cleaned_svg(target_id)

    closest_svg_no_fill = remove_svg_fill(closest_svg_content)
    return format_svg_for_display(closest_svg_no_fill), f"✅ Template loaded for {target_id}"


def laa_generate_curvature_lineplot(target_id, target_type):

    db_handler = MongoDBHandler("svg_data")

    if target_type == "sample":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    sample_id = target_id
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    curvature_data = doc["curvature_data"]

    """Generate curvature line plot with lip markers."""
    arc_lengths = np.array(curvature_data["arc_lengths"])
    curvature = np.array(curvature_data["curvature"])
    lip_anchor = curvature_data.get("lip_anchor", {})
    lip_angle_arc = lip_anchor.get("angle", {}).get("arc_length")
    lip_curv_arc = lip_anchor.get("curvature", {}).get("arc_length")

    buf = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", linestyle="--")
    plt.plot(arc_lengths, curvature, color="black", label="Curvature")

    if lip_angle_arc is not None:
        plt.axvline(lip_angle_arc, color="blue", linestyle="--", linewidth=2, label="Lip (angle)")
    if lip_curv_arc is not None:
        plt.axvline(lip_curv_arc, color="red", linestyle="--", linewidth=2, label="Lip (curvature)")

    plt.title(f"Curvature along normalized arc length (sample {sample_id})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Curvature κ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf), f"✅ Curvature plot generated for sample_id {sample_id}"


def laa_generate_curvature_color_map(analysis_config, target_id, target_type):

    db_handler = MongoDBHandler("svg_data")

    if target_type == "sample":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    sample_id = target_id
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    curvature_data = doc["curvature_data"]

    """Generate curvature color map overlaid on the shard geometry."""
    curvature = np.array(curvature_data["curvature"])
    smooth_method = curvature_data.get("settings", {}).get("smooth_method", analysis_config.get("smooth_method"))
    smooth_factor = curvature_data.get("settings", {}).get("smooth_factor", analysis_config.get("smooth_factor"))
    smooth_window = curvature_data.get("settings", {}).get("smooth_window", analysis_config.get("smooth_window"))

    svg_file_like = io.StringIO(doc.get("cropped_svg") or doc.get("cleaned_svg"))
    paths, _ = svg2paths(svg_file_like)
    path = paths[0]

    ts = np.linspace(0, 1, len(curvature))
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))
    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=-np.max(np.abs(curvature)), vmax=np.max(np.abs(curvature)) * 0.8)

    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    lc.set_array(curvature)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.invert_yaxis()
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Curvature Color Map")
    plt.colorbar(lc, ax=ax, label="Curvature κ")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf), f"✅ Curvature color map generated for sample_id {sample_id}"


def laa_generate_direction_lineplot(target_id, target_type):

    db_handler = MongoDBHandler("svg_data")

    if target_type == "sample":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    sample_id = target_id
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    curvature_data = doc["curvature_data"]

    """Generate direction (angle) plot with lip markers."""
    arc_lengths = np.array(curvature_data["arc_lengths"])
    directions = np.unwrap(np.array(curvature_data["directions"]))
    directions_deg = np.degrees(directions)
    lip_anchor = curvature_data.get("lip_anchor", {})
    lip_angle_arc = lip_anchor.get("angle", {}).get("arc_length")
    lip_curv_arc = lip_anchor.get("curvature", {}).get("arc_length")

    buf = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.plot(arc_lengths, directions_deg, color="blue", label="Direction")

    if lip_angle_arc is not None:
        plt.axvline(lip_angle_arc, color="blue", linestyle="--", linewidth=2, label="Lip (angle)")
    if lip_curv_arc is not None:
        plt.axvline(lip_curv_arc, color="red", linestyle="--", linewidth=2, label="Lip (curvature)")

    plt.title(f"Direction along normalized arc length (sample {sample_id})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Angle to x-axis [°]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf), f"✅ Direction plot generated for sample_id {sample_id}"
