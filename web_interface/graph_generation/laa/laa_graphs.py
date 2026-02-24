import matplotlib.pyplot as plt
import matplotlib

from analysis.calculation.laa.laa_calcualtion import find_all_lip_index_by_angle
from analysis.analyze_curvature import normalize_path
matplotlib.use("Agg")
from PIL import Image
import io
import numpy as np
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


def visualize_laa_overlap(analysis_config, template_id):
    db_handler = analysis_config.get("db_handler")
    sample_id = analysis_config.get("sample_id")
    n_samples = analysis_config.get("n_samples")

    # Load sample doc
    db_handler.use_collection("svg_raw")
    sample_doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not sample_doc:
        return None, f"Sample {sample_id} not found"

    # Find the overlap data for this specific template
    laa_overlap_data = sample_doc.get("laa_overlap_data", [])
    overlap_entry = next((e for e in laa_overlap_data if e[0] == template_id), None)
    if overlap_entry is None:
        return None, f"No overlap data found for template {template_id}"

    _, best_n_resample, best_shard_idx, best_template_idx = overlap_entry

    # Generate sample points from SVG at n_samples points (same as calculation)
    sample_svg = sample_doc.get("cropped_svg") or sample_doc.get("cleaned_svg")
    if not sample_svg:
        return None, "No SVG found for sample"
    sample_paths, _ = svg2paths(io.StringIO(sample_svg))
    if len(sample_paths) == 0:
        return None, "No paths in sample SVG"
    sample_path = sample_paths[0]
    sample_ts = np.linspace(0, 1, n_samples)
    sample_points = np.array([sample_path.point(t) for t in sample_ts])
    sample_points = np.column_stack((sample_points.real, sample_points.imag))

    # Apply same smoothing as when curvature was calculated
    smooth_window_sample = max(5, int(0.005 * n_samples))
    sample_points = normalize_path(sample_points, smooth_method="savgol", smooth_factor=2, smooth_window=smooth_window_sample)

    # Load template SVG
    db_handler.use_collection("svg_template_types")
    template_doc = db_handler.collection.find_one({"sample_id": template_id})
    if not template_doc or "raw_content" not in template_doc:
        return None, f"Template {template_id} not found"
    template_paths, _ = svg2paths(io.StringIO(template_doc["raw_content"]))
    if len(template_paths) == 0:
        return None, "No paths in template SVG"
    path_template = template_paths[0]

    # Sample points on template at best_n_resample points (same as calculation)
    ts = np.linspace(0, 1, best_n_resample)
    template_points = np.array([path_template.point(t) for t in ts])
    template_points = np.column_stack((template_points.real, template_points.imag))

    # Apply same smoothing as in the calculation
    smooth_window_template = max(5, int(0.005 * best_n_resample))
    template_points = normalize_path(template_points, smooth_method="savgol", smooth_factor=2, smooth_window=smooth_window_template)

    # Roll template the same way the distance calculation does
    shift = best_shard_idx - best_template_idx
    template_rolled = np.roll(template_points, shift, axis=0)

    # Align both at best_shard_idx
    actual_shard_idx = min(best_shard_idx, len(sample_points) - 1)
    sample_aligned = sample_points - sample_points[actual_shard_idx]
    template_aligned = template_rolled - template_rolled[actual_shard_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(sample_aligned[:, 0], sample_aligned[:, 1], color="orange", linewidth=1.5, label="Sample")
    ax.plot(template_aligned[:, 0], template_aligned[:, 1], color="blue", linewidth=1.5, label="Template")
    ax.scatter([0], [0], color="red", zorder=5, label="Alignment point (0°)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f"LAA Overlap: {sample_id} vs {template_id}")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf), "OK"