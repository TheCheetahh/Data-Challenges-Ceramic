from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display, remove_svg_fill, crop_svg_path

import math
import re
import xml.etree.ElementTree as ET


def _parse_svg_points(points_str: str):
    """Parse an SVG polyline/polygon 'points' attribute into a list of (x, y) floats."""
    if not points_str:
        return []

    # SVG allows either "x,y x,y" or "x y x y" or mixed whitespace/newlines.
    s = points_str.replace(",", " ")
    parts = [p for p in s.split() if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        # Fallback for exotic formatting.
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]

    if len(nums) % 2 != 0:
        raise ValueError("Invalid SVG points attribute (odd number of coordinates).")

    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _format_svg_points(points):
    """Format points back into a compact SVG-compatible 'points' string."""
    return " ".join(f"{x:.6g} {y:.6g}" for x, y in points)


def _rebase_closed_contours_to_lowest_y(svg_str: str, *, close_factor: float = 5.0, close_ratio: float = 0.10) -> str:
    """Rotate near-closed polylines/polygons so the visually lowest point (max y) is start/end.

    For closed/circular contours, the stored start/end can be arbitrary (often near the top).
    For cropping-at-ends we want a deterministic seam, so we choose the bottom-most point.

    Heuristics for "closed":
      - <polygon> is always closed
      - polyline with fill != 'none' is treated as closed
      - or end-to-start distance is small relative to bbox (close_ratio) and/or average segment length (close_factor)
    """
    if not svg_str or "<svg" not in svg_str:
        return svg_str

    # Preserve optional XML declaration.
    xml_decl = ""
    m = re.match(r"\s*(<\?xml[^>]*\?>\s*)", svg_str)
    if m:
        xml_decl = m.group(1)
        svg_body = svg_str[m.end():]
    else:
        svg_body = svg_str

    try:
        root = ET.fromstring(svg_body)
    except Exception:
        # Fallback: try parsing the full string.
        root = ET.fromstring(svg_str)
        xml_decl = ""

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0].strip("{")
        # Avoid ns0 prefixes on serialization
        ET.register_namespace("", ns)

    def q(tag: str) -> str:
        return f"{{{ns}}}{tag}" if ns else tag

    def is_nearly_closed(points):
        if len(points) < 3:
            return False
        x0, y0 = points[0]
        x1, y1 = points[-1]
        end_dist = math.hypot(x1 - x0, y1 - y0)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        diag_thresh = close_ratio * diag if diag > 0 else 0.0

        segs = [
            math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)
        ]
        avg_seg = (sum(segs) / len(segs)) if segs else 0.0
        seg_thresh = close_factor * avg_seg if avg_seg > 0 else 0.0

        thresh = max(diag_thresh, seg_thresh)
        return end_dist <= thresh

    changed = False
    for elem in root.iter():
        if elem.tag not in (q("polyline"), q("polygon")):
            continue

        pts_attr = elem.get("points")
        if not pts_attr:
            continue

        points = _parse_svg_points(pts_attr)
        if len(points) < 3:
            continue

        # Determine if this contour should be treated as closed.
        closed = (elem.tag == q("polygon"))
        if not closed:
            fill = (elem.get("fill") or "").strip().lower()
            if fill and fill != "none":
                closed = True
        if not closed:
            closed = is_nearly_closed(points)

        if not closed:
            continue

        # If last point duplicates the first, drop it before rotating.
        if math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) < 1e-9:
            points = points[:-1]

        # "Lowest" in SVG space is max y.
        idx = max(range(len(points)), key=lambda i: points[i][1])
        points = points[idx:] + points[:idx]

        # Explicitly close by repeating the first point.
        points = points + [points[0]]

        elem.set("points", _format_svg_points(points))
        changed = True

    if not changed:
        return svg_str

    out = ET.tostring(root, encoding="unicode")
    if xml_decl and not out.lstrip().startswith("<?xml"):
        out = xml_decl + out
    return out


def _crop_svg_path_with_bottom_seam(full_svg: str, crop_start: float, crop_end: float) -> str:
    """Wrapper around crop_svg_path that normalizes closed contours to start/end at bottom."""
    rebased = _rebase_closed_contours_to_lowest_y(full_svg)
    return crop_svg_path(rebased, crop_start, crop_end)


def change_crop_svg_dropdown(sample_id):
    """Called when dropdown changes - loads saved settings from database"""
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Get the document from database
    document = db_handler.collection.find_one({"sample_id": sample_id})

    if not document:
        # print(f"No document found for sample_id {sample_id}")
        return "", "", 0.0, 1.0

    full_svg = document.get("cleaned_svg") or document.get("svg")

    # Load saved crop settings if they exist, otherwise use defaults
    saved_crop_start = document.get("crop_start")
    saved_crop_end = document.get("crop_end")

    # Use saved values if available, otherwise use default values (0.0 and 1.0)
    crop_start = saved_crop_start if saved_crop_start is not None else 0.0
    crop_end = saved_crop_end if saved_crop_end is not None else 1.0

    # print(f"Loaded crop settings: start={crop_start}, end={crop_end}")

    # Remove fill from full SVG
    full_svg_no_fill = remove_svg_fill(full_svg)
    display_full_svg = format_svg_for_display(full_svg_no_fill)

    # Create cropped SVG (already has fill removed in crop_svg_path)
    cropped_svg = _crop_svg_path_with_bottom_seam(full_svg, crop_start, crop_end)
    display_cropped_svg = format_svg_for_display(cropped_svg)

    return display_full_svg, display_cropped_svg, crop_start, crop_end


def update_crop_preview(sample_id, crop_start, crop_end):
    """Called when sliders change - uses current slider values"""
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Get the document from database
    document = db_handler.collection.find_one({"sample_id": sample_id})

    if not document:
        # print(f"No document found for sample_id {sample_id}")
        return "", ""

    full_svg = document.get("cleaned_svg") or document.get("svg")

    # Remove fill from full SVG
    full_svg_no_fill = remove_svg_fill(full_svg)
    display_full_svg = format_svg_for_display(full_svg_no_fill)

    # Create cropped SVG with current slider values
    cropped_svg = _crop_svg_path_with_bottom_seam(full_svg, crop_start, crop_end)
    display_cropped_svg = format_svg_for_display(cropped_svg)

    return display_full_svg, display_cropped_svg