from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import crop_svg_path, remove_svg_fill

import math
import re
import xml.etree.ElementTree as ET


def _parse_svg_points(points_str: str):
    if not points_str:
        return []

    s = points_str.replace(",", " ")
    parts = [p for p in s.split() if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]

    if len(nums) % 2 != 0:
        raise ValueError("Invalid SVG points attribute (odd number of coordinates).")

    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _format_svg_points(points):
    return " ".join(f"{x:.6g} {y:.6g}" for x, y in points)


def _rebase_closed_contours_to_lowest_y(svg_str: str, *, close_factor: float = 5.0, close_ratio: float = 0.10) -> str:
    """Rotate near-closed polylines/polygons so the visually lowest point (max y) is start/end."""
    if not svg_str or "<svg" not in svg_str:
        return svg_str

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
        root = ET.fromstring(svg_str)
        xml_decl = ""

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0].strip("{")
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

        return end_dist <= max(diag_thresh, seg_thresh)

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

        closed = (elem.tag == q("polygon"))
        if not closed:
            fill = (elem.get("fill") or "").strip().lower()
            if fill and fill != "none":
                closed = True
        if not closed:
            closed = is_nearly_closed(points)

        if not closed:
            continue

        if math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) < 1e-9:
            points = points[:-1]

        idx = max(range(len(points)), key=lambda i: points[i][1])
        points = points[idx:] + points[:idx]
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
    rebased = _rebase_closed_contours_to_lowest_y(full_svg)
    return crop_svg_path(rebased, crop_start, crop_end)


def click_save_cropped_svg(sample_id, crop_start, crop_end):
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    full_svg, error = db_handler.get_cleaned_svg(sample_id)

    # Check if SVG was retrieved successfully
    if error or not full_svg:
        # print(f"Error retrieving SVG for sample_id {sample_id}: {error}")
        return f"⚠ Error retrieving SVG for {sample_id}"

    # Create cropped SVG (already has fill removed in crop_svg_path)
    cropped_svg = _crop_svg_path_with_bottom_seam(full_svg, crop_start, crop_end)

    # Remove fill from cropped SVG before saving
    cropped_svg = remove_svg_fill(cropped_svg)

    # Update database
    result = db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set":
            {
                "cropped_svg": cropped_svg,
                "crop_start": crop_start,
                "crop_end": crop_end,
                "outdated_curvature": True,
                "icp_data": None
            }
        }
    )

    # Print confirmation
    if result.modified_count > 0:
        # print(f"✓ Saved cropped SVG for sample_id {sample_id} (crop: {crop_start} to {crop_end})")
        return f"✓ Saved cropped SVG for {sample_id}"
    else:
        # print(f"⚠ No document updated for sample_id {sample_id}")
        return f"⚠ No document updated for {sample_id}"

