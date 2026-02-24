from xml.etree import ElementTree as ET

import math
import re


def format_svg_for_display(cleaned_svg):
    """
    Wrap the SVG in a bordered white box for display on the web page.

    :param cleaned_svg: cleaned SVG
    """
    return f"""
    <div style="
        border: 2px solid black;
        background-color: white;
        padding: 10px;
        width: 500px;
        height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        {cleaned_svg}
    </div>
    """


def remove_svg_fill(svg_string):
    """
    Remove fill from SVG polyline to show only the stroke.

    :param svg_string: SVG as string
    :return: Modified SVG string with fill removed
    """

    # Register namespace to avoid ns0: prefixes
    ET.register_namespace('', 'http://www.w3.org/2000/svg')

    try:
        # Parse the SVG
        root = ET.fromstring(svg_string)

        # Find the polyline element
        polyline = root.find('.//{http://www.w3.org/2000/svg}polyline')

        if polyline is None:
            # Try without namespace
            polyline = root.find('.//polyline')

        if polyline is not None:
            # Remove fill and ensure stroke is visible
            if 'fill' in polyline.attrib:
                del polyline.attrib['fill']
            polyline.set('fill', 'none')

        # Convert back to string
        return ET.tostring(root, encoding='unicode', method='xml')
    except Exception as e:
        # print(f"Warning: Could not remove fill from SVG: {e}")
        return svg_string  # Return original on error


def crop_svg_path(svg_string, crop_start, crop_end):
    """Crop an SVG polyline/polygon by keeping only the segment between crop_start and crop_end.

    Zusätzliche Logik für kreisrunde/geschlossene Konturen:
    - Anfang/Ende werden als *verbunden* behandelt (explizit geschlossen)
    - der visuell *tiefste* Punkt wird als neuer Start/Endpunkt gesetzt

    Hinweis: In SVG-Koordinaten wächst y nach unten. "Tiefster" Punkt => max(y).
    """

    def _parse_points(points_str: str):
        if not points_str:
            return []
        s = points_str.replace(',', ' ')
        parts = [p for p in s.split() if p.strip()]
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
        if len(nums) % 2 != 0:
            return []
        return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]

    def _fmt_points(points):
        return ' '.join(f"{x:.6g} {y:.6g}" for x, y in points)

    def _is_nearly_closed(points, *, close_ratio: float = 0.10, close_factor: float = 5.0):
        if len(points) < 3:
            return False
        (x0, y0), (x1, y1) = points[0], points[-1]
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

    # Register namespace to avoid ns0: prefixes
    ET.register_namespace('', 'http://www.w3.org/2000/svg')

    # Parse the SVG
    root = ET.fromstring(svg_string)

    # Find the element
    poly = root.find('.//{http://www.w3.org/2000/svg}polyline')
    if poly is None:
        poly = root.find('.//polyline')
    if poly is None:
        poly = root.find('.//{http://www.w3.org/2000/svg}polygon')
    if poly is None:
        poly = root.find('.//polygon')
    if poly is None:
        return svg_string

    points = _parse_points(poly.get('points'))
    if len(points) < 2:
        return svg_string

    # Decide if contour is "closed-like"
    is_polygon = poly.tag.endswith('polygon')
    fill = (poly.get('fill') or '').strip().lower()
    closed_like = is_polygon or (fill not in ('', 'none')) or _is_nearly_closed(points)

    eps = 1e-9
    if closed_like and len(points) >= 3:
        # Unique vertices (drop explicit closing duplicate if present)
        if math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]) < eps:
            unique = points[:-1]
        else:
            unique = points

        # Rebase seam to bottom-most point
        idx = max(range(len(unique)), key=lambda i: unique[i][1])
        unique = unique[idx:] + unique[:idx]

        # Explicitly close the contour for stroke rendering
        closed_points = unique + [unique[0]]
        base_for_crop = unique
    else:
        closed_points = None
        base_for_crop = points

    # Clamp and compute indices over the *unique* list
    crop_start_f = float(crop_start)
    crop_end_f = float(crop_end)
    total_points = len(base_for_crop)
    start_idx = int(total_points * crop_start_f)
    end_idx = int(total_points * crop_end_f)
    start_idx = max(0, min(start_idx, total_points - 1))
    end_idx = max(start_idx + 1, min(end_idx, total_points))

    full_range = (crop_start_f <= 0.0) and (crop_end_f >= 1.0)
    if closed_like and full_range and closed_points is not None:
        cropped_points = closed_points
    else:
        cropped_points = base_for_crop[start_idx:end_idx]

    if not cropped_points:
        return svg_string

    # Update points
    poly.set('points', _fmt_points(cropped_points))

    # Remove fill and ensure stroke is visible
    if 'fill' in poly.attrib:
        del poly.attrib['fill']
    poly.set('fill', 'none')

    # Recalculate viewBox to fit cropped path
    xs = [p[0] for p in cropped_points]
    ys = [p[1] for p in cropped_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 2
    min_x -= padding
    min_y -= padding
    width = (max_x - min_x) + 2 * padding
    height = (max_y - min_y) + 2 * padding

    root.set('viewBox', f"{min_x} {min_y} {width} {height}")
    root.set('width', f"{width}mm")
    root.set('height', f"{height}mm")

    return ET.tostring(root, encoding='unicode', method='xml')
