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

        # Find polyline/polygon element
        elem = root.find('.//{http://www.w3.org/2000/svg}polyline')
        if elem is None:
            elem = root.find('.//{http://www.w3.org/2000/svg}polygon')
        if elem is None:
            elem = root.find('.//polyline')
        if elem is None:
            elem = root.find('.//polygon')

        if elem is not None:
            # Remove fill and ensure stroke is visible
            if 'fill' in elem.attrib:
                del elem.attrib['fill']
            elem.set('fill', 'none')

        # Convert back to string
        return ET.tostring(root, encoding='unicode', method='xml')
    except Exception:
        return svg_string  # Return original on error


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_points(points_str: str):
    """
    Parse an SVG polyline/polygon 'points' attribute into a list of (x, y) floats.
    Handles commas, whitespace, newlines, and scientific notation.
    """
    if not points_str:
        return []

    s = points_str.replace(",", " ")
    parts = [p for p in s.split() if p.strip()]
    nums = None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        nums = [float(x) for x in _FLOAT_RE.findall(s)]

    if len(nums) < 4 or len(nums) % 2 != 0:
        return []

    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_diag(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _is_close(a, b, tol):
    return _dist(a, b) <= tol


def _unique_open_points(points, tol):
    """
    For a potentially closed polyline (first == last), return the unique open list.
    """
    if len(points) >= 2 and _is_close(points[0], points[-1], tol):
        return points[:-1]
    return points


def _is_closed_contour(elem, points, tol):
    """
    Determine whether we should treat the contour as closed for seam normalization.
    """
    # polygon is closed by definition
    if elem is not None and elem.tag.lower().endswith('polygon'):
        return True

    # filled polyline/polygon often indicates closed contour intent
    fill = None
    if elem is not None:
        fill = elem.get('fill')
    if fill is not None and fill.strip().lower() not in ('', 'none'):
        return True

    # heuristic: endpoints nearly coincide
    if len(points) >= 3 and _is_close(points[0], points[-1], tol):
        return True

    return False


def _total_loop_length(points):
    """
    points: open unique points (no duplicate closure).
    Returns loop length including closing segment last->first.
    """
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += _dist(points[i], points[i + 1])
    total += _dist(points[-1], points[0])
    return total


def _seam_point_from_deepest(points, seam_position):
    """
    points: open unique points (no closure duplicate), in traversal order.
    seam_position: float in [0,100]. 50 => deepest point (max y).
                  <50 => reverse direction; >50 => forward direction.
                  Distance offset from deepest is proportional to |pos-50|/50 of loop length.

    Returns: (new_points_open, closed_requested)
        new_points_open: open list starting at seam point (no duplicate closure at end)
        closed_requested: bool (whether we reversed direction)
    """
    if len(points) < 3:
        return points, False

    pos = float(seam_position) if seam_position is not None else 50.0

    # Reverse direction if below 50
    reversed_dir = pos < 50.0
    pts = list(reversed(points)) if reversed_dir else list(points)

    # deepest point in SVG coordinates: maximum y
    deep_idx = max(range(len(pts)), key=lambda i: pts[i][1])

    loop_len = _total_loop_length(pts)
    if loop_len <= 1e-9:
        # degenerate
        return pts, reversed_dir

    offset_fraction = abs(pos - 50.0) / 100.0  # 0..0.5 (52 => +2% of loop)
    target_dist = offset_fraction * loop_len

    # walk forward from deepest point along the closed loop
    n = len(pts)
    curr_idx = deep_idx
    dist_left = target_dist

    # if target_dist ~ 0, seam at deepest vertex
    if dist_left <= 1e-9:
        seam_pt = pts[deep_idx]
        seg_start_idx = (deep_idx - 1) % n  # seam is at vertex; treat as after seg_start_idx
        # Rotate so seam_pt is first
        rotated = pts[deep_idx:] + pts[:deep_idx]
        return rotated, reversed_dir

    while True:
        next_idx = (curr_idx + 1) % n
        a = pts[curr_idx]
        b = pts[next_idx]
        seg_len = _dist(a, b)
        if seg_len <= 1e-12:
            curr_idx = next_idx
            # avoid infinite loops on bad data
            if curr_idx == deep_idx:
                break
            continue

        if dist_left <= seg_len:
            t = dist_left / seg_len
            seam_pt = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            seg_start_idx = curr_idx
            break

        dist_left -= seg_len
        curr_idx = next_idx
        if curr_idx == deep_idx:
            # completed loop
            seam_pt = pts[deep_idx]
            seg_start_idx = (deep_idx - 1) % n
            break

    # Build rotated open list starting at seam point, then continue with points after the segment
    # seam lies on segment seg_start_idx -> seg_start_idx+1
    s = seg_start_idx
    # Points after seam start at (s+1)%n
    tail = pts[s + 1 :] if (s + 1) < n else []
    head = pts[: s + 1]  # includes pts[s] (segment start) at end
    rotated_open = [seam_pt] + tail + head

    # Remove immediate duplicate seam if it matches next point
    if len(rotated_open) >= 2 and _is_close(rotated_open[0], rotated_open[1], 1e-9):
        rotated_open.pop(0)

    return rotated_open, reversed_dir


def crop_svg_path(svg_string, crop_start, crop_end, seam_position=50.0):
    """
    Crop an SVG polyline/polygon to keep only the segment between crop_start and crop_end.
    For (nearly) closed contours, start/end are re-declared at a seam point:
      - default seam_position=50 selects the visually lowest point (max y).
      - seam_position > 50 walks forward along the path from that point.
      - seam_position < 50 reverses traversal direction (and walks forward in that reversed direction).

    Args:
        svg_string: Full SVG as string
        crop_start: Start position as fraction (0.0-0.5)
        crop_end: End position as fraction (0.51-1.0)
        seam_position: float in [0,100], default 50

    Returns:
        Cropped SVG string
    """

    # Register namespace to avoid ns0: prefixes
    ET.register_namespace('', 'http://www.w3.org/2000/svg')

    # Parse the SVG
    root = ET.fromstring(svg_string)

    # Find the polyline/polygon element
    elem = root.find('.//{http://www.w3.org/2000/svg}polyline')
    if elem is None:
        elem = root.find('.//{http://www.w3.org/2000/svg}polygon')
    if elem is None:
        elem = root.find('.//polyline')
    if elem is None:
        elem = root.find('.//polygon')

    if elem is None:
        return svg_string  # Return original if no element found

    points_str = elem.get('points') or ''
    points = _parse_points(points_str)
    if len(points) < 2:
        return svg_string

    diag = _bbox_diag(points) if len(points) >= 2 else 0.0
    tol = max(1e-6, diag * 0.01)  # 1% of bbox diagonal, min epsilon

    closed = _is_closed_contour(elem, points, tol)

    # If closed, normalize seam (start/end) using deepest point + seam slider
    if closed and len(points) >= 3:
        open_pts = _unique_open_points(points, tol)
        seam_open_pts, _ = _seam_point_from_deepest(open_pts, seam_position)
        points_for_crop = seam_open_pts  # open representation
    else:
        points_for_crop = points

    total_points = len(points_for_crop)
    if total_points < 2:
        return svg_string

    # Calculate indices from fractions (0.0-1.0 range)
    start_idx = int(total_points * float(crop_start))
    end_idx = int(total_points * float(crop_end))

    # Ensure valid range
    start_idx = max(0, min(start_idx, total_points - 1))
    end_idx = max(start_idx + 1, min(end_idx, total_points))

    cropped_points = points_for_crop[start_idx:end_idx]
    if not cropped_points:
        return svg_string

    # If full range on closed contour, close it again (stroke closure)
    if closed and float(crop_start) <= 0.0 and float(crop_end) >= 1.0:
        if not _is_close(cropped_points[0], cropped_points[-1], tol):
            cropped_points = list(cropped_points) + [cropped_points[0]]

    # Convert back to string format
    cropped_points_str = ' '.join([f"{x} {y}" for x, y in cropped_points])
    elem.set('points', cropped_points_str)

    # Remove fill and ensure stroke is visible
    if 'fill' in elem.attrib:
        del elem.attrib['fill']
    elem.set('fill', 'none')

    # Recalculate viewBox to fit cropped path
    xs = [p[0] for p in cropped_points]
    ys = [p[1] for p in cropped_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add padding
    padding = 2
    min_x -= padding
    min_y -= padding
    width = (max_x - min_x) + 2 * padding
    height = (max_y - min_y) + 2 * padding

    root.set('viewBox', f"{min_x} {min_y} {width} {height}")
    root.set('width', f"{width}mm")
    root.set('height', f"{height}mm")

    return ET.tostring(root, encoding='unicode', method='xml')
