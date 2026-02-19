from xml.etree import ElementTree as ET


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
    """
    Crop an SVG path to keep only the segment between crop_start and crop_end.

    Args:
        svg_string: Full SVG as string
        crop_start: Start position as fraction (0.0-0.5)
        crop_end: End position as fraction (0.51-1.0)

    Returns:
        Cropped SVG string
    """

    # Register namespace to avoid ns0: prefixes
    ET.register_namespace('', 'http://www.w3.org/2000/svg')

    # Parse the SVG
    root = ET.fromstring(svg_string)

    # Find the polyline element (or path if using path instead)
    polyline = root.find('.//{http://www.w3.org/2000/svg}polyline')

    if polyline is None:
        # Try without namespace
        polyline = root.find('.//polyline')

    if polyline is None:
        # print("ERROR: No polyline found in SVG")
        return svg_string  # Return original if no polyline found

    # Get the points
    points_str = polyline.get('points')
    # print(f"Original points count: {len(points_str.split()) // 2}")

    # Parse points into list of (x, y) tuples
    points = []
    coords = points_str.strip().split()
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            points.append((float(coords[i]), float(coords[i + 1])))

    # Calculate indices from fractions (0.0-1.0 range)
    total_points = len(points)
    start_idx = int(total_points * crop_start)
    end_idx = int(total_points * crop_end)

    """print(f"Total points: {total_points}")
    print(f"Crop range: {crop_start} to {crop_end}")
    print(f"Keeping points {start_idx} to {end_idx} ({end_idx - start_idx} points)")"""

    # Ensure valid range
    start_idx = max(0, min(start_idx, total_points - 1))
    end_idx = max(start_idx + 1, min(end_idx, total_points))

    # Crop the points - keep segment between start and end
    cropped_points = points[start_idx:end_idx]

    if not cropped_points:
        print("ERROR: No points after cropping!")
        return svg_string

    # Convert back to string format
    cropped_points_str = ' '.join([f"{x} {y}" for x, y in cropped_points])

    # Update the polyline with cropped points
    polyline.set('points', cropped_points_str)

    # Remove fill and ensure stroke is visible
    if 'fill' in polyline.attrib:
        del polyline.attrib['fill']
    polyline.set('fill', 'none')  # Explicitly set fill to none

    # Recalculate viewBox to fit cropped path
    xs = [p[0] for p in cropped_points]
    ys = [p[1] for p in cropped_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add some padding
    padding = 2
    min_x -= padding
    min_y -= padding
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding

    # print(f"New viewBox: {min_x} {min_y} {width} {height}")

    root.set('viewBox', f"{min_x} {min_y} {width} {height}")
    root.set('width', f"{width}mm")
    root.set('height', f"{height}mm")

    # Convert back to string
    cropped_svg = ET.tostring(root, encoding='unicode', method='xml')

    """print("Cropped SVG:")
    print(cropped_svg[:500])  # Print first 500 chars
    print("...")"""

    return cropped_svg
