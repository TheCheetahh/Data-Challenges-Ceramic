import os
import re
import xml.etree.ElementTree as ET
import argparse

from web_interface.formating_functions.format_svg import crop_svg_path


# ---------------------------------------------------------
# Path Complexity
# ---------------------------------------------------------

def estimate_path_complexity(d):
    """Estimate SVG path complexity by number of numeric coordinates."""
    if not d:
        return 0
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    return len(numbers)


# ---------------------------------------------------------
# Cleaning Logic (Legacy CLI version)
# ---------------------------------------------------------

def filter_most_complex_black_fill(input_path, output_path):
    """
    Legacy CLI function:
    Keep only the most complex black-filled object in the SVG.
    """

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(input_path)
    root = tree.getroot()

    ns = {"svg": root.tag.split('}')[0].strip('{')}
    all_elements = root.findall(".//*", ns)

    black_elements = []

    for el in all_elements:
        fill = el.attrib.get("fill", "").strip().lower()

        if fill in ("#000000", "black"):
            if "d" in el.attrib:
                complexity = estimate_path_complexity(el.attrib["d"])
            elif el.tag.endswith(("polygon", "polyline")):
                points = el.attrib.get("points", "")
                complexity = len(re.findall(r"[-+]?\d*\.?\d+", points))
            elif el.tag.endswith(("rect", "circle", "ellipse")):
                complexity = 4
            else:
                complexity = 0

            black_elements.append((el, complexity))

    if not black_elements:
        print("⚠️ Keine schwarzen Flächen gefunden.")
        return

    most_complex = max(black_elements, key=lambda x: x[1])[0]

    new_svg = ET.Element(root.tag, root.attrib)
    new_svg.append(most_complex)
    ET.ElementTree(new_svg).write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Gespeichert: {output_path}")


# ---------------------------------------------------------
# Theory Mode → Pick most complex non-black shape
# ---------------------------------------------------------

def find_best_non_black_shape(all_elements):
    """
    Used for theory templates: pick most complex visible shape.
    """

    candidates = []

    for el in all_elements:
        tag = el.tag.lower()

        if tag.endswith("path") and "d" in el.attrib:
            complexity = estimate_path_complexity(el.attrib["d"])
            candidates.append((el, complexity))

        elif tag.endswith(("polygon", "polyline")):
            points = el.attrib.get("points", "")
            complexity = len(re.findall(r"[-+]?\d*\.?\d+", points))
            candidates.append((el, complexity))

        elif tag.endswith(("rect", "circle", "ellipse")):
            candidates.append((el, 4))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])[0]


# ---------------------------------------------------------
# MAIN CLEANER (used in DB)
# ---------------------------------------------------------

def get_most_complex_black_fill(raw_svg):
    """
    For samples: returns the most complex black-filled shape.
    For theory types: if no black exists → pick the most complex shape.
    """

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(raw_svg)

    all_elements = root.findall(".//*")

    # -------------------------
    # Try SAMPLE mode (black shapes)
    # -------------------------
    black_elements = []

    for el in all_elements:
        fill = el.attrib.get("fill", "").strip().lower()

        if fill in ("#000000", "black"):

            if "d" in el.attrib:
                complexity = estimate_path_complexity(el.attrib["d"])
            elif el.tag.lower().endswith(("polygon", "polyline")):
                points = el.attrib.get("points", "")
                complexity = len(re.findall(r"[-+]?\d*\.?\d+", points))
            elif el.tag.lower().endswith(("rect", "circle", "ellipse")):
                complexity = 4
            else:
                complexity = 0

            black_elements.append((el, complexity))

    # -------------------------
    # CASE A: sample mode
    # -------------------------
    if black_elements:
        most_complex = max(black_elements, key=lambda x: x[1])[0]

    # -------------------------
    # CASE B: theory mode
    # -------------------------
    else:
        most_complex = find_best_non_black_shape(all_elements)
        if most_complex is None:
            return None

    # -------------------------
    # Build new SVG containing only the selected shape
    # -------------------------
    new_svg = ET.Element(root.tag, root.attrib)
    new_svg.append(most_complex)

    return ET.tostring(new_svg, encoding="unicode")


# ---------------------------------------------------------
# Bulk cleaner used by DB
# ---------------------------------------------------------

def clean_all_svgs(db_handler, svg_file_type):
    """Create cleaned SVGs for all in DB."""

    collection = (
        db_handler.use_collection("svg_raw")
        if svg_file_type == "sample"
        else db_handler.use_collection("svg_template_types")
    )

    docs = db_handler.collection.find({})
    counter = 0

    for doc in docs:
        raw_content = doc.get("raw_content")

        if not raw_content or doc.get("cleaned_svg"):
            continue

        cleaned_svg = get_most_complex_black_fill(raw_content)

        if not cleaned_svg:
            print(f"⚠️ Cropping failed for {doc.get('filename')}")
            cropped_svg = None
            crop_start = None
            crop_end = None
        else:
            crop_start = 0.05
            crop_end = 0.95
            cropped_svg = crop_svg_path(cleaned_svg, crop_start, crop_end)

        if cleaned_svg:
            counter += 1
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"cleaned_svg": cleaned_svg,
                          "cropped_svg": cropped_svg,
                          "crop_start": crop_start,
                          "crop_end": crop_end,
                          "outdated_curvature": True,
                          "icp_data": None
                          }}
            )
            print(f"✅ Updated {doc['filename']}")
        else:
            print(f"⚠️ No usable shape found in {doc['filename']}")

    return f"{counter} svgs cleaned"


# ---------------------------------------------------------
# Standalone CLI mode
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVG cleanup tool")
    parser.add_argument("--input_svg", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="cleaned_svg/")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.input_svg))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = args.output_dir + base_name + ".svg"
    filter_most_complex_black_fill(args.input_svg, output_path)
