import os
from svgpathtools import svg2paths2, Path
import xml.etree.ElementTree as ET


def crop_and_center_svg(
    input_svg,
    output_svg,
    canvas_size=512,
    margin=10
):
    paths, attributes, svg_attr = svg2paths2(input_svg)

    if len(paths) != 1:
        raise ValueError("SVG muss genau einen Pfad enthalten")

    path: Path = paths[0]

    xmin, xmax, ymin, ymax = path.bbox()
    width = xmax - xmin
    height = ymax - ymin

    target_inner = canvas_size - 2 * margin

    scale = min(
        target_inner / width,
        target_inner / height
    )

    path = path.translated(-xmin - 1j * ymin)
    path = path.scaled(scale)

    new_width = width * scale
    new_height = height * scale

    offset_x = (canvas_size - new_width) / 2
    offset_y = (canvas_size - new_height) / 2
    path = path.translated(offset_x + 1j * offset_y)

    svg = ET.Element(
        "svg",
        attrib={
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(canvas_size),
            "height": str(canvas_size),
            "viewBox": f"0 0 {canvas_size} {canvas_size}",
        }
    )

    ET.SubElement(
        svg,
        "path",
        attrib={
            "d": path.d(),
            "fill": "none",
            "stroke": "black",
        }
    )

    tree = ET.ElementTree(svg)
    tree.write(
        output_svg,
        encoding="utf-8",
        xml_declaration=True
    )


def process_svg_folder(
    input_folder,
    output_folder,
    canvas_size=512,
    margin=10
):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".svg"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            crop_and_center_svg(
                input_path,
                output_path,
                canvas_size,
                margin
            )
            print(f"[OK] {filename}")
        except Exception as e:
            print(f"[SKIP] {filename}: {e}")


process_svg_folder(
    "cleaned_svg/",
    "Centered_svgs2/",
    canvas_size=512,
    margin=10
)
