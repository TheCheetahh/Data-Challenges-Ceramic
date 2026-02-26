import argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import cairosvg
import matplotlib.pyplot as plt




# -------------------------
# Load SVG → OpenCV image (no preprocessing)
# Given that DISK and Orb are not working directly on the SVG vector representation it needs to be converted
# -------------------------
def load_svg(svg_input, size=(256, 256)):
    """
    Load SVG from:
        - ET.Element
        - ET.ElementTree
        - SVG string
    and convert to OpenCV image (Grayscale).
    """

    # Convert input to SVG string
    if isinstance(svg_input, ET.Element):
        svg_string = ET.tostring(svg_input, encoding="unicode")

    elif isinstance(svg_input, ET.ElementTree):
        svg_string = ET.tostring(svg_input.getroot(), encoding="unicode")

    elif isinstance(svg_input, str):
        svg_string = svg_input

    else:
        raise TypeError("svg_input must be Element, ElementTree, or SVG string")

    # Render SVG → PNG bytes (in memory)
    png_bytes = cairosvg.svg2png(
        bytestring=svg_string.encode("utf-8"),
        output_width=size[0],
        output_height=size[1],
        background_color="white"
    )

    # Convert PNG bytes → OpenCV image
    png_data = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(png_data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not render SVG.")

    return img
