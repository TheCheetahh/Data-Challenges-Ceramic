from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display, remove_svg_fill, crop_svg_path


def change_crop_svg_dropdown(sample_id):
    """Called when dropdown changes - loads saved settings from database"""
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Get the document from database
    document = db_handler.collection.find_one({"sample_id": sample_id})

    if not document:
        print(f"No document found for sample_id {sample_id}")
        return "", "", 0.0, 1.0

    full_svg = document.get("cleaned_svg") or document.get("svg")

    # Load saved crop settings if they exist, otherwise use defaults
    saved_crop_start = document.get("crop_start")
    saved_crop_end = document.get("crop_end")

    # Use saved values if available, otherwise use default values (0.0 and 1.0)
    crop_start = saved_crop_start if saved_crop_start is not None else 0.0
    crop_end = saved_crop_end if saved_crop_end is not None else 1.0

    print(f"Loaded crop settings: start={crop_start}, end={crop_end}")

    # Remove fill from full SVG
    full_svg_no_fill = remove_svg_fill(full_svg)
    display_full_svg = format_svg_for_display(full_svg_no_fill)

    # Create cropped SVG (already has fill removed in crop_svg_path)
    cropped_svg = crop_svg_path(full_svg, crop_start, crop_end)
    display_cropped_svg = format_svg_for_display(cropped_svg)

    return display_full_svg, display_cropped_svg, crop_start, crop_end


def update_crop_preview(sample_id, crop_start, crop_end):
    """Called when sliders change - uses current slider values"""
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Get the document from database
    document = db_handler.collection.find_one({"sample_id": sample_id})

    if not document:
        print(f"No document found for sample_id {sample_id}")
        return "", ""

    full_svg = document.get("cleaned_svg") or document.get("svg")

    # Remove fill from full SVG
    full_svg_no_fill = remove_svg_fill(full_svg)
    display_full_svg = format_svg_for_display(full_svg_no_fill)

    # Create cropped SVG with current slider values
    cropped_svg = crop_svg_path(full_svg, crop_start, crop_end)
    display_cropped_svg = format_svg_for_display(cropped_svg)

    return display_full_svg, display_cropped_svg