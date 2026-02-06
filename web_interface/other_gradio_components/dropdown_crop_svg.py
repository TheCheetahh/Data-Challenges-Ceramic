from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display, remove_svg_fill, crop_svg_path


def change_crop_svg_dropdown(sample_id, crop_start, crop_end):
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    full_svg, error = db_handler.get_cleaned_svg(sample_id)

    # Remove fill from full SVG
    full_svg_no_fill = remove_svg_fill(full_svg)
    display_full_svg = format_svg_for_display(full_svg_no_fill)

    # Create cropped SVG (already has fill removed in crop_svg_path)
    cropped_svg = crop_svg_path(full_svg, crop_start, crop_end)
    display_cropped_svg = format_svg_for_display(cropped_svg)

    return display_full_svg, display_cropped_svg




