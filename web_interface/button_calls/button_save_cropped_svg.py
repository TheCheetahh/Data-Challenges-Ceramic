from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import crop_svg_path, remove_svg_fill


def click_save_cropped_svg(sample_id, crop_start, crop_end):
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    full_svg, error = db_handler.get_cleaned_svg(sample_id)

    # Check if SVG was retrieved successfully
    if error or not full_svg:
        # print(f"Error retrieving SVG for sample_id {sample_id}: {error}")
        return f"⚠ Error retrieving SVG for {sample_id}"

    # Create cropped SVG (already has fill removed in crop_svg_path)
    cropped_svg = crop_svg_path(full_svg, crop_start, crop_end)

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
                "outdated_curvature": True
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

