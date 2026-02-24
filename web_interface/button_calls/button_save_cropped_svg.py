from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import crop_svg_path, remove_svg_fill


def click_save_cropped_svg(sample_id, crop_start, crop_end, seam_pos):
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    full_svg, error = db_handler.get_cleaned_svg(sample_id)

    if error or not full_svg:
        return f"⚠ Error retrieving SVG for {sample_id}"

    # Create cropped SVG (already has fill removed in crop_svg_path)
    cropped_svg = crop_svg_path(full_svg, crop_start, crop_end, seam_pos)

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
                "crop_seam_position": float(seam_pos) if seam_pos is not None else 50.0,
                "outdated_curvature": True,
                "icp_data": None
            }
        }
    )

    if result.modified_count > 0:
        return f"✓ Saved cropped SVG for {sample_id}"
    else:
        return f"⚠ No document updated for {sample_id}"
