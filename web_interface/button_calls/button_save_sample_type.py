from database_handler import MongoDBHandler


def click_save_sample_type(sample_id, new_type):
    db = MongoDBHandler("svg_data")
    db.use_collection("svg_raw")

    # Validate ID
    try:
        sample_id = sample_id
    except (ValueError, TypeError):
        return "❌ Invalid sample ID."

    # Update type field
    success, msg = db.update_type(sample_id, new_type)
    return msg
