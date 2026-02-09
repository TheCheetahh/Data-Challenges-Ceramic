from bson import ObjectId
from web_interface.button_calls.button_add_rule import load_rules
from database_handler import MongoDBHandler


def click_delete_rule(selected_index, ids):
    """Delete a selected rule and return updated table and reset selection."""

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    if selected_index is None or not ids:
        # Nothing to delete, return current table and empty outputs
        table_rows, new_ids = load_rules()
        return table_rows, new_ids, None, "**Selected rule:** none"

    rule_id = ObjectId(ids[selected_index])
    db_handler.collection.delete_one({"_id": rule_id})

    # Reload rules after deletion
    table_rows, new_ids = load_rules()

    selected_label = "**Selected rule:** none"

    # Reset selected_row to None after deleting
    return table_rows, new_ids, None, selected_label
