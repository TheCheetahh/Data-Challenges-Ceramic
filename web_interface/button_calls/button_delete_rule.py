from database_handler import MongoDBHandler
from web_interface.button_calls.button_add_rule import load_rules


def click_delete_rule(delete_input):

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    if delete_input:

        delete_name = delete_input.strip().lower()
        groups = list(db_handler.collection.find({}))

        for group in groups:

            names = [n.strip().lower() for n in group.get("members", [])]
            if delete_name in names:

                updated_names = [n for n in names if n != delete_name]
                if len(updated_names) >= 2:

                    db_handler.collection.update_one(
                        {"_id": group["_id"]},
                        {"$set": {"members": updated_names}}
                    )

                else:

                    db_handler.collection.delete_one(
                        {"_id": group["_id"]}
                    )

    table_rows, ids = load_rules()

    return table_rows, ids, None, "**Selected rule:** none"
