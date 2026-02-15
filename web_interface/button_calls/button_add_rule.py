from datetime import datetime
from database_handler import MongoDBHandler


def click_add_rule(name, synonym):

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    if not name or not synonym:
        return load_rules()

    name = name.strip()
    synonym = synonym.strip()

    rules = list(db_handler.find())

    group_a = None
    group_b = None

    # Find existing groups
    for rule in rules:
        members = rule.get("members", [])

        if name in members:
            group_a = rule

        if synonym in members:
            group_b = rule

    # Case 1: neither exists → create new group
    if not group_a and not group_b:

        db_handler.insert({
            "members": [name, synonym],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Case 2: name exists → add synonym
    elif group_a and not group_b:

        updated = list(set(group_a["members"] + [synonym]))

        db_handler.collection.update_one(
            {"_id": group_a["_id"]},
            {"$set": {"members": updated}}
        )

    # Case 3: synonym exists → add name
    elif group_b and not group_a:

        updated = list(set(group_b["members"] + [name]))

        db_handler.collection.update_one(
            {"_id": group_b["_id"]},
            {"$set": {"members": updated}}
        )

    # Case 4: both exist in different groups → merge
    elif group_a["_id"] != group_b["_id"]:

        merged = list(set(group_a["members"] + group_b["members"]))

        db_handler.collection.update_one(
            {"_id": group_a["_id"]},
            {"$set": {"members": merged}}
        )

        db_handler.collection.delete_one({"_id": group_b["_id"]})

    return load_rules()


def load_rules():

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    rules = list(db_handler.find())

    table_rows = []
    ids = []

    for i, rule in enumerate(rules, start=1):

        members = sorted(rule.get("members", []))

        table_rows.append([
            i,
            ", ".join(members),
            len(members),
            rule.get("created_at", "")
        ])

        ids.append(str(rule["_id"]))

    return table_rows, ids
