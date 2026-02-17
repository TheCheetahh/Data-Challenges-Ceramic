from database_handler import MongoDBHandler


def update_checkbox_synonym(sample_id, state=True):
    if state:
        return filter_synonym_matches(sample_id)
    else:
        db_handler = MongoDBHandler("svg_data")
        db_handler.use_collection("svg_raw")
        doc = db_handler.collection.find_one({"sample_id": sample_id})

        if not doc:
            return []

        full_matches = doc.get("full_closest_matches", [])

        db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {"closest_matches": full_matches}}
        )

        return full_matches


def filter_synonym_matches(sample_id):

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    doc = db_handler.collection.find_one({"sample_id": sample_id})

    if not doc:
        return []

    closest_matches = doc.get("closest_matches", [])

    if not closest_matches:
        db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {"closest_matches": []}}
        )
        return []

    db_handler.use_collection("svg_synonym_rules")

    synonym_docs = list(db_handler.collection.find({}))

    name_to_group = {}

    for group in synonym_docs:

        group_names = group.get("names", [])

        group_id = str(group["_id"])

        for name in group_names:
            name_to_group[name] = group_id

    filtered_matches = []
    seen_groups = []

    for match in closest_matches:

        if isinstance(match, dict):
            name = match.get("sample_id")
        else:
            name = match

        group_id = name_to_group.get(name, name)

        already_seen = False

        for seen in seen_groups:
            if seen == group_id:
                already_seen = True
                break

        if not already_seen:
            filtered_matches.append(match)
            seen_groups.append(group_id)

    db_handler.use_collection("svg_raw")

    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {"closest_matches": filtered_matches}}
    )

    return filtered_matches
