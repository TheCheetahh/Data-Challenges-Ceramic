from database_handler import MongoDBHandler


def filter_synonym_matches(sample_id, checked_state):

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    doc = db_handler.collection.find_one({"sample_id": sample_id})

    if checked_state:
        closest_matches = doc.get("full_closest_matches", [])

        db_handler.use_collection("svg_synonym_rules")
        synonym_docs = list(db_handler.collection.find({}))

        name_to_group = {}
        for group in synonym_docs:
            group_names = group.get("members", [])
            group_id = str(group["_id"])
            for name in group_names:
                name_to_group[name] = group_id

        filtered_matches = []
        seen_groups = set()

        for match in closest_matches:  # just check first 5
            if isinstance(match, tuple):
                name = match[0]
            elif isinstance(match, dict):
                name = match.get("id")
            else:
                name = match

            group_id = name_to_group.get(name, name)

            if group_id not in seen_groups:
                filtered_matches.append(match)
                seen_groups.add(group_id)

        db_handler.use_collection("svg_raw")
        db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {"closest_matches": filtered_matches}}
        )

    else:
        filtered_matches = doc.get("full_closest_matches", [])
        db_handler.use_collection("svg_raw")
        db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {"closest_matches": filtered_matches}}
        )

    print(len(filtered_matches))

    return filtered_matches