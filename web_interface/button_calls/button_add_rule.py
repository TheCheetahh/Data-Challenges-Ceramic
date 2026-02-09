from datetime import datetime
from database_handler import MongoDBHandler


def click_add_rule(name, synonym):
    """Insert new rule into MongoDB and return updated table."""

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    if not name or not synonym:
        return load_rules()

    rule = {
        "name": name.strip(),
        "synonym": synonym.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    db_handler.insert(rule)

    return load_rules()


def load_rules():
    """Load all rules from MongoDB into table format, safely handles empty DB."""

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    rules = list(db_handler.find())  # returns [] if no documents

    table_rows = []
    ids = []

    for i, r in enumerate(rules, start=1):
        table_rows.append([
            i,  # row index
            r.get("name", ""),
            r.get("synonym", ""),
            r.get("created_at", "")
        ])
        ids.append(str(r["_id"]))

    # If empty, return empty list for table and empty list for IDs
    if not table_rows:
        table_rows = []
        ids = []

    return table_rows, ids
