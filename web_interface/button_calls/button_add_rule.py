from datetime import datetime
from database_handler import MongoDBHandler


def click_add_rule(name, synonym):
    """Insert new rule into MongoDB and return updated table."""

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_synonym_rules")

    if not name or not synonym:
        return load_rules(db_handler)

    rule = {
        "name": name.strip(),
        "synonym": synonym.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    db_handler.insert(rule)

    return load_rules(db_handler)


def load_rules(db_handler):
    """Load all rules from MongoDB into table format."""
    rules = db_handler.find_all()
    return [
        [r["name"], r["synonym"], r["created_at"]]
        for r in rules
    ]
