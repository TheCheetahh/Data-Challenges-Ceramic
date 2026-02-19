import csv
import io
import tempfile
import os
from database_handler import MongoDBHandler


def click_csv_download():
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    docs = list(db_handler.collection.find({}, {"sample_id": 1, "closest_matches": 1, "_id": 0}))
    if not docs:
        return None

    # Find the maximum number of closest matches across all documents
    max_matches = max(len(doc.get("closest_matches", [])) for doc in docs)

    fieldnames = ["sample_id"] + [f"match_{i+1}" for i in range(max_matches)]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", delimiter=";")
    writer.writeheader()
    for doc in docs:
        row = {"sample_id": doc.get("sample_id", "")}
        for i, match in enumerate(doc.get("closest_matches", [])):
            row[f"match_{i+1}"] = match.get("id", "")
        writer.writerow(row)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                      newline="", encoding="utf-8")
    tmp.write(output.getvalue())
    tmp.close()

    return tmp.name


# everything in one cell
"""EXPORT_FIELDS = ["sample_id", "closest_matches"]

def click_csv_download():
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    docs = list(db_handler.collection.find({}, {field: 1 for field in EXPORT_FIELDS} | {"_id": 0}))
    if not docs:
        return None

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS, extrasaction="ignore", delimiter=";")
    writer.writeheader()
    for doc in docs:
        flat = {
            "sample_id": doc.get("sample_id", ""),
            # Extract just the ids from closest_matches list
            "closest_matches": ", ".join(m.get("id", "") for m in doc.get("closest_matches", []))
        }
        writer.writerow(flat)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                      newline="", encoding="utf-8")
    tmp.write(output.getvalue())
    tmp.close()

    return tmp.name"""