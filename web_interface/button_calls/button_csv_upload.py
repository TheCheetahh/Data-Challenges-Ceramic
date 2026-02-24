import chardet
import pandas as pd

from database_handler import MongoDBHandler


def click_csv_upload(csv_file):
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    if csv_file is None:
        return "⚠️ No CSV file provided."

    # Try encodings until umlauts render correctly
    csv_df = None
    for enc in ["utf-8", "utf-8-sig", "cp850", "cp1252", "latin-1", "mac_roman", "iso-8859-1"]:
        try:
            df = pd.read_csv(csv_file.name, sep=';', encoding=enc)
            sample = df["Sample.Form"].dropna().astype(str).str.cat()
            if 'Å' not in sample and '\ufffd' not in sample and '\x81' not in sample:
                csv_df = df
                break
        except Exception:
            continue

    if csv_df is None:
        return "Error: Could not determine correct file encoding."

    # Merge duplicate Sample.Id rows
    csv_df_grouped = csv_df.groupby("Sample.Id").agg(lambda x: '|'.join(map(str, x.dropna()))).reset_index()

    # Build lookup dict, strip whitespace from keys
    csv_lookup = {
        str(k).strip(): v
        for k, v in csv_df_grouped.set_index("Sample.Id").to_dict(orient="index").items()
    }

    # Map CSV columns to DB field names (strip "Sample." prefix)
    field_mapping = {
        "Sample.Warenart": "Warenart",
        "Sample.Form": "Form",
        "Sample.Typ": "Typ",
        "Sample.Randerhaltung": "Randerhaltung"
    }

    updated_count = 0
    skipped_count = 0

    for doc in db_handler.collection.find():
        sample_id = str(doc.get("sample_id", "")).strip()
        if not sample_id or sample_id not in csv_lookup:
            skipped_count += 1
            continue

        info = csv_lookup[sample_id]
        mapped_info = {db_key: info[csv_key] for csv_key, db_key in field_mapping.items() if csv_key in info}

        db_handler.collection.update_one(
            {"_id": doc["_id"]},
            {"$set": mapped_info}
        )
        updated_count += 1

    return f"CSV data added: {updated_count} documents updated, {skipped_count} were not found."