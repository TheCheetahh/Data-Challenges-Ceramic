from pymongo import MongoClient
from datetime import datetime
import os
import pandas as pd
import chardet


class MongoDBHandler:
    """MongoDB handler class to manage database connections and collections."""

    def __init__(self, db_name, uri="mongodb://localhost:27017/"):
        """initialise the MongoDB handler object.
        mongodb://localhost:27017/ is standard path"""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = None  # will be set when needed


    def use_collection(self, collection_name):
        """set or use collection."""
        self.collection = self.db[collection_name]


    def count(self, filter_query=None):
        """Count documents in the current collection."""
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.count_documents(filter_query or {})


    def insert(self, document):
        """insert document into current collection."""
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.insert_one(document)


    def insert_svg_files(self, files):
        """Insert multiple SVG files into the collection svg_raw."""
        messages = []  # return string list

        # if no files were uploaded
        if not files:
            messages.append("no files to upload")
            return "\n".join(messages)

        # set collection
        if self.collection is None:
            self.use_collection("svg_raw")

        # duplicate entries will not be inserted but counted
        duplicate_counter = 0

        # upload each file
        for svg_file in files:
            try:
                # open file
                with open(svg_file.name, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Extract just the file name (without path) and the sample_id (svg have unnecessary recons in name)
                    filename_only = os.path.basename(svg_file.name)
                    sample_id = int(filename_only.split("_")[1].split(".")[0])

                    # Check if the file is already in the database
                    if self.collection.find_one({"filename": filename_only}):
                        duplicate_counter += 1
                        continue

                # build doc and insert
                doc = {
                    "sample_id": sample_id,
                    "filename": filename_only,
                    "raw_content": content,
                    "uploaded_at": datetime.utcnow()
                }
                self.insert(doc)
                messages.append(f"Uploaded '{svg_file.name}' successfully.")

            except Exception as e:
                messages.append(f"Error '{svg_file.name}': {e}")

        # return messages
        messages.append(str(duplicate_counter) + " duplicate files were not added to collection")
        messages.append(f"Total files in collection: {self.count()}")
        return "\n".join(messages)


    def action_add_csv_data(self, csv_file):
        """Add CSV data to the SVG documents in the database."""
        if csv_file is None:
            return "⚠️ No CSV file provided."

        # set collection
        if self.collection is None:
            self.use_collection("svg_raw")

        # Read file as bytes. This gets encoding of the csv
        rawdata = open(csv_file.name, "rb").read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']

        # Load CSV into a DataFrame
        try:
            csv_df = pd.read_csv(csv_file.name, sep=';', encoding=encoding)
            # print(csv_df.columns) # debug
        except Exception as e:
            return f"Error reading CSV: {e}"

        # there are duplicate entries in the Excel/csv for the same sample_id. Need to figure this out and make git
        # issue
        duplicates = csv_df["Sample.Id"][csv_df["Sample.Id"].duplicated()]
        if not duplicates.empty:
            print(f"⚠️ Warning: duplicate Sample.Id values found: {duplicates.tolist()}")
        # csv_df_grouped = csv_df.groupby("Sample.Id").first().reset_index() # group rejects all lines except the first
        # merge all entries
        csv_df_grouped = csv_df.groupby("Sample.Id").agg(lambda x: '|'.join(map(str, x.dropna()))).reset_index()
        # Convert CSV to a dictionary for fast lookup: {Id: row_dict}
        csv_lookup = csv_df_grouped.set_index("Sample.Id").to_dict(orient="index")

        # Iterate over documents in the collection
        updated_count = 0
        skipped_count = 0  # svgs that did not get new data from the csv
        for doc in self.find():
            sample_id = doc.get("sample_id")
            if not sample_id:
                skipped_count += 1
                continue

            # if the sample_id of a svg in the database has a corresponding sample_id in the csv
            if sample_id in csv_lookup:
                info = csv_lookup[sample_id]
                # Update the document in MongoDB with CSV fields
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "Warenart": info.get("Sample.Warenart"),
                        "Form": info.get("Sample.Form"),
                        "Typ": info.get("Sample.Typ"),
                        "Randerhaltung": info.get("Sample.Randerhaltung")
                    }}
                )
                updated_count += 1
            else:
                skipped_count += 1

        return f"CSV data added: {updated_count} documents updated, {skipped_count} were not found."


    def find(self, filter_query=None):
        """find all documents matching the provided filter."""
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.find(filter_query or {})


    def close(self):
        """close the connection to the database."""
        self.client.close()


    def get_cleaned_svg(self, sample_id):
        """gets a cleaned SVG from the database."""
        self.use_collection("svg_raw")

        try:
            sample_id = int(sample_id)
        except ValueError:
            return None, "❌ sample_id must be a number."

        doc = self.collection.find_one({"sample_id": sample_id})
        if not doc:
            return None, f"❌ No entry found for sample_id {sample_id}."

        cleaned_svg = doc.get("cleaned_svg")
        if not cleaned_svg:
            return None, f"⚠️ No cleaned SVG stored for sample_id {sample_id}."

        return cleaned_svg, None


    def list_svg_ids(self):
        """Return a sorted list of sample_ids for all SVGs in 'svg_raw'."""
        self.use_collection("svg_raw")
        docs = self.collection.find({}, {"sample_id": 1})
        sample_ids = [doc["sample_id"] for doc in docs if "sample_id" in doc]
        return sorted(sample_ids)


    def store_curvature_in_db(self, sample_id, arc_lengths, curvature, smooth_method, smooth_factor, smooth_window,
                              n_samples):
        """store the 1D curvature values"""
        self.use_collection("svg_raw")
        self.collection.update_one(
            {"sample_id": int(sample_id)},
            {"$set": {
                "curvature_data": {
                    "arc_lengths": arc_lengths.tolist(),
                    "curvature": curvature.tolist(),
                    "smoothing_settings": {
                        "smooth_method": smooth_method,
                        "smooth_factor": smooth_factor,
                        "smooth_window": smooth_window,
                        "n_samples": n_samples
                    }
                }
            }},
            upsert=False
        )

    def get_sample_type(self, sample_id):
        """get sample type from database"""
        db = MongoDBHandler("svg_data")
        db.use_collection("svg_raw")

        doc = db.collection.find_one({"sample_id": int(sample_id)}, {"Typ": 1})
        if not doc:
            return None
        return doc.get("Typ", None)

    def update_type(self, sample_id, new_type):
        """Update the 'type' field of a document"""
        try:
            sample_id = int(sample_id)
        except (ValueError, TypeError):
            return False, "❌ sample_id must be an integer."

        if not isinstance(new_type, str):
            return False, "❌ new_type must be a string."

        result = self.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {"Typ": new_type}}
        )

        if result.matched_count == 0:
            return False, f"❌ No document found with sample_id {sample_id}"

        return True, f"✅ Updated type for sample_id {sample_id} to '{new_type}'"

    def save_closest_matches(self, sample_id, matches):
        """
        Save [{'id': X, 'distance': Y}, ...] for sample_id.
        """
        self.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {"closest_matches": matches}},
            upsert=True
        )

    def get_closest_matches(self, sample_id):
        doc = self.collection.find_one(
            {"sample_id": sample_id},
            {"closest_matches": 1}
        )
        return doc.get("closest_matches", []) if doc else []
