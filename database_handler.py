from pymongo import MongoClient
from datetime import datetime
import os


class MongoDBHandler:
    """MongoDB handler class to manage database connections and collections."""

    def __init__(self, db_name, uri="mongodb://localhost:27017/"): # mongodb://localhost:27017/ is standard installation path
        """initialise the MongoDB handler object."""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = None  # will set dynamically


    def use_collection(self, collection_name):
        """set or use collection."""
        self.collection = self.db[collection_name]


    def count(self, filter_query=None):
        """Count documents in the current collection."""
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.count_documents(filter_query or {})


    def insert(self, document):
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.insert_one(document)


    def insert_svg_files(self, files):
        """
        Insert multiple SVG files into the collection 'svg_raw'.

        Parameters:
        - files (list of file objects)

        Returns:
        - status_messages (list of str): one message per file
        """
        if self.collection is None:
            self.use_collection("svg_raw")

        duplicate_counter = 0

        messages = []
        for svg_file in files:
            try:
                with open(svg_file.name, "r", encoding="utf-8") as f:
                    content = f.read()

                    # Extract just the file name (without path)
                    filename_only = os.path.basename(svg_file.name)
                    # Check if the file is already in the database
                    if self.collection.find_one({"filename": filename_only}):
                        duplicate_counter += 1
                        continue

                doc = {
                    "filename": filename_only,
                    "content": content,
                    "uploaded_at": datetime.utcnow()
                }
                self.insert(doc)
                messages.append(f"Uploaded '{svg_file.name}' successfully.")

            except Exception as e:
                messages.append(f"Error '{svg_file.name}': {e}")

        # include total count
        messages.append(str(duplicate_counter) + " duplicate files were not added to collection")
        messages.append(f"Total files in collection: {self.count()}")
        return "\n".join(messages)


    def find(self, filter_query=None):
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.find(filter_query or {})


    def close(self):
        self.client.close()