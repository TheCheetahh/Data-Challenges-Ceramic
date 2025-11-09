from pymongo import MongoClient
from datetime import datetime

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

    def insert_svg_file(self, file_path):
        """
        Read an SVG file and insert it into the collection 'svg_files'.
        Returns a status string.
        """
        self.use_collection("svg_raw")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = {
                "filename": file_path.name,
                "content": content,
                "uploaded_at": datetime.utcnow()
            }
            self.insert(doc)
            count = self.count()
            return f"SVG '{file_path.name}' uploaded successfully! Total files in database: {count}"

        except Exception as e:
            return f"Error uploading SVG: {e}"

    def find(self, filter_query=None):
        if self.collection is None:
            raise ValueError("No collection selected.")
        return self.collection.find(filter_query or {})

    def close(self):
        self.client.close()