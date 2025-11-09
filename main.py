from database_handler import MongoDBHandler
from app import demo


def main():
    """main function of project. Run this file to start the program"""

    # connect to database
    db = MongoDBHandler("svg_data")
    db.use_collection("svg_raw")
    print("svg_raw count:", db.count())

    # TODO startup database check.

    # launch web ui (http://127.0.0.1:7860/)
    demo.launch()

    db.close()


if __name__ == "__main__":
    main()