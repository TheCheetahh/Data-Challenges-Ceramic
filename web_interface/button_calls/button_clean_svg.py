from database_handler import MongoDBHandler
from analysis.clean_SVG import clean_all_svgs


def click_clean_svg():
    """clean svg. This function currently only cleans the svg"""

    # get a database handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    message = clean_all_svgs(db_handler)

    return message
