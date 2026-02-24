from analysis.clean_SVG import clean_all_svgs
from database_handler import MongoDBHandler
import gradio as gr

from web_interface.other_gradio_components.dropdown import update_sample_id_dropdown


def click_svg_upload(svg_input, svg_file_type):
    """
    called by button
    calls database to store the svg files

    :param svg_input:
    :return: message and new dropdown content
    """

    # get db_handler
    db_handler = MongoDBHandler("svg_data")

    message = [db_handler.insert_svg_files(svg_input, svg_file_type)]
    # insert the svg files into database

    if svg_file_type == "sample":
        # Return both status message and dropdown update

        message.append(clean_all_svgs(db_handler, svg_file_type))

        dropdown_update = update_sample_id_dropdown()
        return "\n".join(message), dropdown_update

    else:
        message.append(clean_all_svgs(db_handler, svg_file_type))

        # set all prev method to None for recalculation
        db_handler.use_collection("svg_raw")
        db_handler.collection.update_many(
            {},
            {
                "$set": {
                    "last_distance_method": None,
                    "last_laa_config": None,
                    "closest_matches_valid": False,
                    "outdated_curvature": True
                }
            }
        )

        return "\n".join(message)
