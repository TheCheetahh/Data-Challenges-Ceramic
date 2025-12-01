from analysis.clean_SVG import clean_all_svgs
from database_handler import MongoDBHandler
import gradio as gr


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
        svg_id_list = db_handler.list_svg_ids()
        dropdown_update = gr.update(choices=[str(sid) for sid in svg_id_list])

        message.append(clean_all_svgs(db_handler, svg_file_type))

        return "\n".join(message), dropdown_update
    else:

        message.append(clean_all_svgs(db_handler, svg_file_type))

        return message
