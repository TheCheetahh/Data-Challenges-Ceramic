from database_handler import MongoDBHandler
import gradio as gr


def click_svg_upload(svg_input):
    """
    called by button
    calls database to store the svg files

    :param svg_input:
    :return: message and new dropdown content
    """

    # get db_handler
    db_handler = MongoDBHandler("svg_data")

    # insert the svg files into database
    message = db_handler.insert_svg_files(svg_input)

    # Return both status message and dropdown update
    svg_id_list = db_handler.list_svg_ids()
    dropdown_update = gr.update(choices=[str(sid) for sid in svg_id_list])

    return message, dropdown_update
