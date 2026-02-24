from database_handler import MongoDBHandler
import gradio as gr


def update_sample_id_dropdown():
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    sample_id_list = db_handler.collection.distinct("sample_id")
    dropdown_update = gr.update(choices=[str(sid) for sid in sample_id_list])

    return dropdown_update
