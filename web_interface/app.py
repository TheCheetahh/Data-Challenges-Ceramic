import gradio as gr

from database_handler import MongoDBHandler
from web_interface.button_calls.button_add_rule import click_add_rule, load_rules
from web_interface.button_calls.button_analyze_svg import click_analyze_svg
from web_interface.button_calls.button_delete_rule import click_delete_rule
from web_interface.button_calls.button_next_closest_sample import click_next_closest_sample
from web_interface.button_calls.button_previous_closest_sample import click_previous_closest_sample
from web_interface.button_calls.button_save_cropped_svg import click_save_cropped_svg
from web_interface.button_calls.button_save_sample_type import click_save_sample_type
from web_interface.button_calls.button_svg_upload import click_svg_upload
from web_interface.button_calls.button_batch_analyze import click_batch_analyze
from web_interface.other_gradio_components.crop_svg import change_crop_svg_dropdown, update_crop_preview

css = """
/* target by elem_id and common class names used by Gradio versions */
#svg_upload .gr-file-list,
#svg_upload .gr-file-preview,
#svg_upload .file-list,
#svg_upload .filePreview,
#svg_upload .file-preview {
    min-height: 200px !important;
    max-height: 200px !important;
    overflow-y: auto !important;
    display: block !important;
}
#centered_md {
    text-align: center;
    width: 100%;
}

"""

# main webpage code
with gr.Blocks(title="Ceramics Analysis", css=css) as demo:
    db_handler = MongoDBHandler("svg_data")

    # states work like a variable
    current_sample_state = gr.State(None)
    closest_list_state = gr.State([])
    current_index_state = gr.State(0)

    with gr.Tabs():
        # Tab for all upload related things
        with gr.Tab("Manage files"):
            gr.Markdown(
                "## 🌀 SVG-Krümmungsanalyse\nLade SVG-Dateien hoch und füge zusätzliche information mittels CSV-Datai hinzu.")

            # uploads
            with gr.Accordion("File Upload", open=True):
                with gr.Row():
                    # svg upload
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("### SVG Uploads för sämples")
                            svg_input = gr.File(label="SVG-Dateien hochladen", file_types=[".svg"],
                                                file_count="multiple", elem_id="svg_upload")
                            button_svg_upload = gr.Button("Upload .svg files")

                    # csv upload
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("### CSV Uploads för sämples")
                            csv_input = gr.File(label="CSV-Datei hochladen", file_types=[".csv"], elem_id="svg_upload")
                            csv_upload_button = gr.Button("Upload .csv file")

                    # svg upload for theory types
                    with gr.Group():
                        with gr.Column():
                            gr.Markdown("### SVG Uploads för theörie tüps fröm le bööks")
                            theory_template_input = gr.File(label="SVG-Datei einer Theorie Vorlage hochladen",
                                                            file_types=[".svg"], file_count="multiple",
                                                            elem_id="svg_upload")
                            theory_template_upload_button = gr.Button("Upload .svg file of a theory template")

            # downloads
            with gr.Accordion("File Download", open=True):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            # csv download
                            gr.Markdown("### CSV Download för sämples täble")
                            csv_download_button = gr.Button("Download Project CSV")

            # generate clean svg from raw svg in database
            # clean_svg_button = gr.Button("🚀 Clean SVG")

            # status box is output for the messages from the buttons of this tab
            with gr.Row():
                status_output_text = gr.Textbox(label="Status", interactive=False, lines=8)

        # Tab for all analysis related tasks
        with gr.Tab("Analyse files"):
            gr.Markdown("## 🌀 SVG-Krümmungsanalyse\nAnalysiere die SVG Dateien.")

            # settings for analysis
            with gr.Row():
                distance_type_dataset = gr.Dropdown(choices=["theory types"], value="theory types",
                                                    label="Distanzberechnung Datensatz")
                distance_value_dataset = gr.Dropdown(choices=["lip_aligned_angle"],
                                                     label="Distanzberechnung Datenpunkte")
                distance_calculation = gr.Dropdown(choices=["Euclidean Distance", "Cosine Similarity",
                                                            "Correlation Distance", "dynamic time warping",
                                                            "integral difference"],
                                                   label="Distanzberechnung Datensatz")
                smooth_method_dropdown = gr.Dropdown(choices=["savgol", "gauss", "bspline", "none"], value="savgol",
                                                     label="Glättungsmethode")
                smooth_factor = gr.Slider(0, 5, value=3, step=0.1, label="Glättungsfaktor")
                smooth_window_slider = gr.Slider(3, 351, value=125, step=10, label="Glättungsfenster")
                samples = gr.Slider(200, 5000, value=1000, step=100, label="Anzahl Abtastpunkte")

            # status box
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False)

            # display analysis content
            with gr.Row():
                # Left column: inspected svg
                with gr.Column(scale=1, min_width=600):
                    svg_dropdown = gr.Dropdown(
                        choices=[str(sid) for sid in db_handler.list_svg_ids()],
                        label="Select SVG to display"
                    )
                    with gr.Row():
                        analyze_button = gr.Button("Analyze SVG")
                        batch_analyse_button = gr.Button("Analyze all Samples")

                    sample_type_output = gr.Textbox(
                        label="Sample Type",
                        interactive=True
                    )

                    save_type_button = gr.Button("Save changes")

                    svg_output = gr.HTML(
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>"
                    )
                    curvature_plot_output = gr.Image(label="Curvature Plot")
                    curvature_color_output = gr.Image(label="Curvature Color Map")
                    angle_plot_output = gr.Image(label="Angle Plot")

                # Right column: closest svg
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("## Closest Match")

                    closest_sample_id_output = gr.Textbox(label="Closest Sample ID", interactive=False)

                    closest_type_output = gr.Textbox(
                        label="Closest Sample Type",
                        interactive=True
                    )

                    with gr.Row():
                        previous_sample_button = gr.Button("<-")
                        index_display = gr.Markdown("-/-", elem_id="centered_md")
                        next_sample_button = gr.Button("->")

                    closest_svg_output = gr.HTML(
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>")
                    closest_curvature_plot_output = gr.Image(label="Curvature Plot")
                    closest_curvature_color_output = gr.Image(label="Curvature Color Map")
                    closest_angle_plot_output = gr.Image(label="Angle Plot")

        with gr.Tab("Edit SVG Path"):
            gr.Markdown("### Crop SVG Path")

            crop_svg_dropdown = gr.Dropdown(
                choices=[str(sid) for sid in db_handler.list_svg_ids()],
                label="Select SVG to display",
                interactive=True
            )

            crop_start = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                value=0.0,
                step=0.01,
                label="Crop start",
                interactive=True
            )

            crop_end = gr.Slider(
                minimum=0.51,
                maximum=1.0,
                value=1.0,
                step=0.01,
                label="Crop end",
                interactive=True
            )

            save_cropped_svg_button = gr.Button("Save cropped svg path")
            save_status = gr.Textbox(label="Save Status", interactive=False)

            with gr.Row():
                full_svg_display = gr.HTML(
                    label="Full SVG"
                )

                cropped_svg_display = gr.HTML(
                    label="Cropped Path Preview"
                )

        with gr.Tab("Synonym Rules"):
            gr.Markdown("### Synonym Rules")

            # Add rule section
            with gr.Row():
                name_input = gr.Textbox(
                    label="Sample name",
                    placeholder="e.g. Drag.33",
                    scale=2
                )
                synonym_input = gr.Textbox(
                    label="Synonym",
                    placeholder="e.g. Nb.9",
                    scale=2
                )
                add_button = gr.Button("Add", scale=1)

            # Table section
            synonym_table = gr.Dataframe(
                headers=["#", "Name", "Synonym", "Created at"],  # added "#"
                datatype=["number", "str", "str", "str"],
                row_count=0,
                col_count=(4, "fixed"),
                label="Existing Rules",
                interactive=False
            )

            # Delete section
            selected_label = gr.Markdown("**Selected rule:** none")
            with gr.Row():
                delete_button = gr.Button("Delete selected rule", variant="stop")
            selected_row = gr.State(None)
            rule_ids = gr.State([])

    # Button logic:
    state_svg_type_sample = gr.State("sample")
    state_svg_type_template = gr.State("template")

    # On load of gradio
    demo.load(
        fn=lambda: load_rules(),
        outputs=[synonym_table, rule_ids]
    )

    # svg upload
    button_svg_upload.click(
        fn=click_svg_upload,
        inputs=[svg_input, state_svg_type_sample],
        outputs=[status_output_text, svg_dropdown]
    )

    # csv upload
    csv_upload_button.click(
        fn=db_handler.action_add_csv_data,
        inputs=[csv_input],
        outputs=[status_output_text]
    )

    theory_template_upload_button.click(
        fn=click_svg_upload,
        inputs=[theory_template_input, state_svg_type_template],
        outputs=[status_output_text]
    )

    analyze_button.click(
        fn=click_analyze_svg,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, svg_dropdown,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[svg_output,
                 curvature_plot_output,
                 curvature_color_output,
                 angle_plot_output,
                 status_output,
                 closest_svg_output,
                 closest_curvature_plot_output,
                 closest_curvature_color_output,
                 closest_angle_plot_output,
                 closest_sample_id_output,
                 sample_type_output,
                 closest_type_output,
                 closest_list_state,
                 current_index_state,
                 index_display
                 ]
    )

    save_type_button.click(
        fn=click_save_sample_type,
        inputs=[svg_dropdown, sample_type_output],  # using svg_dropdown might cause issues
        outputs=[status_output_text]
    )

    next_sample_button.click(
        fn=click_next_closest_sample,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_state,
                closest_list_state, current_index_state,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[
            closest_svg_output,
            closest_curvature_plot_output,
            closest_curvature_color_output,
            closest_angle_plot_output,
            closest_type_output,
            current_index_state,
            closest_sample_id_output,
            index_display
        ]
    )

    previous_sample_button.click(
        fn=click_previous_closest_sample,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_state,
                closest_list_state, current_index_state,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[
            closest_svg_output,  # svg_html
            closest_curvature_plot_output,  # plot_img
            closest_curvature_color_output,  # color_img
            closest_angle_plot_output,  # angle_plot_img
            closest_type_output,  # typ_text
            current_index_state,  # new_index
            closest_sample_id_output,  # label_text
            index_display
        ]
    )

    batch_analyse_button.click(
        fn=click_batch_analyze,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, svg_dropdown,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[svg_output,
                 curvature_plot_output,
                 curvature_color_output,
                 angle_plot_output,
                 status_output,
                 closest_svg_output,
                 closest_curvature_plot_output,
                 closest_curvature_color_output,
                 closest_angle_plot_output,
                 closest_sample_id_output,
                 sample_type_output,
                 closest_type_output,
                 closest_list_state,
                 current_index_state,
                 index_display
                 ]
    )

    # Dropdown change - loads from database and updates sliders
    crop_svg_dropdown.change(
        fn=change_crop_svg_dropdown,
        inputs=[crop_svg_dropdown],
        outputs=[full_svg_display, cropped_svg_display, crop_start, crop_end]
    )

    # Slider change
    crop_start.change(
        fn=update_crop_preview,
        inputs=[crop_svg_dropdown, crop_start, crop_end],
        outputs=[full_svg_display, cropped_svg_display]
    )

    crop_end.change(
        fn=update_crop_preview,
        inputs=[crop_svg_dropdown, crop_start, crop_end],
        outputs=[full_svg_display, cropped_svg_display]
    )

    save_cropped_svg_button.click(
        fn=click_save_cropped_svg,
        inputs=[crop_svg_dropdown, crop_start, crop_end],
        outputs=[save_status]
    )

    add_button.click(
        fn=click_add_rule,
        inputs=[name_input, synonym_input],
        outputs=[synonym_table, rule_ids]
    )

    delete_button.click(
        fn=click_delete_rule,
        inputs=[selected_row, rule_ids],
        outputs=[synonym_table, rule_ids, selected_row, selected_label]
    )

    def select_on_row(evt: gr.SelectData):
        return evt.index[0], f"**Selected rule:** row {evt.index[0] + 1}"

    synonym_table.select(
        fn=select_on_row,
        outputs=[selected_row, selected_label]
    )
