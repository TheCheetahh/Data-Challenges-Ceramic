import gradio as gr

from database_handler import MongoDBHandler
from web_interface.button_calls.button_clean_svg import click_clean_svg
from web_interface.button_calls.button_next_closest_sample import click_next_closest_sample
from web_interface.button_calls.button_previous_closest_sample import click_previous_closest_sample
from web_interface.button_calls.button_save_sample_type import click_save_sample_type
from web_interface.button_calls.button_analyze_svg import click_analyze_svg
from web_interface.button_calls.button_svg_upload import click_svg_upload

# main webpage code
with gr.Blocks(title="Ceramics Analysis") as demo:
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

            # svg upload
            with gr.Row():
                svg_input = gr.File(label="SVG-Dateien hochladen", file_types=[".svg"], file_count="multiple")
                button_svg_upload = gr.Button("Upload .svg files")

            # csv upload
            with gr.Row():
                csv_input = gr.File(label="CSV-Datei hochladen", file_types=[".csv"])
                csv_upload_button = gr.Button("Upload .csv file")

            # generate clean svg from raw svg in database
            clean_svg_button = gr.Button("🚀 Clean SVG")

            # status box is output for the messages from the buttons of this tab
            with gr.Row():
                status_output_text = gr.Textbox(label="Status", interactive=False)

        # Tab for all analysis related tasks
        with gr.Tab("Analyse files"):
            gr.Markdown("## 🌀 SVG-Krümmungsanalyse\nAnalysiere die SVG Dateien.")

            # settings for analysis
            with gr.Row():
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
                    analyze_button = gr.Button("Analyze SVG")

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
                        next_sample_button = gr.Button("->")

                    closest_svg_output = gr.HTML(
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>")
                    closest_curvature_plot_output = gr.Image(label="Curvature Plot")
                    closest_curvature_color_output = gr.Image(label="Curvature Color Map")
                    closest_angle_plot_output = gr.Image(label="Angle Plot")

    # Button logic:
    # svg upload
    button_svg_upload.click(
        fn=click_svg_upload,
        inputs=[svg_input],
        outputs=[status_output_text, svg_dropdown]
    )

    # csv upload
    csv_upload_button.click(
        fn=db_handler.action_add_csv_data,
        inputs=[csv_input],
        outputs=[status_output_text]
    )

    clean_svg_button.click(
        fn=click_clean_svg,
        inputs=[],
        outputs=[status_output_text]
    )

    analyze_button.click(
        fn=click_analyze_svg,
        inputs=[svg_dropdown, smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
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
                 current_index_state
                 ]
    )

    save_type_button.click(
        fn=click_save_sample_type,
        inputs=[svg_dropdown, sample_type_output],  # using svg_dropdown might cause issues
        outputs=[status_output_text]
    )

    next_sample_button.click(
        fn=click_next_closest_sample,
        inputs=[current_sample_state, closest_list_state, current_index_state, smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[
            closest_svg_output,
            closest_curvature_plot_output,
            closest_curvature_color_output,
            closest_angle_plot_output,
            closest_type_output,
            current_index_state,
            closest_sample_id_output
        ]
    )

    previous_sample_button.click(
        fn=click_previous_closest_sample,
        inputs=[current_sample_state, closest_list_state, current_index_state, smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[
            closest_svg_output,  # svg_html
            closest_curvature_plot_output,  # plot_img
            closest_curvature_color_output,  # color_img
            closest_angle_plot_output,  # angle_plot_img
            closest_type_output,  # typ_text
            current_index_state,  # new_index
            closest_sample_id_output  # label_text
        ]
    )
