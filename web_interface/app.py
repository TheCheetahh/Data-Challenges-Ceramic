import gradio as gr

from database_handler import MongoDBHandler
from web_interface.button_calls.button_add_rule import click_add_rule, load_rules
from web_interface.button_calls.button_analyze_svg import click_analyze_svg, update_analyze_button_color
from web_interface.button_calls.button_csv_download import click_csv_download
from web_interface.button_calls.button_csv_upload import click_csv_upload
from web_interface.button_calls.button_navigate_closest_sample import (
    click_navigate_closest_sample,
    click_select_closest_sample,
    update_closest_match_dropdown,
)
from web_interface.button_calls.button_delete_rule import click_delete_rule
from web_interface.button_calls.button_pin import click_pin_button
from web_interface.button_calls.button_save_cropped_svg import click_save_cropped_svg
from web_interface.button_calls.button_auto_crop_faulty_svgs import click_auto_crop_faulty_svgs
from web_interface.button_calls.button_save_sample_type import click_save_sample_type
from web_interface.button_calls.button_svg_upload import click_svg_upload
from web_interface.button_calls.button_batch_analyze import click_batch_analyze
from web_interface.other_gradio_components.crop_svg import change_crop_svg_dropdown, update_crop_preview
from web_interface.other_gradio_components.dropdown import update_sample_id_dropdown

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
    next_index_one = gr.State(1)
    prev_index_one = gr.State(-1)
    # IMPORTANT: We use Dropdown.input (user-only) instead of Dropdown.change.
    # This avoids double-loading when the dropdown value is updated programmatically.

    # Needed for svg uploads
    state_svg_type_sample = gr.State("sample")
    state_svg_type_template = gr.State("template")

    # Remember the last state
    last_analysis_state = gr.State(None)


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
                            csv_file_output = gr.File(label="Download", visible=False)

            # generate clean svg from raw svg in database
            # clean_svg_button = gr.Button("🚀 Clean SVG")

            # status box is output for the messages from the buttons of this tab
            with gr.Row():
                status_output_text = gr.Textbox(label="Status", interactive=False, lines=8)

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

            seam_pos = gr.Slider(
                minimum=0.0,
                maximum=100.0,
                value=50.0,
                step=0.5,
                label="Start/End Position (50 = lowest point)",
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


            with gr.Accordion("Batch auto-crop saved SVGs", open=False):
                
                with gr.Row():
                    auto_crop_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        value=75.0,
                        step=0.5,
                        label="Fault threshold (min start/end y%)",
                        interactive=True,
                    )
                    auto_crop_overwrite = gr.Checkbox(
                        label="Overwrite existing cropped_svg",
                        value=False,
                    )
                    auto_crop_dry_run = gr.Checkbox(
                        label="Dry run (no DB write)",
                        value=True,
                    )

                with gr.Row():
                    auto_crop_crop_start = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.05,
                        step=0.01,
                        label="Batch crop start",
                        interactive=True,
                    )
                    auto_crop_crop_end = gr.Slider(
                        minimum=0.51,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        label="Batch crop end",
                        interactive=True,
                    )

                with gr.Row():
                    auto_crop_seam_pos = gr.Slider(
                        minimum=0.0,
                        maximum=100.0,
                        value=50.0,
                        step=0.5,
                        label="Batch seam position",
                        interactive=True,
                    )
                    auto_crop_button = gr.Button("Auto-crop faulty SVGs")

                auto_crop_status = gr.Textbox(
                    label="Auto-crop status",
                    interactive=False,
                    lines=8
                )

        # Tab for all analysis related tasks
        with gr.Tab("Analyse files"):
            gr.Markdown("## 🌀 Sample Analysis")

            # settings for analysis
            with gr.Row():
                with gr.Accordion("Settings", open=False):
                    distance_type_dataset = gr.State("theory types")
                    """distance_type_dataset = gr.Dropdown(choices=["theory types"], value="theory types",
                                                        label="Distanzberechnung Datensatz")"""
                    distance_value_dataset = gr.Dropdown(choices=["ICP", "lip_aligned_angle", "Orb", "DISK"],
                                                         label="Calculation Algorithm")
                    distance_calculation = gr.Dropdown(choices=["Euclidean Distance", "Cosine Similarity",
                                                                "Correlation Distance",
                                                                "integral difference"],
                                                       label="Distance Metric (currently not implemented)")
                    # "gauss", "bspline", "none"
                    smooth_method_dropdown = gr.Dropdown(choices=["savgol"], value="savgol",
                                                         label="Smoothing Method")
                    smooth_factor = gr.Slider(0, 5, value=3, step=0.1, label="Smoothing Factor (default: 3)")
                    smooth_window_slider = gr.Slider(3, 351, value=125, step=10, label="Smoothing Window (default: 125)")
                    samples = gr.Slider(200, 5000, value=800, step=100, label="Number of Datapoints (default: 800) HIGH VALUES INCREASE CALCULATION TIME BY A LOT")
                    duplicate_synonym_checkbox = gr.Checkbox(label=" Show only first result of all synonym groups",
                                                             value=False)

            # status box
            with gr.Row():
                status_output = gr.Textbox(label="Calculation Status", interactive=False)

            # display analysis content
            with gr.Row():

                # Left column: inspected svg
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("## Current Sample")

                    svg_dropdown = gr.Dropdown(
                        choices=[str(sid) for sid in db_handler.list_svg_ids()],
                        label="Select SVG to analyze (after a change in selection you must press analyze svg)"
                    )

                    sample_type_output = gr.Textbox(
                        label="Sample Type",
                        interactive=True
                    )

                    with gr.Row():
                        batch_analyse_button = gr.Button("Analyze all Samples", variant="stop")
                        save_type_button = gr.Button("Save sample type")
                        analyze_button = gr.Button("Analyze SVG", variant="primary")

                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("## Pinned Match")

                    # displays type and distance
                    pinned_sample_id_output = gr.Textbox(label="Pinned Sample ID", interactive=False)

                    # change this to synonym group
                    pinned_synonyme_output = gr.Textbox(
                        label="Synonymes",
                        interactive=True
                    )

                    with gr.Row():
                        pinned_index_display = gr.Markdown("-", elem_id="centered_md")
                        pin_button = gr.Button("Pin Match")

                # Right column: closest svg
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("## Closest Match")

                    # Replaces the old "Closest Template ID" textbox
                    closest_match_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Closest Template ID",
                        interactive=True,
                    )

                    # Keep hidden output to stay compatible with existing analyze/batch callbacks
                    closest_sample_id_output = gr.Textbox(
                        label="Closest Template ID (hidden)",
                        interactive=False,
                        visible=False,
                    )

                    # change this to synonym group
                    closest_template_synonymes = gr.Textbox(
                        label="Synonymes",
                        interactive=True
                    )

                    with gr.Row():
                        previous_sample_button = gr.Button("←")
                        index_display = gr.Markdown("-/-", elem_id="centered_md")
                        next_sample_button = gr.Button("→")

            with gr.Row():
                # Left column: inspected svg
                with gr.Column(scale=1, min_width=400):
                    svg_output = gr.HTML(
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>"
                    )
                with gr.Column(scale=1, min_width=400):
                    pinned_svg_output = gr.HTML(
                        visible=True,
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;overflow:hidden;'>SVG will appear here</div>"
                    )

                    pinned_icp_output = gr.Image(
                        label="ICP Overlap",
                        visible=False
                    )
                with gr.Column(scale=1, min_width=400):
                    closest_svg_output = gr.HTML(
                        visible=False,
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;overflow:hidden;'>SVG will appear here</div>"
                    )

                    closest_icp_output = gr.Image(
                        label="ICP Overlap", interactive=False,
                        visible=True
                    )

            with gr.Accordion("More Graphs", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=400):
                        curvature_plot_output = gr.Image(label="Curvature Plot")
                        angle_plot_output = gr.Image(label="Angle Plot")
                        curvature_color_output = gr.Image(label="Curvature Color Map")

                    with gr.Column(scale=1, min_width=400):
                        pinned_curvature_plot_output = gr.Image(label="Curvature Plot")
                        pinned_angle_plot_output = gr.Image(label="Angle Plot")
                        pinned_curvature_color_output = gr.Image(label="Curvature Color Map")

                    # right column
                    with gr.Column(scale=1, min_width=400):
                        closest_curvature_plot_output = gr.Image(label="Curvature Plot", interactive=False)
                        closest_angle_plot_output = gr.Image(label="Angle Plot", interactive=False)
                        closest_curvature_color_output = gr.Image(label="Curvature Color Map", interactive=False)

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
                delete_input = gr.Textbox(
                    label="Synonym to delete from group",
                    placeholder="e.g. Nb.9",
                    scale=2
                )
                delete_button = gr.Button("Delete selected rule", variant="stop")
            selected_row = gr.State(None)
            rule_ids = gr.State([])


    # On load of gradio
    demo.load(
        fn=lambda: load_rules(),
        outputs=[synonym_table, rule_ids]
    )

    demo.load(
        fn=update_sample_id_dropdown,
        outputs=[svg_dropdown]
    )

    demo.load(
        fn=update_sample_id_dropdown,
        outputs=[crop_svg_dropdown]
    )

    demo.load(
        fn=change_crop_svg_dropdown,
        inputs=[crop_svg_dropdown],
        outputs=[full_svg_display, cropped_svg_display, crop_start, crop_end, seam_pos]
    )

    # svg upload
    button_svg_upload.click(
        fn=click_svg_upload,
        inputs=[svg_input, state_svg_type_sample],
        outputs=[status_output_text, svg_dropdown]
    ).then(
        fn=update_sample_id_dropdown,
        inputs=[],
        outputs=[crop_svg_dropdown]
    )

    # csv upload
    csv_upload_button.click(
        fn=click_csv_upload,
        inputs=[csv_input],
        outputs=[status_output_text]
    )

    theory_template_upload_button.click(
        fn=click_svg_upload,
        inputs=[theory_template_input, state_svg_type_template],
        outputs=[status_output_text]
    )

    analysis_button_inputs = [
        current_sample_state,
        svg_dropdown,
        distance_value_dataset,
        distance_calculation,
        smooth_method_dropdown,
        smooth_factor,
        smooth_window_slider,
        samples,
        last_analysis_state,
    ]

    analysis_controls = [
        svg_dropdown,
        distance_value_dataset,
        distance_calculation,
        smooth_method_dropdown,
        smooth_factor,
        smooth_window_slider,
        samples,
    ]

    analyze_button.click(
        fn=click_analyze_svg,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, svg_dropdown,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples, duplicate_synonym_checkbox],
        outputs=[svg_output,
                 curvature_plot_output,
                 curvature_color_output,
                 angle_plot_output,
                 status_output,
                 closest_svg_output,
                 closest_icp_output,
                 closest_curvature_plot_output,
                 closest_curvature_color_output,
                 closest_angle_plot_output,
                 closest_sample_id_output,
                 sample_type_output,
                 closest_template_synonymes,
                 closest_list_state,
                 current_index_state,
                 index_display,
                 current_sample_state,
                 last_analysis_state,
                 ]
    ).then(
        fn=update_closest_match_dropdown,
        inputs=[closest_list_state, current_index_state],
        outputs=[closest_match_dropdown],
    ).then(
        fn=click_pin_button,
        inputs=[distance_value_dataset,
                closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_match_dropdown,
                closest_template_synonymes,
                current_index_state],
        outputs=[
            pinned_svg_output,
            pinned_icp_output,
            pinned_curvature_plot_output,
            pinned_curvature_color_output,
            pinned_angle_plot_output,
            pinned_sample_id_output,
            pinned_synonyme_output,
            pinned_index_display
        ]
    ).then(
        fn=update_analyze_button_color,
        inputs=analysis_button_inputs,
        outputs=[analyze_button]
    )

    save_type_button.click(
        fn=click_save_sample_type,
        inputs=[current_sample_state, sample_type_output],  # using svg_dropdown might cause issues
        outputs=[status_output_text]
    )

    next_sample_button.click(
        fn=click_navigate_closest_sample,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_state, closest_list_state, current_index_state,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples, next_index_one],
        outputs=[
            closest_svg_output,        # gr.update (SVG visible/hidden)
            closest_icp_output,        # gr.update (ICP image visible/hidden)

            closest_curvature_plot_output,
            closest_curvature_color_output,
            closest_angle_plot_output,

            closest_template_synonymes,
            current_index_state,
            closest_sample_id_output,
            index_display
        ]
    ).then(
        fn=update_closest_match_dropdown,
        inputs=[closest_list_state, current_index_state],
        outputs=[closest_match_dropdown],
    )

    previous_sample_button.click(
        fn=click_navigate_closest_sample,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_state, closest_list_state, current_index_state,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples, prev_index_one],
        outputs=[
            closest_svg_output,  # gr.update (SVG visible/hidden)
            closest_icp_output,  # gr.update (ICP image visible/hidden)

            closest_curvature_plot_output,
            closest_curvature_color_output,
            closest_angle_plot_output,

            closest_template_synonymes,
            current_index_state,
            closest_sample_id_output,
            index_display
        ]
    ).then(
        fn=update_closest_match_dropdown,
        inputs=[closest_list_state, current_index_state],
        outputs=[closest_match_dropdown],
    )

    batch_analyse_button.click(
        fn=click_batch_analyze,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, svg_dropdown,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples, duplicate_synonym_checkbox],
        outputs=[svg_output,
                 curvature_plot_output,
                 curvature_color_output,
                 angle_plot_output,
                 status_output,
                 closest_svg_output,
                 closest_icp_output,
                 closest_curvature_plot_output,
                 closest_curvature_color_output,
                 closest_angle_plot_output,
                 closest_sample_id_output,
                 sample_type_output,
                 closest_template_synonymes,
                 closest_list_state,
                 current_index_state,
                 index_display,
                 current_sample_state,
                 last_analysis_state
                 ]
    ).then(
        fn=update_closest_match_dropdown,
        inputs=[closest_list_state, current_index_state],
        outputs=[closest_match_dropdown],
    ).then(
        fn=click_pin_button,
        inputs=[distance_value_dataset,
                closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_match_dropdown,
                closest_template_synonymes,
                current_index_state],
        outputs=[
            pinned_svg_output,
            pinned_icp_output,
            pinned_curvature_plot_output,
            pinned_curvature_color_output,
            pinned_angle_plot_output,
            pinned_sample_id_output,
            pinned_synonyme_output,
            pinned_index_display
        ]
    ).then(
        fn=update_analyze_button_color,
        inputs=analysis_button_inputs,
        outputs=[analyze_button]
    )

    # Jump-to-match dropdown
    # Use `.input` (user interaction only). Programmatic updates via gr.update(value=...) do NOT trigger this.
    closest_match_dropdown.input(
        fn=click_select_closest_sample,
        inputs=[distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_state, closest_list_state, current_index_state,
                smooth_method_dropdown, smooth_factor, smooth_window_slider, samples, closest_match_dropdown],
        outputs=[
            closest_svg_output,
            closest_icp_output,

            closest_curvature_plot_output,
            closest_curvature_color_output,
            closest_angle_plot_output,

            closest_template_synonymes,
            current_index_state,
            closest_sample_id_output,
            index_display,
        ],
    )

    # Dropdown change - loads from database and updates sliders
    crop_svg_dropdown.change(
        fn=change_crop_svg_dropdown,
        inputs=[crop_svg_dropdown],
        outputs=[full_svg_display, cropped_svg_display, crop_start, crop_end, seam_pos]
    )

    # Slider change
    crop_start.change(
        fn=update_crop_preview,
        inputs=[crop_svg_dropdown, crop_start, crop_end, seam_pos],
        outputs=[full_svg_display, cropped_svg_display]
    )

    crop_end.change(
        fn=update_crop_preview,
        inputs=[crop_svg_dropdown, crop_start, crop_end, seam_pos],
        outputs=[full_svg_display, cropped_svg_display]
    )

    seam_pos.change(
        fn=update_crop_preview,
        inputs=[crop_svg_dropdown, crop_start, crop_end, seam_pos],
        outputs=[full_svg_display, cropped_svg_display]
    )

    save_cropped_svg_button.click(
        fn=click_save_cropped_svg,
        inputs=[crop_svg_dropdown, crop_start, crop_end, seam_pos],
        outputs=[save_status]
    )

    auto_crop_button.click(
        fn=click_auto_crop_faulty_svgs,
        inputs=[
            auto_crop_threshold,
            auto_crop_overwrite,
            auto_crop_seam_pos,
            auto_crop_crop_start,
            auto_crop_crop_end,
            auto_crop_dry_run,
        ],
        outputs=[auto_crop_status],
    )

    add_button.click(
        fn=click_add_rule,
        inputs=[name_input, synonym_input],
        outputs=[synonym_table, rule_ids]
    )

    delete_button.click(
        fn=click_delete_rule,
        inputs=[delete_input],
        outputs=[synonym_table, rule_ids, selected_row, selected_label]
    )

    def select_on_row(evt: gr.SelectData):
        return evt.index[0], f"**Selected rule:** row {evt.index[0] + 1}"

    synonym_table.select(
        fn=select_on_row,
        outputs=[selected_row, selected_label]
    )

    pin_button.click(
        fn=click_pin_button,
        inputs=[distance_value_dataset,

                closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_match_dropdown,
                closest_template_synonymes,
                current_index_state],
        outputs=[
            pinned_svg_output,
            pinned_icp_output,
            pinned_curvature_plot_output,
            pinned_curvature_color_output,
            pinned_angle_plot_output,
            pinned_sample_id_output,
            pinned_synonyme_output,
            pinned_index_display
        ]
    )

    csv_download_button.click(
        fn=click_csv_download,
        inputs=[],
        outputs=[csv_file_output, csv_file_output]
    )

    # change color when any relevant control changes
    for control in analysis_controls:
        control.change(
            fn=update_analyze_button_color,
            inputs=analysis_button_inputs,
            outputs=[analyze_button],
        )
