import gradio as gr
from analyzeCurvature import action_analyse_svg, find_closest_curvature, compute_or_load_curvature, compute_and_store_curvature_for_all
from database_handler import MongoDBHandler


def action_store_svg(svg_input):
    """
    called by button
    calls database to store the svg files

    :param svg_input:
    :return: message and new dropdown content
    """

    db_handler = MongoDBHandler("svg_data")
    message = db_handler.insert_svg_files(svg_input)

    # Return both status message and dropdown update
    svg_id_list = db_handler.list_svg_ids()
    dropdown_update = gr.update(choices=[str(sid) for sid in svg_id_list])

    return message, dropdown_update


def action_show_and_analyze_svg(sample_id, smooth_method, smooth_factor, smooth_window, n_samples):
    """
    called by button
    calculates the graph data, stores it in db and displays it

    :param sample_id:
    :param smooth_method:
    :param smooth_factor:
    :param smooth_window:
    :param n_samples:
    :return:
    """

    db_handler = MongoDBHandler("svg_data")

    # Validate sample_id
    try:
        sample_id = int(sample_id)
    except (ValueError, TypeError):
        placeholder_html = "<p style='color:red;'>❌ Invalid or no sample selected.</p>"
        return (
            placeholder_html, None, None, "❌ Invalid sample ID.",
            placeholder_html, None, None, "❌ No closest match."
        )

    # Load cleaned SVG
    cleaned_svg, error = db_handler.get_cleaned_svg(sample_id)
    if error:
        placeholder_html = f"<p style='color:red;'>❌ {error}</p>"
        return (
            placeholder_html, None, None, f"❌ {error}",
            placeholder_html, None, None, "❌ No closest match."
        )

    svg_html = format_svg_for_display(cleaned_svg)

    # Ensure all SVGs have curvature data, else compute and store it
    compute_status = compute_and_store_curvature_for_all(
        smooth_method=smooth_method,
        smooth_factor=smooth_factor,
        smooth_window=smooth_window,
        n_samples=n_samples
    )

    # Compute or load curvature for selected sample
    curvature_plot_img, curvature_color_img, status_msg = compute_or_load_curvature(
        sample_id, smooth_method, smooth_factor, smooth_window, n_samples
    )

    # Find closest match
    closest_id, distance, closest_msg = find_closest_curvature(sample_id)
    if closest_id is not None:
        # Load its SVG
        closest_svg_content, closest_error = db_handler.get_cleaned_svg(closest_id)
        if closest_error:
            closest_svg_html = f"<p style='color:red;'>Error loading closest SVG: {closest_error}</p>"
        else:
            closest_svg_html = format_svg_for_display(closest_svg_content)

        # Load its curvature data
        closest_plot_img, closest_color_img, _ = compute_or_load_curvature(
            closest_id,
            smooth_method=smooth_method,
            smooth_factor=smooth_factor,
            smooth_window=smooth_window,
            n_samples=n_samples
        )
        closest_id_text = f"Closest match: {closest_id} (distance={distance:.4f})"
    else:
        closest_svg_html = "<p>No closest match found</p>"
        closest_plot_img = None
        closest_color_img = None
        closest_id_text = "No closest match found"

    # Get the type of the sample from the database
    sample_type = db_handler.get_sample_type(sample_id)
    closest_type = db_handler.get_sample_type(closest_id)

    # Combine status messages
    final_status_message = f"{compute_status}\n{status_msg}"

    # Return all outputs
    return (
        svg_html,                  # Selected SVG
        curvature_plot_img,        # Selected curvature line plot
        curvature_color_img,       # Selected curvature color map
        final_status_message,      # Status message for selected sample
        closest_svg_html,          # Closest SVG
        closest_plot_img,          # Closest curvature line plot
        closest_color_img,         # Closest curvature color map
        closest_id_text            # Text showing closest sample ID + distance
    )


def format_svg_for_display(cleaned_svg):
    """
    Wrap the SVG in a bordered white box for display on the web page.

    :param cleaned_svg: cleaned SVG
    """
    return f"""
    <div style="
        border: 2px solid black;
        background-color: white;
        padding: 10px;
        width: 500px;
        height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        {cleaned_svg}
    </div>
    """


# main webpage code
with gr.Blocks(title="Ceramics Analysis") as demo:
    db_handler = MongoDBHandler("svg_data")

    with gr.Tabs():
        # Tab for all upload related things
        with gr.Tab("Manage files"):
            gr.Markdown("## 🌀 SVG-Krümmungsanalyse\nLade SVG-Dateien hoch und füge zusätzliche information mittels CSV-Datai hinzu.")

            # svg upload
            with gr.Row():
                svg_input = gr.File(label="SVG-Dateien hochladen", file_types=[".svg"], file_count="multiple")
                svg_upload_button = gr.Button("Upload .svg files")

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
                smooth_factor = gr.Slider(0, 5, value=3, step=0.005, label="Glättungsfaktor")
                smooth_window_slider = gr.Slider(3, 300, value=150, step=2, label="Glättungsfenster")
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

                    svg_output = gr.HTML(
                        value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>"
                    )
                    curvature_plot_output = gr.Image(label="Curvature Plot")
                    curvature_color_output = gr.Image(label="Curvature Color Map")

                # Right column: closest svg
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("## Closest Match")
                    closest_sample_id_output = gr.Textbox(label="Closest Sample ID", interactive=False)
                    closest_svg_output = gr.HTML(value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>")
                    closest_curvature_plot_output = gr.Image(label="Curvature Plot")
                    closest_curvature_color_output = gr.Image(label="Curvature Color Map")


# Button logic:
    # svg upload
    svg_upload_button.click(
        fn=action_store_svg,
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
        fn=action_analyse_svg,
        inputs=[],
        outputs=[status_output_text]
    )

    analyze_button.click(
        fn=action_show_and_analyze_svg,
        inputs=[svg_dropdown, smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[svg_output, curvature_plot_output, curvature_color_output, status_output, closest_svg_output,
            closest_curvature_plot_output, closest_curvature_color_output, closest_sample_id_output
        ]
    )