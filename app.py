import gradio as gr

from analyzeCurvature import analyze_svg_curvature, analyse_svg, load_and_plot_curvature, \
    compute_and_store_curvature_for_all, find_closest_curvature
from database_handler import MongoDBHandler

def store_svg(svg_input):
    db_handler = MongoDBHandler("svg_data")
    message = db_handler.insert_svg_files(svg_input)
    # Return both status message and dropdown update
    svg_id_list = db_handler.list_svg_ids()
    dropdown_update = gr.update(choices=[str(sid) for sid in svg_id_list])
    return message, dropdown_update

def refresh_svg_dropdown(db_handler):
    # Fetch the latest SVG IDs from the database
    svg_id_list = db_handler.list_svg_ids()
    # Return an updated Dropdown object
    return gr.Dropdown.update(choices=[str(sid) for sid in svg_id_list])


def show_and_analyze_svg(sample_id, smooth_method, smooth_factor, smooth_window, n_samples):
    # Get the cleaned SVG
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")
    cleaned_svg, error = db_handler.get_cleaned_svg(sample_id)

    if error:
        svg_html = f"<p style='color:red;'>{error}</p>"
        return svg_html, None, None, error

    svg_html = format_svg_for_display(cleaned_svg)

    # Compute and store curvature data
    compute_status = compute_and_store_curvature_for_all(
        smooth_method=smooth_method,
        smooth_factor=smooth_factor,
        smooth_window=smooth_window,
        n_samples=n_samples
    )

    # If computation failed
    if compute_status.startswith("❌"):
        return svg_html, None, None, compute_status

    # load plot for the selected one
    curvature_plot_img, curvature_color_img, plot_status = load_and_plot_curvature(sample_id)

    # Run curvature analysis
    curvature_plot_img, curvature_color_img, status_msg = analyze_svg_curvature( # TODO why call this function here
        sample_id, smooth_method, smooth_factor, smooth_window, n_samples)

    closest_id, distance, msg = find_closest_curvature(sample_id)
    if closest_id is not None:
        closest_plot_img, closest_color_img, _ = load_and_plot_curvature(closest_id)
        closest_id_text = f"Closest match: {closest_id} (distance={distance:.4f})"

        # Load the closest SVG
        closest_cleaned_svg, err = db_handler.get_cleaned_svg(closest_id)
        if not err:
            closest_svg_html = format_svg_for_display(closest_cleaned_svg)
        else:
            closest_svg_html = f"<p style='color:red;'>Error loading closest SVG: {err}</p>"
    else:
        closest_plot_img = None
        closest_color_img = None
        closest_svg_html = "<div style='width:300px; height:300px; border:1px solid #ccc;'>SVG not found</div>"
        closest_id_text = "No closest match found"

    final_status_message = f"{compute_status}\n{plot_status}"

    return svg_html, curvature_plot_img, curvature_color_img, final_status_message, closest_svg_html, closest_plot_img, closest_color_img, closest_id_text


"""
def run_analysis(svg_file, output_dir, smooth_method, smooth_factor, smooth_window, num_samples):
    # 1️⃣ Validierung
    if svg_file is None:
        return "❌ Keine Datei hochgeladen.", None, None
    
    # 2️⃣ Standard-Ausgabeverzeichnis sicherstellen
    if not output_dir:
        output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 3️⃣ Analyse aufrufen
        output_paths = analyze_svg_curvature(
            svg_file.name, output_dir, smooth_method, smooth_factor, smooth_window, num_samples
        )

        # 4️⃣ Erwartete Rückgabe prüfen
        curvature_plot = output_paths.get("curvature_plot") if isinstance(output_paths, dict) else None
        color_map = output_paths.get("color_map") if isinstance(output_paths, dict) else None

        if not curvature_plot:
            return "⚠️ Analyse abgeschlossen, aber keine Plot-Datei gefunden.", None, None

        return "✅ Analyse abgeschlossen!", curvature_plot, color_map

    except Exception as e:
        return f"🚨 Fehler: {str(e)}", None, None
"""


def format_svg_for_display(cleaned_svg):
    """Wrap the SVG in a bordered white box for display on the web page."""
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

def show_svg_ui(sample_id):
    """Get SVG by sample_id. and format it for display."""
    cleaned_svg, error = db_handler.get_cleaned_svg(sample_id)
    if error:
        return f"<p style='color:red;'>{error}</p>"

    return format_svg_for_display(cleaned_svg)


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
                smooth_factor = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Glättungsfaktor")
                smooth_window_slider = gr.Slider(3, 51, value=15, step=2, label="Glättungsfenster")
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

                # Right column: reserved for other content
                with gr.Column(scale=1, min_width=400):
                    gr.Markdown("## Closest Match")
                    closest_sample_id_output = gr.Textbox(label="Closest Sample ID", interactive=False)
                    closest_svg_output = gr.HTML(value="<div style='width:500px; height:500px; border:1px solid #ccc; display:flex; align-items:center; justify-content:center;'>SVG will appear here</div>")
                    closest_curvature_plot_output = gr.Image(label="Curvature Plot")
                    closest_curvature_color_output = gr.Image(label="Curvature Color Map")


# Button logic:
    # svg upload
    svg_upload_button.click(
        fn=store_svg,
        inputs=[svg_input],
        outputs=[status_output_text, svg_dropdown]
    )

    # csv upload
    csv_upload_button.click(
        fn=db_handler.add_csv_data,
        inputs=[csv_input],
        outputs=[status_output_text]
    )

    clean_svg_button.click(
        fn=analyse_svg,
        inputs=[],
        outputs=[status_output_text]
    )

    analyze_button.click(
        fn=show_and_analyze_svg,
        inputs=[svg_dropdown, smooth_method_dropdown, smooth_factor, smooth_window_slider, samples],
        outputs=[svg_output, curvature_plot_output, curvature_color_output, status_output, closest_svg_output,
            closest_curvature_plot_output, closest_curvature_color_output, closest_sample_id_output
        ]
    )