import os
import gradio as gr
from analyzeCurvature import analyze_svg_curvature, analyse_svg
from database_handler import MongoDBHandler


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


def format_svg_for_display(cleaned_svg):
    """
    Wrap the SVG in a bordered white box for display on the web page.
    """
    return f"""
    <div style="
        border: 2px solid black;
        background-color: white;
        padding: 10px;
        display: inline-block;
    ">
        {cleaned_svg}
    </div>
    """

def show_svg_ui(sample_id):
    cleaned_svg, error = db_handler.get_cleaned_svg(sample_id)
    if error:
        return f"<p style='color:red;'>{error}</p>"

    return format_svg_for_display(cleaned_svg)


with gr.Blocks(title="SVG-Krümmungsanalyse") as demo:
    db_handler = MongoDBHandler("svg_data")

    gr.Markdown("## 🌀 SVG-Krümmungsanalyse\nLade eine SVG-Datei hoch und analysiere die Krümmung des Pfads.")

    # svg upload
    with gr.Row():
        svg_input = gr.File(label="SVG-Dateien hochladen", file_types=[".svg"], file_count="multiple")
        svg_upload_button = gr.Button("Upload .svg files")

    # csv upload
    with gr.Row():
        csv_input = gr.File(label="CSV-Datei hochladen", file_types=[".csv"])
        csv_upload_button = gr.Button("Upload .csv file")

    with gr.Row():
        output_dir_input = gr.Textbox(label="Ausgabeverzeichnis", value="./outputs")

    with gr.Row():
        smooth_method_dropdown = gr.Dropdown(choices=["savgol", "gauss", "bspline", "none"], value="savgol", label="Glättungsmethode")
        smooth_factor = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Glättungsfaktor")
        smooth_window_slider = gr.Slider(3, 51, value=15, step=2, label="Glättungsfenster")
        samples = gr.Slider(200, 5000, value=1000, step=100, label="Anzahl Abtastpunkte")

    clean_svg_button = gr.Button("🚀 Clean SVG")

    with gr.Row():
        output_text = gr.Textbox(label="Status", interactive=False)

    svg_id_list = db_handler.list_svg_ids()
    svg_dropdown = gr.Dropdown(
        choices=[str(sid) for sid in svg_id_list],
        label="Select SVG to display"
    )
    show_button = gr.Button("Show SVG")

    svg_output = gr.HTML()

    # svg upload
    svg_upload_button.click(
        fn=db_handler.insert_svg_files,
        inputs=[svg_input],
        outputs=[output_text]
    )

    # csv upload
    csv_upload_button.click(
        fn=db_handler.add_csv_data,
        inputs=[csv_input],
        outputs=[output_text]
    )

    clean_svg_button.click(
        fn=analyse_svg,
        inputs=[],
        outputs=[output_text]
    )

    show_button.click(
        fn=show_svg_ui,
        inputs=[svg_dropdown],
        outputs=svg_output
    )

demo.launch()
