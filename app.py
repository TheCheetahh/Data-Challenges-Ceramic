import os
import gradio as gr
from analyzeCurvature import analyze_svg_curvature
from database_handler import MongoDBHandler

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
#"""

def helloworld():
    return "helloworld"

def upload_svg(svg_file):
    """Upload SVG using the database handler."""
    if svg_file is None:
        return "No file uploaded."
    return db_handler.insert_svg_file(svg_file)

with gr.Blocks(title="SVG-Krümmungsanalyse") as demo:
    db_handler = MongoDBHandler("svg_data")

    gr.Markdown("## 🌀 SVG-Krümmungsanalyse\nLade eine SVG-Datei hoch und analysiere die Krümmung des Pfads.")

    with gr.Row():
        svg_input = gr.File(label="SVG-Datei hochladen", file_types=[".svg"])
        output_dir_input = gr.Textbox(label="Ausgabeverzeichnis", value="./outputs")
        upload_button = gr.Button("Upload .svg files")

    with gr.Row():
        smooth_method_dropdown = gr.Dropdown(choices=["savgol", "gauss", "bspline", "none"], value="savgol", label="Glättungsmethode")
        smooth_factor = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Glättungsfaktor")
        smooth_window_slider = gr.Slider(3, 51, value=15, step=2, label="Glättungsfenster")
        samples = gr.Slider(200, 5000, value=1000, step=100, label="Anzahl Abtastpunkte")

    run_button = gr.Button("🚀 Analyse starten")

    output_text = gr.Textbox(label="Status", interactive=False)
    curvature_plot = gr.Image(label="Krümmungsdiagramm")
    color_map = gr.Image(label="Farbkarte der Krümmung")

    upload_button.click(
        fn=upload_svg,
        inputs=[svg_input],
        outputs=[output_text]
    )

    run_button.click(
        fn=helloworld,
        inputs=[],
        outputs=[output_text]
    )

demo.launch()
