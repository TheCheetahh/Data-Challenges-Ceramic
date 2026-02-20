from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display
from analysis.calculation.icp.icp import generate_icp_overlap_image
import gradio as gr

from web_interface.graph_generation.generate_graph import generate_graph


def click_navigate_closest_sample(distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_id, closest_list_state, closest_index_state, smooth_method, smooth_factor, smooth_window, n_samples, next_or_prev):
    """
    Show the next closest sample using Gradio state variables.

    Returns:
        svg_html, plot_img, color_img, angle_plot_img,
        type_text, new_index, label_text
    """

    # get a database handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_template_types")

    # If state is empty or invalid → fallback
    if not isinstance(closest_list_state, list) or len(closest_list_state) == 0:
        return (
            gr.update(value="<p>No closest samples available</p>", visible=True),
            gr.update(visible=False),

            None,
            None,
            None,

            "No type",
            closest_index_state,
            "No closest sample",
            "-/-"
        )

    # --- ensure list contains only valid dict entries ---
    cleaned_list = []
    for item in closest_list_state:
        if isinstance(item, dict) and "id" in item:
            cleaned_list.append(item)
    closest_list_state = cleaned_list

    if len(closest_list_state) == 0:
        return (
            gr.update(value="<p>No closest samples available</p>", visible=True),
            gr.update(visible=False),

            None,
            None,
            None,

            "No type",
            closest_index_state,
            "No closest sample",
            "-/-"
        )

    # --- compute next index ---
    new_index = (closest_index_state + next_or_prev) % len(closest_list_state)

    # --- get the next item ---
    next_item = closest_list_state[new_index]
    next_id = next_item.get("id")
    distance = next_item.get("distance")

    # --- build label ---
    if distance is not None:
        label_text = f"Closest sample: {next_id} (distance={distance:.6f})"
    else:
        label_text = f"Closest sample: {next_id}"

    analysis_config = {
        "db_handler": db_handler,
        "sample_id": current_sample_id,
        "distance_type_dataset": distance_type_dataset,
        "distance_value_dataset": distance_value_dataset,
        "distance_calculation": distance_calculation,
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples
    }

    # --- load SVG + curvature plots + type text ---
    plot_img, _ = generate_graph(analysis_config, next_id, "template", "curvature_plot")
    color_img, _ = generate_graph(analysis_config, next_id, "template", "curvature_color")
    angle_plot_img, _ = generate_graph(analysis_config, next_id, "template", "angle_plot")

    # Load cleaned SVG of that sample id
    if distance_value_dataset == "ICP":
        # hide SVG
        svg_update = gr.update(visible=False)

        # generate NEW ICP plot for the new closest sample
        icp_img , _ = generate_graph(analysis_config, next_id, "template", "overlap_plot")
        icp_update = gr.update(value=icp_img, visible=True)

    else:
        # laa case. Gets the template svg
        svg_html, _ = generate_graph(analysis_config, next_id, "template", "get_template")
        svg_update = gr.update(value=svg_html, visible=True)
        icp_update = gr.update(visible=False)

    typ_text = db_handler.get_sample_type(next_id)

    return (
        svg_update,     # closest_svg_output (gr.update)
        icp_update,     # closest_icp_output (gr.update)

        plot_img,
        color_img,
        angle_plot_img,

        typ_text,
        new_index,
        label_text,
        f"{new_index+1} / {len(closest_list_state)}"
    )
