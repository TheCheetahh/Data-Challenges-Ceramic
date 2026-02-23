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

    db_handler.use_collection("svg_synonym_rules")
    rule = db_handler.collection.find_one({"members": next_id})
    typ_text = ", ".join(rule.get("members", [])) if rule else ""

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




def update_closest_match_dropdown(closest_list_state, closest_index_state):
    """Build dropdown choices from closest_list_state and select the current index."""

    if not isinstance(closest_list_state, list) or len(closest_list_state) == 0:
        return gr.update(choices=[], value=None)

    cleaned_list = [x for x in closest_list_state if isinstance(x, dict) and "id" in x]
    if len(cleaned_list) == 0:
        return gr.update(choices=[], value=None)

    # Guard index
    if not isinstance(closest_index_state, int):
        closest_index_state = 0
    closest_index_state = max(0, min(closest_index_state, len(cleaned_list) - 1))

    # Use (label, value) tuples so the UI can show distance while the value stays the template id
    choices = []
    for i, item in enumerate(cleaned_list):
        sid = str(item.get("id"))
        dist = item.get("distance")
        if dist is None:
            label = f"{i+1}: {sid}"
        else:
            try:
                label = f"{i+1}: {sid} (d={float(dist):.6g})"
            except Exception:
                label = f"{i+1}: {sid} (d={dist})"
        choices.append((label, sid))

    current_id = str(cleaned_list[closest_index_state].get("id"))
    return gr.update(choices=choices, value=current_id)


def click_select_closest_sample(distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_id, closest_list_state, closest_index_state, smooth_method, smooth_factor, smooth_window, n_samples, selected_match):
    """Jump directly to a selected closest match (dropdown)."""

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

    cleaned_list = [x for x in closest_list_state if isinstance(x, dict) and "id" in x]
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

    # --- determine target id from dropdown value ---
    target_id = None
    if isinstance(selected_match, dict) and "id" in selected_match:
        target_id = selected_match.get("id")
    elif selected_match is not None:
        target_id = str(selected_match)

    # Find index; fallback to current index
    new_index = None
    if target_id is not None:
        for i, item in enumerate(closest_list_state):
            if str(item.get("id")) == str(target_id):
                new_index = i
                break
    if new_index is None:
        # keep current (but guard)
        if not isinstance(closest_index_state, int):
            closest_index_state = 0
        new_index = max(0, min(closest_index_state, len(closest_list_state) - 1))

    next_item = closest_list_state[new_index]
    next_id = next_item.get("id")
    distance = next_item.get("distance")

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

    plot_img, _ = generate_graph(analysis_config, next_id, "template", "curvature_plot")
    color_img, _ = generate_graph(analysis_config, next_id, "template", "curvature_color")
    angle_plot_img, _ = generate_graph(analysis_config, next_id, "template", "angle_plot")

    if distance_value_dataset == "ICP":
        svg_update = gr.update(visible=False)
        icp_img, _ = generate_graph(analysis_config, next_id, "template", "overlap_plot")
        icp_update = gr.update(value=icp_img, visible=True)
    else:
        svg_html, _ = generate_graph(analysis_config, next_id, "template", "get_template")
        svg_update = gr.update(value=svg_html, visible=True)
        icp_update = gr.update(visible=False)

    db_handler.use_collection("svg_synonym_rules")
    rule = db_handler.collection.find_one({"members": next_id})
    typ_text = ", ".join(rule.get("members", [])) if rule else ""

    return (
        svg_update,
        icp_update,
        plot_img,
        color_img,
        angle_plot_img,
        typ_text,
        new_index,
        label_text,
        f"{new_index+1} / {len(closest_list_state)}"
    )
