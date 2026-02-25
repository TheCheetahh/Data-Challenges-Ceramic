from __future__ import annotations

import gradio as gr

from database_handler import MongoDBHandler
from web_interface.graph_generation.generate_graph import generate_graph


def _clean_closest_list(closest_list_state):
    """Return list of dicts with at least an 'id' key."""
    if not isinstance(closest_list_state, list):
        return []
    cleaned = []
    for item in closest_list_state:
        if isinstance(item, dict) and item.get("id") is not None:
            cleaned.append(item)
    return cleaned


def update_closest_match_dropdown(closest_list_state, closest_index_state):
    """Build dropdown choices from closest_list_state and select the current index.

    Returns:
        dropdown_update
    """
    cleaned = _clean_closest_list(closest_list_state)
    if not cleaned:
        return gr.update(choices=[], value=None)

    # clamp index
    try:
        idx = int(closest_index_state)
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(cleaned) - 1))

    choices = []
    for rank, item in enumerate(cleaned, start=1):
        template_id = str(item.get("id"))
        dist = item.get("distance")
        if isinstance(dist, (int, float)):
            label = f"{rank}. {template_id} (d={dist:.6f})"
        else:
            label = f"{rank}. {template_id}"
        # (label, value)
        choices.append((label, template_id))

    selected_id = str(cleaned[idx].get("id"))

    # We set value so the UI shows the selection.
    # In the app we must use Dropdown.input (user-only) to avoid double-loading on programmatic updates.
    return gr.update(choices=choices, value=selected_id)


def click_navigate_closest_sample(
    distance_type_dataset,
    distance_value_dataset,
    distance_calculation,
    current_sample_id,
    closest_list_state,
    closest_index_state,
    smooth_method,
    smooth_factor,
    smooth_window,
    n_samples,
    next_or_prev,
):
    """Navigate next/prev within the closest list and render the closest match."""

    cleaned = _clean_closest_list(closest_list_state)
    if not cleaned:
        return (
            gr.update(value="<p>No closest samples available</p>", visible=True),
            gr.update(visible=False),
            None,
            None,
            None,
            "",
            closest_index_state,
            "No closest sample",
            "-/-",
        )

    try:
        base_idx = int(closest_index_state)
    except Exception:
        base_idx = 0

    new_index = (base_idx + int(next_or_prev)) % len(cleaned)
    next_id = str(cleaned[new_index].get("id"))

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_template_types")

    analysis_config = {
        "db_handler": db_handler,
        "sample_id": current_sample_id,
        "distance_type_dataset": distance_type_dataset,
        "distance_value_dataset": distance_value_dataset,
        "distance_calculation": distance_calculation,
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples,
    }

    plot_img, _ = generate_graph(analysis_config, next_id, "template", "curvature_plot")
    color_img, _ = generate_graph(analysis_config, next_id, "template", "curvature_color")
    angle_plot_img, _ = generate_graph(analysis_config, next_id, "template", "angle_plot")

    if distance_value_dataset == "ICP":
        svg_update = gr.update(visible=False)
        icp_img, _ = generate_graph(analysis_config, next_id, "template", "overlap_plot")
        icp_update = gr.update(value=icp_img, visible=True)
    elif distance_value_dataset == "lip_aligned_angle":
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
        next_id,
        f"{new_index + 1} / {len(cleaned)}",
    )


def click_select_closest_sample(
    distance_type_dataset,
    distance_value_dataset,
    distance_calculation,
    current_sample_id,
    closest_list_state,
    closest_index_state,
    smooth_method,
    smooth_factor,
    smooth_window,
    n_samples,
    selected_match,
):
    """Jump to the selected closest match.

    Returns 9 outputs:
      closest_svg_update, closest_icp_update,
      curvature_plot, curvature_color, angle_plot,
      synonyms_text, new_index_state, selected_id_text, index_display
    """

    cleaned = _clean_closest_list(closest_list_state)
    if not cleaned or selected_match is None:
        return (
            gr.update(value="<p>No closest samples available</p>", visible=True),
            gr.update(visible=False),
            None,
            None,
            None,
            "",
            closest_index_state,
            "No closest sample",
            "-/-",
        )

    selected_id = str(selected_match)

    # find index
    new_index = None
    for i, item in enumerate(cleaned):
        if str(item.get("id")) == selected_id:
            new_index = i
            break

    if new_index is None:
        # fallback to current index
        try:
            new_index = int(closest_index_state)
        except Exception:
            new_index = 0
        new_index = max(0, min(new_index, len(cleaned) - 1))
        selected_id = str(cleaned[new_index].get("id"))

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_template_types")

    analysis_config = {
        "db_handler": db_handler,
        "sample_id": current_sample_id,
        "distance_type_dataset": distance_type_dataset,
        "distance_value_dataset": distance_value_dataset,
        "distance_calculation": distance_calculation,
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples,
    }

    plot_img, _ = generate_graph(analysis_config, selected_id, "template", "curvature_plot")
    color_img, _ = generate_graph(analysis_config, selected_id, "template", "curvature_color")
    angle_plot_img, _ = generate_graph(analysis_config, selected_id, "template", "angle_plot")

    if distance_value_dataset == "ICP":
        svg_update = gr.update(visible=False)
        icp_img, _ = generate_graph(analysis_config, selected_id, "template", "overlap_plot")
        icp_update = gr.update(value=icp_img, visible=True)
    elif distance_value_dataset == "Orb" or distance_value_dataset == "DISK":
        svg_update, _ = generate_graph(analysis_config, selected_id, "template", "get_template")
        svg_update = gr.update(visible=True, value=svg_update)
        icp_update = gr.update(visible=False)
    else:
        # laa case. Gets the template svg
        svg_update = gr.update(visible=False)
        icp_img, _ = generate_graph(analysis_config, selected_id, "template", "overlap_plot")
        icp_update = gr.update(value=icp_img, visible=True)

    db_handler.use_collection("svg_synonym_rules")
    rule = db_handler.collection.find_one({"members": selected_id})
    typ_text = ", ".join(rule.get("members", [])) if rule else ""

    return (
        svg_update,
        icp_update,
        plot_img,
        color_img,
        angle_plot_img,
        typ_text,
        new_index,
        selected_id,
        f"{new_index + 1} / {len(cleaned)}",
    )
