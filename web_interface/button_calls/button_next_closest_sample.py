from analysis.compute_curvature_data import generate_all_plots
from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display


def click_next_closest_sample(distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_id, closest_list_state, closest_index_state, smooth_method, smooth_factor, smooth_window, n_samples):
    """
    Show the next closest sample using Gradio state variables.

    Returns:
        svg_html, plot_img, color_img, angle_plot_img,
        type_text, new_index, label_text
    """

    # get a database handler
    db_handler = MongoDBHandler("svg_data")
    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")

    # If state is empty or invalid → fallback
    if not isinstance(closest_list_state, list) or len(closest_list_state) == 0:
        return (
            "<p>No closest samples available</p>",
            None, None, None,
            "No type",
            0,
            "No closest sample"
        )

    # --- ensure list contains only valid dict entries ---
    cleaned_list = []
    for item in closest_list_state:
        if isinstance(item, dict) and "id" in item:
            cleaned_list.append(item)
    closest_list_state = cleaned_list

    if len(closest_list_state) == 0:
        return (
            "<p>No usable closest samples available</p>",
            None, None, None,
            "No type",
            0,
            "No closest sample"
        )

    # --- compute next index ---
    new_index = (closest_index_state + 1) % len(closest_list_state)

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
        "sample_id": next_id,
        "distance_type_dataset": distance_type_dataset,
        "distance_value_dataset": distance_value_dataset,
        "distance_calculation": distance_calculation,
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples
    }

    # --- load SVG + curvature plots + type text ---
    plot_img, color_img, angle_plot_img, _ = generate_all_plots(analysis_config)

    # Load cleaned SVG of that sample id
    cleaned_svg, error = db_handler.get_cleaned_svg(next_id)
    if error:
        placeholder_html = f"<p style='color:red;'>❌ {error}</p>"
        return (
            placeholder_html, None, None, f"❌ {error}",
            placeholder_html, None, None, "❌ No closest match."
        )
    # format the svg, so it can be displayed on the web page
    svg_html = format_svg_for_display(cleaned_svg)

    typ_text = db_handler.get_sample_type(next_id)

    return (
        svg_html,
        plot_img,
        color_img,
        angle_plot_img,
        typ_text,
        new_index,
        label_text,
        f"{new_index+1} / 20"
    )
