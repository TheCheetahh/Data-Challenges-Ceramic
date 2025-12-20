from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display
from analysis.compute_curvature_data import generate_all_plots, compute_curvature_for_all_items, \
    find_enhanced_closest_curvature, compute_curvature_for_one_item


def click_analyze_svg(distance_type_dataset, distance_value_dataset, distance_calculation, sample_id, smooth_method,
                      smooth_factor, smooth_window, n_samples):
    """
    called by button
    calculates the graph data, stores it in db and displays it

    :param distance_type_dataset:
    :param distance_calculation:
    :param distance_value_dataset:
    :param sample_id:
    :param smooth_method:
    :param smooth_factor:
    :param smooth_window:
    :param n_samples:
    :return:
    """

    # get a database handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    analysis_config = {
        "db_handler": db_handler,
        "sample_id": sample_id,
        "distance_type_dataset": distance_type_dataset,
        "distance_value_dataset": distance_value_dataset,
        "distance_calculation": distance_calculation,
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples
    }

    # Load cleaned SVG of that sample id
    cleaned_svg, error = db_handler.get_cleaned_svg(sample_id)
    if error:
        placeholder_html = f"<p style='color:red;'>❌ {error}</p>"
        return (
            placeholder_html, None, None, f"❌ {error}",
            placeholder_html, None, None, "❌ No closest match, because there was no valid cleaned_svg"
        )
    # format the svg, so it can be displayed on the web page
    svg_html = format_svg_for_display(cleaned_svg)

    # Ensure all samples have curvature data, else compute and store it
    if distance_type_dataset == "other samples":
        compute_status = compute_curvature_for_all_items(analysis_config)

        # get all plots of current sample
        curvature_plot_img, curvature_color_img, angle_plot_img, status_msg = generate_all_plots(analysis_config)
    else:
        analysis_config["distance_type_dataset"] = "other samples"
        compute_status = compute_curvature_for_one_item(analysis_config, sample_id)

        # get all plots of current sample
        curvature_plot_img, curvature_color_img, angle_plot_img, status_msg = generate_all_plots(analysis_config)
        analysis_config["distance_type_dataset"] = "theory types"
        compute_status = compute_curvature_for_all_items(analysis_config)

    # Find close match
    closest_id, distance, closest_msg = find_enhanced_closest_curvature(analysis_config)
    if closest_id is not None:
        # Load its SVG
        if distance_type_dataset == "other samples":
            db_handler.use_collection("svg_raw")
        else:
            db_handler.use_collection("svg_template_types")

        closest_svg_content, closest_error = db_handler.get_cleaned_svg(closest_id)
        if closest_error:
            closest_svg_html = f"<p style='color:red;'>Error loading closest SVG: {closest_error}</p>"
        else:
            closest_svg_html = format_svg_for_display(closest_svg_content)

        # Load its curvature data
        closest_plot_img, closest_color_img, closest_angle_img, _ = generate_all_plots(analysis_config)
        closest_id_text = f"Closest match: {closest_id} (distance={distance:.4f})"
    else:
        closest_svg_html = "<p>No closest match found</p>"
        closest_plot_img = None
        closest_color_img = None
        closest_angle_img = None
        closest_id_text = "No closest match found"

    # Get the type of the sample from the database
    db_handler.use_collection("svg_raw")
    sample_type = db_handler.get_sample_type(sample_id)
    # Load the full list of closest matches from DB
    closest_matches_list = db_handler.get_closest_matches(sample_id)

    if distance_type_dataset == "other samples":
        db_handler.use_collection("svg_raw")
    else:
        db_handler.use_collection("svg_template_types")
    closest_type = db_handler.get_sample_type(closest_id)

    # Reset navigation state
    current_index = 0  # first one shown is index 0

    # Combine status messages
    final_status_message = f"{compute_status}\n{status_msg}"

    # Return all outputs
    return (
        svg_html,  # Selected SVG
        curvature_plot_img,  # Selected curvature line plot
        curvature_color_img,  # Selected curvature color map
        angle_plot_img,
        final_status_message,  # Status message for selected sample
        closest_svg_html,  # Closest SVG
        closest_plot_img,  # Closest curvature line plot
        closest_color_img,  # Closest curvature color map
        closest_angle_img,
        closest_id_text,  # Text showing close sample ID + distance
        sample_type,
        closest_type,
        closest_matches_list,  # list of closest matches
        current_index  # starting index is always 0
    )
