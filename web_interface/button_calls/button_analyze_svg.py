from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display, remove_svg_fill
from analysis.compute_curvature_data import compute_curvature_for_all_items, \
    compute_curvature_for_one_item
from analysis.calculation.get_closest_matches_list import get_closest_matches_list
import gradio as gr

from web_interface.graph_generation.generate_graph import generate_graph


def click_analyze_svg(distance_type_dataset, distance_value_dataset, distance_calculation, sample_id, smooth_method,
                      smooth_factor, smooth_window, n_samples, duplicate_synonym_checkbox):
    """
    called by button
    calculates the graph data, stores it in db and displays it
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
        "n_samples": n_samples,
        "duplicate_synonym_checkbox": duplicate_synonym_checkbox,
        "top_k" : None
    }

    # Get the document
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc:
        placeholder_html = f"<p style='color:red;'>❌ No document found for sample_id: {sample_id}</p>"
        return (
            placeholder_html, None, None, None, f"❌ No document found",
            placeholder_html, None, None, None, "❌ No closest match"
        )
    # Use cropped_svg if available, otherwise use cleaned_svg
    svg_to_display = doc.get("cropped_svg") or doc.get("cleaned_svg")

    if not svg_to_display:
        placeholder_html = f"<p style='color:red;'>❌ No SVG data found</p>"
        return (
            placeholder_html, None, None, None, f"❌ No SVG data found",
            placeholder_html, None, None, None, "❌ No closest match, because there was no valid SVG"
        )

    # Remove fill and format the svg for display
    svg_no_fill = remove_svg_fill(svg_to_display)
    svg_html = format_svg_for_display(svg_no_fill)

    # compute curvature data for selected sample and all templates only in some calculations
    if distance_value_dataset == "ICP" or distance_value_dataset == "lip_aligned_angle":
        # Ensure all samples have curvature data, else compute and store it
        analysis_config["distance_type_dataset"] = "other samples"

        # Recompute if outdated OR if closest matches are invalid
        if doc.get("outdated_curvature", False) or not doc.get("closest_matches_valid", False):
            compute_status = compute_curvature_for_one_item(analysis_config, sample_id)
            doc = db_handler.collection.find_one({"sample_id": sample_id})  # Reload doc after update

        # get all plots of current sample
        curvature_plot_img, _ = generate_graph(analysis_config, sample_id, "sample", "curvature_plot")
        curvature_color_img, _ = generate_graph(analysis_config, sample_id, "sample", "curvature_color")
        angle_plot_img, _ = generate_graph(analysis_config, sample_id, "sample", "angle_plot")
        analysis_config["distance_type_dataset"] = "theory types"  # THIS MUST HAPPEN AFTER IT WAS CHANGED A FEW LINES ABOVE
        compute_status = compute_curvature_for_all_items(analysis_config)
    else: # this is for keypoint
        curvature_plot_img = None
        curvature_color_img = None
        angle_plot_img = None
        compute_status = None

    # Find close matches. Recalculate them if curvature data was recalculated and close matches are outdated.
    # Otherwise, load the closest match from the DB
    if not doc or not doc.get("closest_matches_valid", False):
        closest_id, distance, closest_msg = get_closest_matches_list(analysis_config)
    else:
        closest_id = doc["closest_matches"][0]["id"]
        distance = doc["closest_matches"][0]["distance"]

    closest_svg_output = None
    closest_icp_output = None
    closest_icp_img = None

    # if there was no error and an id was found
    if closest_id is not None:
        # Load its SVG
        db_handler.use_collection("svg_template_types")

        # get svg of closest match / icp overlap
        if distance_value_dataset == "ICP" or distance_value_dataset == "lip_aligned_angle":
            # Generate ICP overlap plot
            if distance_value_dataset == "ICP":
                closest_icp_img, _ = generate_graph(analysis_config, closest_id, "template", "overlap_plot")
            else: # laa
                # get template SVG (specifically for laa)
                closest_svg_output, _ = generate_graph(analysis_config, closest_id, "template", "get_template")
            # Load curvature data of closest match and generate plots
            closest_plot_img, _ = generate_graph(analysis_config, closest_id, "template", "curvature_plot")
            closest_color_img, _ = generate_graph(analysis_config, closest_id, "template", "curvature_color")
            closest_angle_img, _ = generate_graph(analysis_config, closest_id, "template", "angle_plot")
            closest_id_text = f"Closest match: {closest_id} (distance={distance:.4f})"
        else: # keypoint
            closest_plot_img = None
            closest_color_img = None
            closest_angle_img = None
            closest_id_text = None
    else: # no closest_id found / there is an error
        closest_svg_output = "<p>No closest match found</p>"
        closest_icp_output = None
        closest_plot_img = None
        closest_color_img = None
        closest_angle_img = None
        closest_id_text = "No closest match found"

    """if duplicate_synonym_checkbox:
        filter_synonym_matches(sample_id)"""

    # is output image
    if distance_value_dataset == "ICP":
        closest_icp_output = gr.update(value=closest_icp_img, visible=True)
        closest_svg_output = gr.update(visible=False)
    else: # or html-svg
        closest_svg_output = gr.update(value=closest_svg_output, visible=True)
        closest_icp_output = gr.update(visible=False)

    # Get the type of the sample from the database
    db_handler.use_collection("svg_raw")
    sample_type = db_handler.get_sample_type(sample_id)
    # Load the full list of closest matches from DB
    closest_matches_list = db_handler.get_closest_matches(sample_id)

    db_handler.use_collection("svg_template_types")
    closest_type = db_handler.get_sample_type(closest_id)

    # Reset navigation state
    current_index = 0  # first one shown is index 0

    final_status_message = f"{compute_status}\n"

    # Return all outputs
    return (
        svg_html,                         # svg_output
        curvature_plot_img,               # curvature_plot_output
        curvature_color_img,              # curvature_color_output
        angle_plot_img,                   # angle_plot_output
        final_status_message,             # status_output

        closest_svg_output,               # closest_svg_output
        closest_icp_output,               # closest_icp_output

        closest_plot_img,                 # closest_curvature_plot_output
        closest_color_img,                # closest_curvature_color_output
        closest_angle_img,                # closest_angle_plot_output

        closest_id_text,                  # closest_sample_id_output
        sample_type,                      # sample_type_output
        closest_type,                     # closest_type_output

        closest_matches_list,             # closest_list_state
        current_index,                    # current_index_state
        f"{current_index+1} / {len(closest_matches_list)}",  # index_display
        sample_id                          # current_sample_state
    )
