from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display, remove_svg_fill
from analysis.compute_curvature_data import generate_all_plots, compute_curvature_for_all_items, \
    find_enhanced_closest_curvature, compute_curvature_for_one_item
from analysis.icp import generate_icp_overlap_image
import gradio as gr
from web_interface.other_gradio_components.checkbox_synonym import filter_synonym_matches


def click_analyze_svg(distance_type_dataset, distance_value_dataset, distance_calculation, sample_id, smooth_method,
                      smooth_factor, smooth_window, n_samples, duplicate_synonym_checkbox):
    """
    called by button
    calculates the graph data, stores it in db and displays it
    """

    # get a database handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Default outputs
    closest_svg_update = gr.update(visible=False)
    closest_icp_update = gr.update(visible=False)

    closest_plot_img = None
    closest_color_img = None
    closest_angle_img = None

    closest_id_text = "No closest match found"
    closest_type = None
    sample_type = None
    final_status_message = ""
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

    analysis_config["icp_skipped_targets"] = []

    # Get the document to check for cropped_svg
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc:
        placeholder_html = f"<p style='color:red;'>❌ No document found for sample_id: {sample_id}</p>"
        return (
            placeholder_html,     # svg_output
            None,                 # curvature_plot
            None,                 # curvature_color
            None,                 # angle_plot
            "❌ No document found",# status_output

            placeholder_html,     # closest_svg_output
            None,                 # closest_icp_output

            None,                 # closest_curvature_plot
            None,                 # closest_curvature_color
            None,                 # closest_angle_plot

            "❌ No closest match", # closest_sample_id_output
            None,                 # sample_type_output
            None,                 # closest_type_output

            [],                   # closest_list_state
            0,                    # current_index_state
            "0 / 0",              # index_display
            sample_id             # current_sample_state
        )

    # Use cropped_svg if available, otherwise use cleaned_svg
    svg_to_display = doc.get("cropped_svg") or doc.get("cleaned_svg")

    if not svg_to_display:
        placeholder_html = "<p style='color:red;'>❌ No SVG data found</p>"
        return (
            placeholder_html,     # svg_output
            None,                 # curvature_plot
            None,                 # curvature_color
            None,                 # angle_plot
            "❌ No SVG data found",# status_output

            placeholder_html,     # closest_svg_output
            None,                 # closest_icp_output

            None,                 # closest_curvature_plot
            None,                 # closest_curvature_color
            None,                 # closest_angle_plot

            "❌ No closest match", # closest_sample_id_output
            None,                 # sample_type_output
            None,                 # closest_type_output

            [],                   # closest_list_state
            0,                    # current_index_state
            "0 / 0",              # index_display
            sample_id             # current_sample_state
        )

    # Remove fill and format the svg for display
    svg_no_fill = remove_svg_fill(svg_to_display)
    svg_html = format_svg_for_display(svg_no_fill)

    # Ensure all samples have curvature data, else compute and store it
    # compute curvature data for selected sample and all templates
    analysis_config["distance_type_dataset"] = "other samples"

    # Recompute if outdated OR if closest matches are invalid
    if doc.get("outdated_curvature", False) or not doc.get("closest_matches_valid", False):
        compute_status = compute_curvature_for_one_item(analysis_config, sample_id)
        doc = db_handler.collection.find_one({"sample_id": sample_id})  # Reload doc after update

    # get all plots of current sample
    curvature_plot_img, curvature_color_img, angle_plot_img, status_msg = generate_all_plots(analysis_config)
    analysis_config["distance_type_dataset"] = "theory types"  # THIS MUST HAPPEN AFTER IT WAS CHANGED A FEW LINES ABOVE
    compute_status = compute_curvature_for_all_items(analysis_config)

    # Find close matches. Recalculate them if curvature data was recalculated and close matches are outdated.
    # Otherwise, load the closest match from the DB
    closest_id = None
    distance = None

    if not doc or not doc.get("closest_matches_valid", False):
        closest_id, distance, closest_msg = find_enhanced_closest_curvature(analysis_config)
        doc = db_handler.collection.find_one({"sample_id": sample_id})
    else:
        matches = doc.get("closest_matches", [])
        if matches:
            closest_id = matches[0].get("id")
            distance = matches[0].get("distance")
    # --------------------------------------------------
    # Handle ICP target failure (all distances = inf)
    # --------------------------------------------------
    icp_error = analysis_config.get("icp_target_error")

    if distance_value_dataset == "ICP" and icp_error:
        db_handler.use_collection("svg_raw")
        db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {
                "closest_matches": [],
                "full_closest_matches": [],
                "closest_matches_valid": False,
                "icp_status": f"ICP failed for target: {icp_error}"
            }}
        )

        final_status_message = (
            f"❌ ICP failed for target shape.\n"
            f"Reason: {icp_error}"
        )

        return (
            svg_html,
            curvature_plot_img,
            curvature_color_img,
            angle_plot_img,
            final_status_message,

            gr.update(visible=False),
            gr.update(visible=False),

            None,
            None,
            None,

            "No closest match (ICP failed)",
            sample_type,
            None,

            [],
            0,
            "0 / 0",
            sample_id
        )
    else:
        matches = doc.get("closest_matches", []) if doc else []

        closest_id = None
        distance = None

        if matches:
            closest_id = matches[0].get("id")
            distance = matches[0].get("distance")

    # if there was no error and an id was found
    if closest_id is not None:
        # Load its SVG
        db_handler.use_collection("svg_template_types")

        # get svg of closest match
        if distance_value_dataset == "ICP":
            closest_svg_update = gr.update(visible=False)

            closest_icp_img = generate_icp_overlap_image(
                db_handler,
                sample_id,
                closest_id,
                analysis_config
            )
            closest_icp_update = gr.update(value=closest_icp_img, visible=True)

        else:
            closest_svg_content, closest_error = db_handler.get_cleaned_svg(closest_id)
            if closest_error:
                html = f"<p style='color:red;'>Error loading closest SVG</p>"
            else:
                closest_svg_no_fill = remove_svg_fill(closest_svg_content)
                html = format_svg_for_display(closest_svg_no_fill)

            closest_svg_update = gr.update(value=html, visible=True)
            closest_icp_update = gr.update(visible=False)

        # Load curvature data of closest match and generate plots
        analysis_config["sample_id"] = closest_id
        closest_plot_img, closest_color_img, closest_angle_img, _ = generate_all_plots(analysis_config)
        closest_id_text = f"Closest match: {closest_id} (distance={distance:.4f})"
        analysis_config["sample_id"] = sample_id  # reset sample id
    else:
        closest_svg_html = "<p>No closest match found</p>"
        closest_plot_img = None
        closest_color_img = None
        closest_angle_img = None
        closest_id_text = "No closest match found"

    """if duplicate_synonym_checkbox:
        filter_synonym_matches(sample_id)"""

    # Get the type of the sample from the database
    db_handler.use_collection("svg_raw")
    sample_type = db_handler.get_sample_type(sample_id)
    # Load the full list of closest matches from DB
    closest_matches_list = db_handler.get_closest_matches(sample_id)

    db_handler.use_collection("svg_template_types")
    closest_type = db_handler.get_sample_type(closest_id)

    # Reset navigation state
    current_index = 0  # first one shown is index 0

    # Return all outputs
    return (
        svg_html,                         # svg_output
        curvature_plot_img,               # curvature_plot_output
        curvature_color_img,              # curvature_color_output
        angle_plot_img,                   # angle_plot_output
        final_status_message,             # status_output

        closest_svg_update,               # closest_svg_output (gr.update)
        closest_icp_update,               # closest_icp_output (gr.update)

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