from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import format_svg_for_display, remove_svg_fill
from analysis.compute_curvature_data import generate_all_plots, compute_curvature_for_all_items, \
    find_enhanced_closest_curvature, compute_curvature_for_one_item
import numpy as np
from analysis.icp import plot_icp_overlap, find_icp_closest_matches

def click_analyze_svg(distance_type_dataset, distance_value_dataset, distance_calculation, sample_id, smooth_method,
                      smooth_factor, smooth_window, n_samples):
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
        "n_samples": n_samples
    }

    # Get the document to check for cropped_svg
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

    # Ensure all samples have curvature data, else compute and store it
    # compute curvature data for selected sample and all templates
    analysis_config["distance_type_dataset"] = "other samples"

    # Recompute if outdated OR if closest matches are invalid
    if doc.get("outdated_curvature", False) or not doc.get("closest_matches_valid", False):
        compute_status = compute_curvature_for_one_item(analysis_config, sample_id)
        doc = db_handler.collection.find_one({"sample_id": sample_id})  # Reload doc after update

    # get all plots of current sample
    if distance_calculation != "ICP":
        curvature_plot_img, curvature_color_img, angle_plot_img, status_msg = generate_all_plots(analysis_config)
    else:
        curvature_plot_img = None
        curvature_color_img = None
        angle_plot_img = None
        status_msg = "ICP analysis completed"
    analysis_config["distance_type_dataset"] = "theory types"  # THIS MUST HAPPEN AFTER IT WAS CHANGED A FEW LINES ABOVE
    compute_status = compute_curvature_for_all_items(analysis_config)

    # Find close matches. Recalculate them if curvature data was recalculated and close matches are outdated.
    # Otherwise, load the closest match from the DB
    if distance_calculation == "ICP":
        matches = find_icp_closest_matches(
            analysis_config,
            top_k=5
        )

        if not matches:
            closest_id = None
            distance = None
            closest_matches_list = []
        else:
            closest_id = matches[0]["id"]
            distance = matches[0]["distance"]
            closest_matches_list = matches

    else:
        if not doc or not doc.get("closest_matches_valid", False):
            closest_id, distance, _ = find_enhanced_closest_curvature(analysis_config)
        else:
            closest_id = doc["closest_matches"][0]["id"]
            distance = doc["closest_matches"][0]["distance"]

        closest_matches_list = db_handler.get_closest_matches(sample_id)

    # if there was no error and an id was found
    if closest_id is not None:
        # Load its SVG
        db_handler.use_collection("svg_template_types")

        # get svg of closest match
        closest_svg_content, closest_error = db_handler.get_cleaned_svg(closest_id)
        if closest_error:
            closest_svg_html = f"<p style='color:red;'>Error loading closest SVG: {closest_error}</p>"
        else:
            closest_svg_no_fill = remove_svg_fill(closest_svg_content)
            closest_svg_html = format_svg_for_display(closest_svg_no_fill)

        # Load curvature data of closest match and generate plots
        if distance_calculation != "ICP":
            analysis_config["sample_id"] = closest_id
            closest_plot_img, closest_color_img, closest_angle_img, _ = generate_all_plots(analysis_config)
            closest_id_text = f"Closest match: {closest_id} (distance={distance:.4f})"
            analysis_config["sample_id"] = sample_id  # reset sample id
        else:
            # target comes from svg_raw
            db_handler.use_collection("svg_raw")
            target_doc = db_handler.collection.find_one({"sample_id": sample_id})

            # reference comes from svg_template_types
            db_handler.use_collection("svg_template_types")
            ref_doc = db_handler.collection.find_one({"sample_id": closest_id})

            target_pts = np.array(target_doc["icp_data"]["outline_points"])
            ref_pts = np.array(ref_doc["icp_data"]["outline_points"])

            aligned_target_pts = np.array(matches[0]["aligned_target"])

            overlap_img = plot_icp_overlap(
                target_pts,
                aligned_target_pts,
                ref_pts
            )

            closest_plot_img = overlap_img
            closest_color_img = None
            closest_angle_img = None

            closest_id_text = f"Closest match (ICP): {closest_id} (distance={distance:.4f})"
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
    if distance_calculation != "ICP":
        closest_matches_list = db_handler.get_closest_matches(sample_id)

    db_handler.use_collection("svg_template_types")
    closest_type = db_handler.get_sample_type(closest_id)

    # Reset navigation state
    current_index = 0  # first one shown is index 0

    # Combine status messages
    final_status_message = f"{compute_status}\n{status_msg}"

    if distance_calculation == "ICP" and closest_id is not None:
        # hide ALL curvature plots (target + closest)
        curvature_plot_img = None
        curvature_color_img = None
        angle_plot_img = None

        closest_color_img = None
        closest_angle_img = None

    # Return all outputs
    return (
        svg_html,  # Selected SVG (cropped if available)
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
        current_index,  # starting index is always 0
        f"{1} / 20"
    )