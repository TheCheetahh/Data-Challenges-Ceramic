import gradio as gr

def click_pin_button(distance_value_dataset,
                     closest_svg_output,
                     closest_icp_output,
                     closest_curvature_plot_output,
                     closest_curvature_color_output,
                     closest_angle_plot_output,
                     closest_sample_id_output,
                     closest_template_synonymes,
                     current_index_state
                     ):

    if distance_value_dataset == "ICP":
        closest_svg_output = gr.update(visible=False)
        closest_icp_output = gr.update(visible=True, value=closest_icp_output)
        index_state = "Match Rank: " + str(current_index_state + 1)
        return (closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_sample_id_output,
                closest_template_synonymes,
                index_state
                )
    else:
        closest_svg_output = gr.update(visible=False)
        closest_icp_output = gr.update(visible=True, value=closest_icp_output)
        index_state = "Match Rank: " + str(current_index_state + 1)
        return (closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_sample_id_output,
                closest_template_synonymes,
                index_state
                )
