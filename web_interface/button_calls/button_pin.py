import gradio as gr

def click_pin_button(distance_value_dataset,
                      closest_svg_output,
                      closest_icp_output,
                      closest_curvature_plot_output,
                      closest_curvature_color_output,
                      closest_angle_plot_output,
                      closest_sample_id_output,
                      closest_type_output):

    if distance_value_dataset == "ICP":
        closest_svg_output = gr.update(visible=False)
        closest_icp_output = gr.update(visible=True, value=closest_icp_output)
        return (closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_sample_id_output,
                closest_type_output)
    else:
        closest_svg_output = gr.update(visible=True, value=closest_svg_output)
        closest_icp_output = gr.update(visible=False)
        return (closest_svg_output,
                closest_icp_output,
                closest_curvature_plot_output,
                closest_curvature_color_output,
                closest_angle_plot_output,
                closest_sample_id_output,
                closest_type_output)
