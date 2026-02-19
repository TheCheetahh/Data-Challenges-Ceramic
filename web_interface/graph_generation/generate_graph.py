from web_interface.graph_generation.laa.laa_graphs import laa_generate_curvature_lineplot, laa_generate_curvature_color_map, \
    laa_generate_direction_lineplot


def generate_graph(analysis_config, target_id, target_type, graph_type):
    """

    :param analysis_config:
    :param target_id: sample or template id
    :param target_type: sample or template
    :param graph_type: name of the graph to generate
    :return:
    """

    distance_value_dataset = analysis_config.get("distance_value_dataset")

    if distance_value_dataset == "Keypoints":
        graph = None
        message = "Keypoints have no graphs yet"

    elif distance_value_dataset == "lip_aligned_angle":
        if graph_type == "curvature_plot":
            graph, message = laa_generate_curvature_lineplot(target_id, target_type)
        elif graph_type == "curvature_color":
            graph, message = laa_generate_curvature_color_map(analysis_config, target_id, target_type)
        elif graph_type == "angle_plot":
            graph, message = laa_generate_direction_lineplot(target_id, target_type)

    elif distance_value_dataset == "ICP":
        if graph_type == "curvature_plot":
            graph, message = laa_generate_curvature_lineplot(target_id, target_type)
        elif graph_type == "curvature_color":
            graph, message = laa_generate_curvature_color_map(analysis_config, target_id, target_type)
        elif graph_type == "angle_plot":
            graph, message = laa_generate_direction_lineplot(target_id, target_type)

    else:
        graph = None
        message = "Invalid distance value dataset"

    return graph, message
