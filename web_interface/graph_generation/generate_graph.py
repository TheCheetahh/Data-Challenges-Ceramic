from analysis.calculation.ipc.icp import generate_icp_overlap_image
from web_interface.graph_generation.laa.laa_graphs import laa_generate_curvature_lineplot, \
    laa_generate_curvature_color_map, \
    laa_generate_direction_lineplot, laa_get_template_svg


def generate_graph(analysis_config, target_id, target_type, graph_type):
    """

    :param analysis_config:
    :param target_id: sample or template id
    :param target_type: sample or template
    :param graph_type: name of the graph to generate
    :return:
    """

    distance_value_dataset = analysis_config.get("distance_value_dataset")
    graph = None
    message = None

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
        elif graph_type == "get_template":
            graph, message = laa_get_template_svg(target_id)

    elif distance_value_dataset == "ICP":
        if graph_type == "curvature_plot":
            graph, message = laa_generate_curvature_lineplot(target_id, target_type)
        elif graph_type == "curvature_color":
            graph, message = laa_generate_curvature_color_map(analysis_config, target_id, target_type)
        elif graph_type == "angle_plot":
            graph, message = laa_generate_direction_lineplot(target_id, target_type)
        elif graph_type == "overlap_plot":
            graph = generate_icp_overlap_image(
                analysis_config.get("db_handler"),
                analysis_config.get("sample_id"),  # source sample stays the same
                target_id,  # target changes
                analysis_config
            )

    else:
        graph = None
        message = "Invalid distance value dataset"

    return graph, message
