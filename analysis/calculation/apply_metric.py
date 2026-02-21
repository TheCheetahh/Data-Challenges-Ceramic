from analysis.calculation.distance_methods import euclidean_distance, cosine_similarity_distance, correlation_distance, \
    dtw_distance, integral_difference

def apply_metric(a, b, distance_calculation):
    """

    :param a:
    :param b:
    :param distance_calculation:
    :return:
    """

    if distance_calculation == "Euclidean Distance":
        return euclidean_distance(a, b)

    elif distance_calculation == "Cosine Similarity":
        return cosine_similarity_distance(a, b)

    elif distance_calculation == "Correlation Distance":
        return correlation_distance(a, b)

    elif distance_calculation == "dynamic time warping":
        return dtw_distance(a, b)

    elif distance_calculation == "integral difference":
        return integral_difference(a, b)

    else:
        raise ValueError(f"Unknown distance_calculation: {distance_calculation}")