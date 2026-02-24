import numpy as np
from scipy.spatial.distance import cosine, correlation
from scipy.integrate import simpson


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def cosine_similarity_distance(a, b):
    """Cosine similarity converted to distance (0 = identical, 2 = opposite)."""
    return cosine(a, b)  # scipy returns distance already


def correlation_distance(a, b):
    """1 - correlation coefficient."""
    return correlation(a, b)  # also a distance


def dtw_distance(a, b):
    """Minimal DTW implementation (slow, but works)."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1]   # match
            )
    return dtw[n, m]


def integral_difference(a, b):
    """Integral of absolute difference between curves."""
    diff = np.abs(a - b)
    return simpson(diff, dx=1.0)  # simple equal-step integration
