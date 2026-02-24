import numpy as np
from scipy.signal import savgol_filter
from svgpathtools import svg2paths
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev


def smooth_path_savgol(points, window_length, smooth_factor):
    """
    Glättet den Pfad mit einem Savitzky-Golay-Filter.
    window_length: muss ungerade und kleiner als Anzahl der Punkte sein.
    polyorder: Grad des Polynoms (typisch 2 oder 3)
    """
    polyorder = int(smooth_factor)
    if window_length >= len(points):
        window_length = len(points) - (1 - len(points) % 2)  # ungerade machen
    smoothed_x = savgol_filter(points[:, 0], window_length, polyorder)
    smoothed_y = savgol_filter(points[:, 1], window_length, polyorder)
    return np.column_stack((smoothed_x, smoothed_y))


def smooth_path_bspline(points, s):
    tck, _ = splprep(points.T, s=s)
    u_new = np.linspace(0, 1, len(points))
    x_new, y_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new))


def smooth_path_gaussian(points, sigma):
    x_smooth = gaussian_filter1d(points[:, 0], sigma)
    y_smooth = gaussian_filter1d(points[:, 1], sigma)
    return np.column_stack((x_smooth, y_smooth))


def curvature_from_points(points):
    """Berechnet die *signierte* Krümmung entlang diskreter Punkte."""
    dx = np.gradient(points[:, 0], edge_order=2)
    dy = np.gradient(points[:, 1], edge_order=2)
    ddx = np.gradient(dx, edge_order=2)
    ddy = np.gradient(dy, edge_order=2)

    # signierte Krümmung (Vorzeichen abhängig von Richtung)
    epsilon = 1e-10  # verhindert divide by zero
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + epsilon) ** 1.5
    curvature[np.isnan(curvature)] = 0

    # leichte Glättung für Stabilität
    curvature = np.convolve(curvature, np.ones(5)/5, mode='same')
    return curvature

def ensure_left_to_right(points):
    x0, y0 = points[0]
    x1, y1 = points[-1]
    if (y0 - y1) <= (x0 - x1):  # Steigung 1: y-Differenz <= x-Differenz
        return points[::-1]
    else:
        return points

def normalize_path(points, smooth_method, smooth_factor, smooth_window):
    """Pfad auf Startpunkt (0,0) verschieben und optional ausrichten."""
    points = ensure_left_to_right(points)
    points = points - points[0]

    '''
    align_orientation=False
    if align_orientation:

        dx, dy = points[1] - points[0]
        angle = -np.arctan2(dy, dx)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),  np.cos(angle)]])
        points = (rot_matrix @ points.T).T

        # --- Automatische Orientierungskorrektur ---

        # 1. Start- und Endpunkte
        p_start, p_end = points[0], points[-1]

        # 2. Winkel zwischen Start->Ende und der x-Achse
        angle = np.arctan2(p_end[1] - p_start[1], p_end[0] - p_start[0])

        # 3. Rotationsmatrix, um diese Linie horizontal zu machen
        rotation_matrix = np.array([
            [ np.cos(-angle), -np.sin(-angle)],
            [ np.sin(-angle),  np.cos(-angle)]
        ])

        # 4. Alle Punkte um den Startpunkt rotieren
        points = (points - p_start) @ rotation_matrix.T'''

    # === Glättungsoptionen ===
    SMOOTHING_METHOD = smooth_method   # "savgol", "gaussian", "bspline" oder "none"

    if smooth_method != "none":
        if smooth_method == "savgol":
            points = smooth_path_savgol(points, smooth_window, smooth_factor)
        elif smooth_method == "gaussian":
            points = smooth_path_gaussian(points, smooth_factor)
        elif smooth_method == "bspline":
            points = smooth_path_bspline(points, smooth_factor)
    return points


def sample_svg_path(svg_file, n_samples=1000):
    """Liest den SVG-Pfad und tastet ihn gleichmäßig entlang der Richtung ab,
    in der die Punkte in der Datei gespeichert sind.
    ensure_left_to_right benötigt, weil diese Reihenfolge random sein kann"""
    paths, _ = svg2paths(svg_file)
    path = paths[0]
    ts = np.linspace(1, 0, n_samples)  # Laufrichtung umgekehrt
    pts = np.array([path.point(t) for t in ts])
    return np.column_stack((pts.real, pts.imag))
