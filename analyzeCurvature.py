import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from svgpathtools import svg2paths
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import argparse
from cleanSVG import clean_all_svgs
from database_handler import MongoDBHandler


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
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature[np.isnan(curvature)] = 0

    # leichte Glättung für Stabilität
    curvature = np.convolve(curvature, np.ones(5)/5, mode='same')
    return curvature


def normalize_path(points, smooth_method, smooth_factor, smooth_window):
    """Pfad auf Startpunkt (0,0) verschieben und optional ausrichten."""
    points = points - points[0]

    align_orientation=True
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
        points = (points - p_start) @ rotation_matrix.T


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
    """Liest den SVG-Pfad und tastet ihn gleichmäßig entlang der Länge ab (umgekehrte Richtung)."""
    paths, _ = svg2paths(svg_file)
    path = paths[0]
    ts = np.linspace(1, 0, n_samples)  # Laufrichtung umgekehrt
    pts = np.array([path.point(t) for t in ts])
    return np.column_stack((pts.real, pts.imag))





def analyze_svg_curvature(svg_file, output_dir, smooth_method, smooth_factor, smooth_window, n_samples=2000):
    base_name = os.path.splitext(os.path.basename(svg_file))[0]
    base_dir = os.path.dirname(svg_file)

    # Punkte und Krümmung berechnen
    points = sample_svg_path(svg_file, n_samples)
    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)
    curvature = curvature_from_points(points)

    # Optional: letzte (und erste) Punkte ausschließen, z. B. 2 % an beiden Enden
    trim = int(0.02 * len(curvature))  # 2% der Punkte
    if trim > 0:
        points = points[trim:-trim]
        curvature = curvature[trim:-trim]

    # Bogenlänge berechnen
    diffs = np.diff(points, axis=0)
    arc_lengths = np.concatenate(([0], np.cumsum(np.sqrt((diffs**2).sum(axis=1)))))
    arc_lengths /= arc_lengths[-1]


    # Plot 1: Signierte Krümmung entlang der normierten Bogenlänge
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")  # Nulllinie
    plt.plot(arc_lengths, curvature, color="black")
    plt.title("Signierte Krümmung entlang der normierten Bogenlänge")
    plt.xlabel("Normierte Bogenlänge s/s_max")
    plt.ylabel("Signierte Krümmung κ (rechts = +, links = -)")
    plt.grid(True)

    curvature_plot_path = os.path.join(output_dir, f"{base_name}_curvature_plot_{smooth_method}_{smooth_factor}.png")
    plt.savefig(curvature_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


    # Plot 2: Farbige Linienvisualisierung (mit Vorzeichen)
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)

    # symmetrische Skalierung um 0
    v = np.max(np.abs(curvature))
    norm = Normalize(vmin=-v, vmax=v)

    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)  # Farbschema für ±
    lc.set_array(curvature)
    lc.set_linewidth(1.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_title("Krümmung als Farbwert (rechts = +, links = -)")

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Signierte Krümmung κ")

    curvature_color_path = os.path.join(output_dir, f"{base_name}_curvature_map_{smooth_method}_{smooth_factor}.png")
    plt.savefig(curvature_color_path, dpi=300, bbox_inches="tight")
    plt.close()


    print("✅ Analyse abgeschlossen!")
    print(f"📊 Krümmungsdiagramm gespeichert unter: {curvature_plot_path}")
    print(f"🎨 Farbkarte gespeichert unter:       {curvature_color_path}")


def analyse_svg():
    db_handler = MongoDBHandler("svg_data")
    message = clean_all_svgs(db_handler)

    return message


def main():
    parser = argparse.ArgumentParser(description="Analyse eines SVG-Pfads (Krümmung etc.)")
    parser.add_argument("--input_svg", type=str, required=True, help="Pfad zur Eingabe-SVG-Datei")
    parser.add_argument("--output_dir", type=str, default="results", help="Ordner für Analyseergebnisse")
    parser.add_argument("--smooth_method", default="savgol", help="savgol, gaussian, bspline oder none")
    parser.add_argument("--smooth_factor", type=float, default=3, help="Bei Savgol: Polyorder (zB. 3), bei Gaussian: Sigma (zB. 2), bei Bspline: s (zB. 0.5)")
    parser.add_argument("--smooth_window", type=int, default=51, help="Wird nur bei Savgol gebraucht, muss ungerade sein")
    parser.add_argument("--samples", type=int, default=2000, help="Anzahl der Abtastpunkte entlang des Pfads")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    analyze_svg_curvature(
        args.input_svg,
        args.output_dir,
        args.smooth_method,
        args.smooth_factor,
        args.smooth_window,
        args.samples
    )


if __name__ == "__main__":
    main()