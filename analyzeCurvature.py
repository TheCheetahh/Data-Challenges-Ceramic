import io
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
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
    epsilon = 1e-10 # verhindert divide by zero
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + epsilon) ** 1.5
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

"""
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
"""


def compute_and_store_curvature_for_all(smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
    """
    Compute curvature for ALL documents that have cleaned_svg (skip if already stored).
    """
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    docs = db_handler.collection.find({}, {"sample_id": 1, "cleaned_svg": 1, "curvature_data": 1})

    processed = 0
    skipped = 0
    errors = 0

    # Settings bundle to compare
    current_settings = {
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples
    }

    for doc in docs:
        sample_id = doc.get("sample_id")
        if not sample_id:
            continue

        stored_data = doc.get("curvature_data")
        stored_settings = stored_data.get("settings", {}) if stored_data else {}

        if (
                stored_settings.get("smooth_method") == smooth_method and
                float(stored_settings.get("smooth_factor", 0)) == float(smooth_factor) and
                int(stored_settings.get("smooth_window", 0)) == int(smooth_window) and
                int(stored_settings.get("n_samples", 0)) == int(n_samples)
        ):
            skipped += 1
            continue

        # --- Compute and overwrite stored curvature data ---
        status = compute_and_store_curvature(
            sample_id,
            smooth_method=smooth_method,
            smooth_factor=smooth_factor,
            smooth_window=smooth_window,
            n_samples=n_samples
        )

        if status.startswith("❌"):
            errors += 1
        else:
            processed += 1

    return f"✅ Recomputed: {processed}, ⏭️ Skipped (same settings): {skipped}, ❌ Errors: {errors}"


def compute_and_store_curvature(sample_id, smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
    try:
        sample_id = int(sample_id)
    except ValueError:
        return f"❌ sample_id must be an integer."

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")
    doc = db_handler.collection.find_one({"sample_id": sample_id})

    if not doc or "cleaned_svg" not in doc:
        return f"❌ No cleaned SVG found for sample_id {sample_id}"

    # Parse SVG path
    svg_file_like = io.StringIO(doc["cleaned_svg"])
    paths, _ = svg2paths(svg_file_like)
    if len(paths) == 0:
        return f"❌ No valid path found in SVG."

    # Sample points
    path = paths[0]
    ts = np.linspace(0, 1, n_samples)
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    # Normalize & Smooth
    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    # Compute curvature
    curvature = curvature_from_points(points)
    arc_lengths = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    arc_lengths /= arc_lengths[-1]

    # Store in DB
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {
            "curvature_data": {
                "arc_lengths": arc_lengths.tolist(),
                "curvature": curvature.tolist(),
                "settings": {
                    "smooth_method": smooth_method,
                    "smooth_factor": smooth_factor,
                    "smooth_window": smooth_window,
                    "n_samples": n_samples
                }
            }
        }}
    )

    return f"✅ Curvature computation stored for sample_id {sample_id}"


def load_and_plot_curvature(sample_id):
    try:
        sample_id = int(sample_id)
    except ValueError:
        return None, None, "❌ sample_id must be an integer."

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")
    doc = db_handler.collection.find_one({"sample_id": sample_id})

    if not doc or "curvature_data" not in doc:
        return None, None, f"❌ No stored curvature data for sample_id {sample_id}. Run computation first."

    data = doc["curvature_data"]
    arc_lengths = np.array(data["arc_lengths"])
    curvature = np.array(data["curvature"])

    # --- Line plot ---
    buf1 = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", linestyle="--")
    plt.plot(arc_lengths, curvature, color="black")
    plt.title(f"Curvature along arc length (sample {sample_id})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Curvature κ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf1, format="png")
    plt.close()
    buf1.seek(0)
    curvature_plot_img = Image.open(buf1)

    # --- Color map plot ---
    # You must reload the path to reconstruct points for color plotting
    svg_file_like = io.StringIO(doc["cleaned_svg"])
    paths, _ = svg2paths(svg_file_like)
    path = paths[0]
    n_samples = len(curvature)
    ts = np.linspace(0, 1, n_samples)
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    settings = data["settings"]

    smooth_method = settings.get("smooth_method") or settings.get("method")
    smooth_factor = settings.get("smooth_factor", 0.02)
    smooth_window = settings.get("smooth_window", 15)

    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=-np.max(np.abs(curvature)), vmax=np.max(np.abs(curvature)) * 0.8)

    buf2 = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    lc.set_array(curvature)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.autoscale()
    ax.set_title("Curvature Color Map")
    plt.colorbar(lc, ax=ax, label="Curvature κ")
    plt.tight_layout()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    curvature_color_img = Image.open(buf2)

    return curvature_plot_img, curvature_color_img, f"✅ Displaying stored curvature for sample_id {sample_id}"


def analyze_svg_curvature(sample_id, smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
    """
    Analyze curvature directly from the cleaned SVG stored in the database.
    Returns paths to the generated curvature plots.
    """
    # convert sample_id to int
    try:
        sample_id = int(sample_id)
    except ValueError:
        return None, "❌ sample_id must be a number."

    # Get cleaned SVG from DB
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "cleaned_svg" not in doc:
        return None, None, f"No cleaned SVG found for sample_id {sample_id}"

    cleaned_svg_content = doc["cleaned_svg"]

    # Load paths from SVG content using BytesIO
    svg_file_like = io.StringIO(cleaned_svg_content)
    paths, _ = svg2paths(svg_file_like)
    if len(paths) == 0:
        return None, None, f"No paths found in SVG for sample_id {sample_id}"

    path = paths[0]
    ts = np.linspace(0, 1, n_samples)
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    # Normalize & smooth
    points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    # Curvature
    curvature = curvature_from_points(points)
    arc_lengths = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    arc_lengths /= arc_lengths[-1]

    # store 1D line in DB
    db_handler.store_curvature_in_db(
        sample_id,
        arc_lengths,
        curvature,
        smooth_method,
        smooth_factor,
        smooth_window,
        n_samples
    )

    # --- Generate plots ---
    # Curvature line plot (1D plot)
    buf1 = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", linestyle="--")
    plt.plot(arc_lengths, curvature, color="black")
    plt.title(f"Curvature along normalized arc length ({smooth_method})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Curvature κ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf1, format="png")
    plt.close()
    buf1.seek(0)
    curvature_plot_img = Image.open(buf1)

    # Curvature color map (2D color map)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=-np.max(np.abs(curvature)), vmax=np.max(np.abs(curvature)) * 0.8)
    buf2 = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    lc.set_array(curvature)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Curvature Color Map")
    plt.colorbar(lc, ax=ax, label="Curvature κ")
    plt.tight_layout()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    curvature_color_img = Image.open(buf2)

    print("✅ Analyse abgeschlossen!")
    return curvature_plot_img, curvature_color_img, f"Analysis completed for sample_id {sample_id}"


def analyse_svg():
    """analyse svg. This function currently only cleans the svg"""
    db_handler = MongoDBHandler("svg_data")
    message = clean_all_svgs(db_handler)

    return message


def find_closest_curvature(sample_id):
    """
    Finds the sample whose curvature line is closest to the given sample_id.
    Returns: closest_sample_id, distance
    """

    # Convert sample_id to int
    try:
        sample_id = int(sample_id)
    except:
        return None, None, f"Invalid sample_id {sample_id}"

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Get curvature of the target sample
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "curvature_data" not in doc:
        return None, None, f"No curvature data for sample_id {sample_id}"

    target_curvature = np.array(doc["curvature_data"]["curvature"])

    # Iterate over all other documents
    min_distance = float("inf")
    closest_id = None

    for other_doc in db_handler.collection.find({"sample_id": {"$ne": sample_id}},
                                                {"sample_id": 1, "curvature_data": 1}):
        other_id = other_doc["sample_id"]
        if "curvature_data" not in other_doc:
            continue
        other_curve = np.array(other_doc["curvature_data"]["curvature"])

        # If lengths differ, interpolate to match
        if len(other_curve) != len(target_curvature):
            other_curve = np.interp(
                np.linspace(0, 1, len(target_curvature)),
                np.linspace(0, 1, len(other_curve)),
                other_curve
            )

        # Compute Euclidean distance
        dist = np.linalg.norm(target_curvature - other_curve)

        if dist < min_distance:
            min_distance = dist
            closest_id = other_id

    if closest_id is None:
        return None, None, "No other samples with curvature data found."

    return closest_id, min_distance, f"Closest sample to {sample_id} is {closest_id} with distance {min_distance:.4f}"


"""
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
"""