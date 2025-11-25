import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
from svgpathtools import svg2paths
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
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
    epsilon = 1e-10  # verhindert divide by zero
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + epsilon) ** 1.5
    curvature[np.isnan(curvature)] = 0

    # leichte Glättung für Stabilität
    curvature = np.convolve(curvature, np.ones(5)/5, mode='same')
    return curvature


def normalize_path(points, smooth_method, smooth_factor, smooth_window):
    """Pfad auf Startpunkt (0,0) verschieben und optional ausrichten."""
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
    """Liest den SVG-Pfad und tastet ihn gleichmäßig entlang der Länge ab (umgekehrte Richtung)."""
    paths, _ = svg2paths(svg_file)
    path = paths[0]
    ts = np.linspace(1, 0, n_samples)  # Laufrichtung umgekehrt
    pts = np.array([path.point(t) for t in ts])
    return np.column_stack((pts.real, pts.imag))


def compute_and_store_curvature_for_all(smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
    """
    Compute curvature for ALL documents that have cleaned_svg (skip if already stored).
    For each document that does not have it call compute_and_store_curvature

    :param smooth_method: "savgol" or "gaussian" or "bspline
    :param smooth_factor: smoothing factor
    :param smooth_window: smoothing window
    :param n_samples: number of samples
    :return:
    """

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    docs = db_handler.collection.find({}, {"sample_id": 1, "cleaned_svg": 1, "curvature_data": 1})

    processed = 0
    skipped = 0
    errors = 0

    # Settings to compare
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

        # if the already calculated plot has the same settings as the current settings skip this doc
        if (
                stored_settings.get("smooth_method") == smooth_method and
                float(stored_settings.get("smooth_factor", 0)) == float(smooth_factor) and
                int(stored_settings.get("smooth_window", 0)) == int(smooth_window) and
                int(stored_settings.get("n_samples", 0)) == int(n_samples)
        ):
            skipped += 1
            continue

        # Compute and overwrite stored curvature data if necessary
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


def compute_and_store_curvature(sample_id, smooth_method="savgol",
                                smooth_factor=0.02, smooth_window=15, n_samples=2000):
    """
    computes and stores curvature data for a single sample.

    :param sample_id:
    :param smooth_method:
    :param smooth_factor:
    :param smooth_window:
    :param n_samples:
    :return:
    """

    # sample_id from database must be int
    try:
        sample_id = int(sample_id)
    except ValueError:
        return f"❌ sample_id must be an integer."

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")
    doc = db_handler.collection.find_one({"sample_id": sample_id})

    # must have cleaned svg
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


def compute_or_load_curvature(sample_id, smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
    """
    Compute curvature if missing or settings changed, otherwise load from DB.

    :param sample_id:
    :param smooth_method:
    :param smooth_factor:
    :param smooth_window:
    :param n_samples:
    :return:
    """
    # Convert sample_id to int
    try:
        sample_id = int(sample_id)
    except ValueError:
        return None, None, "❌ sample_id must be an integer."

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # Fetch document
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "cleaned_svg" not in doc:
        return None, None, f"❌ No cleaned SVG found for sample_id {sample_id}"

    # Settings bundle
    current_settings = {
        "smooth_method": smooth_method,
        "smooth_factor": smooth_factor,
        "smooth_window": smooth_window,
        "n_samples": n_samples
    }

    recompute = True
    if "curvature_data" in doc and "smoothing_settings" in doc["curvature_data"]:
        stored_settings = doc["curvature_data"]["smoothing_settings"]
        # Compare stored settings to requested
        if (stored_settings.get("method") == smooth_method and
                stored_settings.get("factor") == smooth_factor and
                stored_settings.get("window") == smooth_window and
                stored_settings.get("samples") == n_samples):
            recompute = False

    if recompute:
        # --- Compute curvature ---
        cleaned_svg_content = doc["cleaned_svg"]
        svg_file_like = io.StringIO(cleaned_svg_content)
        paths, _ = svg2paths(svg_file_like)
        if len(paths) == 0:
            return None, None, f"❌ No paths found in SVG for sample_id {sample_id}"

        path = paths[0]
        ts = np.linspace(0, 1, n_samples)
        points = np.array([path.point(t) for t in ts])
        points = np.column_stack((points.real, points.imag))

        # Normalize & smooth
        points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

        # === Direction ===
        diffs = np.diff(points, axis=0)
        directions = np.arctan2(diffs[:, 1], diffs[:, 0])  # Angle of every segment to x-axis

        # Länge der Arrays angleichen
        directions = np.concatenate(([directions[0]], directions))

        # Curvature
        curvature = curvature_from_points(points)
        arc_lengths = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
        arc_lengths /= arc_lengths[-1]

        # Store in DB
        db_handler.store_curvature_in_db(
            sample_id,
            arc_lengths,
            curvature,
            smooth_method,
            smooth_factor,
            smooth_window,
            n_samples
        )
        status_msg = f"✅ Curvature computed and stored for sample_id {sample_id}"
    else:
        # --- Load curvature from DB ---
        data = doc["curvature_data"]
        arc_lengths = np.array(data["arc_lengths"])
        curvature = np.array(data["curvature"])
        status_msg = f"✅ Loaded stored curvature for sample_id {sample_id}"

        # Reconstruct points for color map
        svg_file_like = io.StringIO(doc["cleaned_svg"])
        paths, _ = svg2paths(svg_file_like)
        path = paths[0]
        n_samples = len(curvature)
        ts = np.linspace(0, 1, n_samples)
        points = np.array([path.point(t) for t in ts])
        points = np.column_stack((points.real, points.imag))

        stored_settings = data.get("smoothing_settings", {})
        smooth_method = stored_settings.get("method", smooth_method)
        smooth_factor = stored_settings.get("factor", smooth_factor)
        smooth_window = stored_settings.get("window", smooth_window)
        points = normalize_path(points, smooth_method, smooth_factor, smooth_window)

    # --- Generate 1D line plot ---
    curvature = -curvature  # einfach nur, weil positive Zahlen hübscher sind
    buf1 = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", linestyle="--")
    plt.plot(arc_lengths, curvature, color="black")
    plt.title(f"Curvature along normalized arc length (sample {sample_id})")
    plt.xlabel("Normalized arc length")
    plt.ylabel("Curvature κ")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf1, format="png")
    plt.close()
    buf1.seek(0)
    curvature_plot_img = Image.open(buf1)

    # --- Generate color map plot ---
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=-np.max(np.abs(curvature)), vmax=np.max(np.abs(curvature)) * 0.8)
    buf2 = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="coolwarm", norm=norm)
    lc.set_array(curvature)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.invert_yaxis()
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Curvature Color Map")
    plt.colorbar(lc, ax=ax, label="Curvature κ")
    plt.tight_layout()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    curvature_color_img = Image.open(buf2)

    # --- Generate direction plot ---
    directions = -directions  # einfach nur, weil positive Zahlen hübscher sind
    directions = np.unwrap(directions)
    buf3 = io.BytesIO()
    plt.figure(figsize=(10, 4))
    plt.plot(arc_lengths, directions, color="blue")
    plt.title(f"Direction along normalized arc length (sample {sample_id})")
    plt.xlabel("Normalised arc length")
    plt.ylabel("Angle to x-Axis [rad]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf3, format="png")
    plt.close()
    buf3.seek(0)
    angle_plot_img = Image.open(buf3)

    return curvature_plot_img, curvature_color_img, angle_plot_img, status_msg


def action_analyse_svg():
    """analyse svg. This function currently only cleans the svg"""
    db_handler = MongoDBHandler("svg_data")
    message = clean_all_svgs(db_handler)

    return message


def find_closest_curvature(sample_id):
    """
    Finds the sample whose curvature line is closest to the given sample_id.

    :param sample_id: sample id
    :return:
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
    distances = []  # list of (other_id, dist)
    cursor = db_handler.collection.find(
        {"sample_id": {"$ne": sample_id}},
        {"sample_id": 1, "curvature_data": 1}
    )

    for other_doc in cursor:
        oid = other_doc["sample_id"]
        data = other_doc.get("curvature_data")
        if not data:
            continue

        other_curve = np.array(data["curvature"])

        # Interpolate if necessary
        if len(other_curve) != len(target_curvature):
            other_curve = np.interp(
                np.linspace(0, 1, len(target_curvature)),
                np.linspace(0, 1, len(other_curve)),
                other_curve
            )

        dist = np.linalg.norm(target_curvature - other_curve)
        distances.append((oid, dist))

    if not distances:
        return None, None, "No comparable samples found."

    # Sort & take top-k (5)
    top_k = 5
    distances.sort(key=lambda x: x[1])
    top_matches = [{"id": sid, "distance": float(dist)} for sid, dist in distances[:top_k]]

    # Store into DB
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {"closest_matches": top_matches}}
    )

    # Return first (closest) for compatibility
    closest_id = top_matches[0]["id"]
    closest_dist = top_matches[0]["distance"]
    msg = f"Closest sample to {sample_id} is {closest_id} (distance={closest_dist:.6f})"

    return closest_id, closest_dist, msg