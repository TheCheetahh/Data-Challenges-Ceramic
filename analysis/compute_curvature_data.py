import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from matplotlib.colors import Normalize
from svgpathtools import svg2paths

from analysis.analyze_curvature import normalize_path, curvature_from_points
from database_handler import MongoDBHandler
from analysis.distance_methods import euclidean_distance, cosine_similarity_distance, correlation_distance, \
    dtw_distance, integral_difference


def compute_curvature_for_all_samples(smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
    """
    Compute curvature for ALL documents that have cleaned_svg (skip if already stored).
    For each document that does not have it call compute_and_store_curvature

    :param smooth_method: "savgol" or "gaussian" or "bspline
    :param smooth_factor: smoothing factor
    :param smooth_window: smoothing window
    :param n_samples: number of samples
    :return:
    """

    # create database handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # get all samples
    docs = db_handler.collection.find({}, {"sample_id": 1, "cleaned_svg": 1, "curvature_data": 1})

    # init counter vars
    processed = 0
    skipped = 0
    errors = 0

    # iterate over all samples
    for doc in docs:
        sample_id = doc.get("sample_id")
        if not sample_id:
            continue

        # load curvature data and smoothing settings of the doc
        stored_data = doc.get("curvature_data")
        stored_settings = stored_data.get("settings", {}) if stored_data else {}

        # if the loaded curvature data has the same settings as the current settings skip this doc
        if (
                stored_settings.get("smooth_method") == smooth_method and
                float(stored_settings.get("smooth_factor", 0)) == float(smooth_factor) and
                int(stored_settings.get("smooth_window", 0)) == int(smooth_window) and
                int(stored_settings.get("n_samples", 0)) == int(n_samples)
        ):
            skipped += 1
            continue

        # Compute and overwrite stored curvature data if necessary
        status = compute_curvature_for_one_sample(
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

    # returns status message
    return f"✅ Recomputed: {processed}, ⏭️ Skipped (same settings): {skipped}, ❌ Errors: {errors}"


def compute_curvature_for_one_sample(sample_id, smooth_method="savgol",
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

    # convert sample_id to int
    try:
        sample_id = int(sample_id)
    except ValueError:
        return f"❌ sample_id must be an integer."

    # create db handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # get the doc of the sample_id
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if doc is None:
        return f"❌ No sample found with sample_id: {sample_id}"
    if "cleaned_svg" not in doc:
        return f"❌ Field 'cleaned_svg' not found in document for sample_id: {sample_id}"

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

    # directions
    diffs = np.diff(points, axis=0)
    directions = np.arctan2(diffs[:, 1], diffs[:, 0])  # Angle of every segment to x-axis
    # Länge der Arrays angleichen
    directions = np.concatenate(([directions[0]], directions))

    # Store in DB
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {
            "curvature_data": {
                "arc_lengths": arc_lengths.tolist(),
                "curvature": curvature.tolist(),
                "directions": directions.tolist(),
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


def generate_all_plots(sample_id, smooth_method="savgol", smooth_factor=0.02, smooth_window=15, n_samples=2000):
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

    # create db handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # get document of the sample id
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "cleaned_svg" not in doc:
        return None, None, f"❌ No cleaned SVG found for sample_id {sample_id}"

    # figure out if data is already computed
    recompute = True
    if "curvature_data" in doc and "smoothing_settings" in doc["curvature_data"]:
        stored_settings = doc["curvature_data"]["smoothing_settings"]
        # Compare stored settings to requested
        if (stored_settings.get("method") == smooth_method and
                stored_settings.get("factor") == smooth_factor and
                stored_settings.get("window") == smooth_window and
                stored_settings.get("samples") == n_samples):
            recompute = False

    # compute the date with current settings
    if recompute:
        compute_curvature_for_one_sample(
            sample_id,
            smooth_method=smooth_method,
            smooth_factor=smooth_factor,
            smooth_window=smooth_window,
            n_samples=n_samples
        )
        # reload document of the sample id
        doc = db_handler.collection.find_one({"sample_id": sample_id})
        if not doc or "cleaned_svg" not in doc:
            return None, None, f"❌ No cleaned SVG found for sample_id {sample_id}"

    # get curvature data
    curvature_data = doc["curvature_data"]
    arc_lengths = np.array(curvature_data["arc_lengths"])
    curvature = np.array(curvature_data["curvature"])
    directions = np.array(curvature_data["directions"])
    status_msg = f"✅ Loaded stored curvature for sample_id {sample_id}"

    # Reconstruct points for color map
    svg_file_like = io.StringIO(doc["cleaned_svg"])
    paths, _ = svg2paths(svg_file_like)
    path = paths[0]
    n_samples = len(curvature)
    ts = np.linspace(0, 1, n_samples)
    points = np.array([path.point(t) for t in ts])
    points = np.column_stack((points.real, points.imag))

    stored_settings = curvature_data.get("smoothing_settings", {})
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


def find_enhanced_closest_curvature(sample_id, distance_dataset, distance_calculation, top_k=5):
    """
    calculate close samples and save to db. return the top result

    :param sample_id:
    :param distance_dataset:
    :param distance_calculation:
    :param top_k:
    :return:
    """

    # convert sample_id to int
    try:
        sample_id = int(sample_id)
    except:
        return None, None, f"Invalid sample_id {sample_id}"

    # create db handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # get the sample from db and its curvature and direction (angle)
    doc = db_handler.collection.find_one({"sample_id": sample_id})
    if not doc or "curvature_data" not in doc:
        return None, None, f"No curvature data for sample_id {sample_id}"

    curvature = np.array(doc["curvature_data"]["curvature"])
    direction = np.array(doc["curvature_data"]["directions"])

    # compute distances
    top_matches = calculate_distances(
        sample_id, curvature, direction, db_handler,
        distance_dataset, distance_calculation, top_k
    )

    if not top_matches:
        return None, None, "No comparable samples found."

    # save results
    db_handler.collection.update_one(
        {"sample_id": sample_id},
        {"$set": {"closest_matches": top_matches}}
    )

    closest = top_matches[0]
    msg = f"Closest sample to {sample_id} is {closest['id']} (distance={closest['distance']:.6f})"

    return closest["id"], closest["distance"], msg


def calculate_distances(sample_id, curvature, direction, db_handler, distance_dataset,
                        distance_calculation, top_k=5):
    """

    :param sample_id:
    :param curvature:
    :param direction:
    :param db_handler:
    :param distance_dataset:
    :param distance_calculation:
    :param top_k:
    :return:
    """

    # setup distances list
    distances = []

    # iterate all other samples
    for other_doc in db_handler.collection.find({"sample_id": {"$ne": sample_id}},
                                                {"sample_id": 1, "curvature_data": 1}):
        oid = other_doc["sample_id"]
        curv_data = other_doc.get("curvature_data")
        if not curv_data:
            continue

        # get other samples data
        other_curv = np.array(curv_data["curvature"])
        other_dir = np.array(curv_data["directions"])

        # get distance list from select_arrays (already calculated)
        dist = get_distance(
            curvature, other_curv,
            direction, other_dir,
            distance_dataset,
            distance_calculation
        )

        distances.append((oid, dist))

    # if no results
    if not distances:
        return []

    # sort + top-k results
    distances.sort(key=lambda x: x[1])
    top_matches = [{"id": sid, "distance": float(dist)} for sid, dist in distances[:top_k]]

    return top_matches


def get_distance(curvature, other_curvature, direction, other_direction, distance_dataset, distance_calculation):
    """

    :param curvature:
    :param other_curvature:
    :param direction:
    :param other_direction:
    :param distance_dataset:
    :param distance_calculation:
    :return:
    """

    # get amount of elements in the cropped 10%
    curve_len = len(curvature)
    crop = int(curve_len * 0.10)

    if curve_len <= 2 * crop:
        return None  # skip malformed samples

    # dataset selection
    if distance_dataset == "only curvature":
        return float(sum([apply_metric(curvature, other_curvature, distance_calculation)]))

    elif distance_dataset == "cropped curvature":
        return float(sum([apply_metric(curvature[crop:-crop], other_curvature[crop:-crop], distance_calculation)]))

    elif distance_dataset == "only angle":
        return float(sum([apply_metric(direction, other_direction, distance_calculation)]))

    elif distance_dataset == "cropped angle":
        return float(sum([apply_metric(direction[crop:-crop], other_direction[crop:-crop], distance_calculation)]))

    elif distance_dataset == "cropped curvature and angle":
        return float(sum([
            apply_metric(curvature[crop:-crop], other_curvature[crop:-crop], distance_calculation),
            apply_metric(direction[crop:-crop], other_direction[crop:-crop], distance_calculation)
        ]))

    # default fallback: full curvature + full angle
    return float(sum([
        apply_metric(curvature, other_curvature, distance_calculation),
        apply_metric(direction, other_direction, distance_calculation)
    ]))


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
