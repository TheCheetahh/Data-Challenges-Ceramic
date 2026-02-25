"""Batch auto-cropping for reconstructed SVGs.

This automates the manual step from the "Edit SVG Path" tab:
Only SVGs that look "faulty" (start/end not near the visually lowest point)
get a cropped_svg written back to the DB.

Heuristic (default):
  - Only consider polyline/polygon SVGS
  - Treat as closed if polygon OR fill != none OR endpoints nearly coincide
  - Compute start/end y-percent relative to min/max y of the contour
  - If min(start%, end%) < threshold (default 90), apply crop_svg_path(...)

The cropping call can be a no-op crop (start=0, end=1) but still fixes the seam
because crop_svg_path normalizes the seam for closed contours.
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple, Optional
from xml.etree import ElementTree as ET

from database_handler import MongoDBHandler
from web_interface.formating_functions.format_svg import crop_svg_path, remove_svg_fill


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_points(points_str: str) -> List[Tuple[float, float]]:
    if not points_str:
        return []
    s = points_str.replace(",", " ")
    parts = [p for p in s.split() if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        nums = [float(x) for x in _FLOAT_RE.findall(s)]

    if len(nums) < 4 or len(nums) % 2 != 0:
        return []
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bbox_diag(points: List[Tuple[float, float]]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _is_closed_like(elem: Optional[ET.Element], points: List[Tuple[float, float]]) -> bool:
    if len(points) < 3:
        return False

    # polygon is closed by definition
    if elem is not None and elem.tag.lower().endswith("polygon"):
        return True

    # filled polyline/polygon often indicates intended closed contour
    fill = elem.get("fill") if elem is not None else None
    if fill is not None and fill.strip().lower() not in ("", "none"):
        return True

    diag = _bbox_diag(points)
    tol = max(1e-6, diag * 0.01)  # 1% of bbox diagonal
    return _dist(points[0], points[-1]) <= tol


def _start_end_y_percent(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return (start%, end%) where 0=min(y), 100=max(y)."""
    ys = [p[1] for p in points]
    y_min = min(ys)
    y_max = max(ys)
    denom = (y_max - y_min)
    if denom <= 1e-12:
        return 100.0, 100.0

    def pct(y: float) -> float:
        return (y - y_min) / denom * 100.0

    return pct(points[0][1]), pct(points[-1][1])


def _extract_points_from_svg(svg_string: str) -> Tuple[Optional[ET.Element], List[Tuple[float, float]]]:
    """Return (element, points) for polyline/polygon; (None, []) on failure."""
    try:
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        root = ET.fromstring(svg_string)
    except Exception:
        return None, []

    elem = root.find('.//{http://www.w3.org/2000/svg}polyline')
    if elem is None:
        elem = root.find('.//{http://www.w3.org/2000/svg}polygon')
    if elem is None:
        elem = root.find('.//polyline')
    if elem is None:
        elem = root.find('.//polygon')
    if elem is None:
        return None, []

    points = _parse_points(elem.get('points') or '')
    return elem, points


def click_auto_crop_faulty_svgs(
    y_threshold: float = 90.0,
    overwrite_existing: bool = False,
    seam_position: float = 50.0,
    crop_start: float = 0.0,
    crop_end: float = 1.0,
    dry_run: bool = False,
):
    """Scan all saved SVGs in DB and write `cropped_svg` only for faulty ones.

    Returns a human-readable status string for Gradio.
    """

    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    scanned = 0
    eligible_closed = 0
    cropped = 0
    skipped_ok = 0
    skipped_existing = 0
    parse_failed = 0
    updated_fail = 0

    # Keep small list for quick inspection in UI
    cropped_ids: List[str] = []

    cursor = db_handler.collection.find(
        {},
        {
            "sample_id": 1,
            "cleaned_svg": 1,
            "svg": 1,
            "cropped_svg": 1,
            "crop_start": 1,
            "crop_end": 1,
            "crop_seam_position": 1,
        },
    )

    for doc in cursor:
        scanned += 1
        sample_id = doc.get("sample_id")
        if sample_id is None:
            parse_failed += 1
            continue

        if (not overwrite_existing) and doc.get("cropped_svg"):
            skipped_existing += 1
            continue

        full_svg = doc.get("cleaned_svg") or doc.get("svg")
        if not full_svg:
            parse_failed += 1
            continue

        elem, points = _extract_points_from_svg(full_svg)
        if len(points) < 3 or elem is None:
            parse_failed += 1
            continue

        if not _is_closed_like(elem, points):
            # only fix seam for closed(-ish) contours
            skipped_ok += 1
            continue

        eligible_closed += 1

        start_pct, end_pct = _start_end_y_percent(points)
        needs_crop = min(start_pct, end_pct) < float(y_threshold)

        if not needs_crop:
            skipped_ok += 1
            continue

        # Apply seam normalization (and optionally trim if crop_start/end not default)
        try:
            cropped_svg = crop_svg_path(full_svg, float(crop_start), float(crop_end), float(seam_position))
            cropped_svg = remove_svg_fill(cropped_svg)
        except Exception:
            updated_fail += 1
            continue

        if dry_run:
            cropped += 1
            if len(cropped_ids) < 30:
                cropped_ids.append(str(sample_id))
            continue

        result = db_handler.collection.update_one(
            {"sample_id": sample_id},
            {"$set": {
                "cropped_svg": cropped_svg,
                "crop_start": float(crop_start),
                "crop_end": float(crop_end),
                "crop_seam_position": float(seam_position),
                "outdated_curvature": True,
                "icp_data": None,
            }},
        )

        if result.modified_count > 0:
            cropped += 1
            if len(cropped_ids) < 30:
                cropped_ids.append(str(sample_id))
        else:
            updated_fail += 1

    mode = "DRY-RUN" if dry_run else "WRITE"
    head = (
        f"Auto-crop finished ({mode}).\n"
        f"Scanned: {scanned}\n"
        f"Closed eligible: {eligible_closed}\n"
        f"Cropped: {cropped}\n"
        f"Skipped (already cropped): {skipped_existing}\n"
        f"Skipped (ok/not-eligible): {skipped_ok}\n"
        f"Parse failed: {parse_failed}\n"
        f"Update failed: {updated_fail}\n"
        f"Threshold: {float(y_threshold)} | seam_position: {float(seam_position)} | crop: {float(crop_start)}..{float(crop_end)}\n"
    )

    if cropped_ids:
        head += "\nExample cropped IDs (max 30):\n" + ", ".join(cropped_ids)

    return head
