# canopy_geometry.py
#
# Purpose:
#   Apply per-leaf architectural parameters (size, curvature, twist, orientation,
#   and vertical attachment position) to a NURBS plant represented as a
#   geomdl.multi.SurfaceContainer:
#     - surf_cont[0] = stalk surface
#     - surf_cont[1:] = leaf surfaces
#
# Key assumptions (your current plant template):
#   - Each leaf surface is a flattened grid of control points with 6 points per row.
#   - Z is the vertical axis (stalk height / leaf attachment height).
#   - “Inter-leaf distances” are sampled as relative values in [0,1] and then
#     mapped onto an available vertical band along the stalk that depends on
#     stalk height (so attachment placement scales with stalk height).
#
# IMPORTANT FIX INCLUDED:
#   theta/phi mismatch corrected in change_leaf_parameters():
#     - thetas = inclination (rotate about Y)
#     - phis   = azimuth (rotate about Z)
#   apply_leaf_orientation() expects (z_angle_deg, y_angle_deg), so we now pass:
#     apply_leaf_orientation(surface, phis[i], thetas[i])

import math
from typing import Iterable, List, Sequence, Optional

import numpy as np
from geomdl import BSpline, operations, utilities
from geomdl.operations import rotate

# Each “row” in your flattened ctrlpt grid has 6 points
PTS_PER_ROW = 6


# ----------------------------
# Stalk scaling (height in Z)
# ----------------------------

def scale_stalk_length(surface, scale_factor: float) -> None:
    """
    Scale stalk height (Z) around its base (minimum Z).

    Parameters
    ----------
    surface : geomdl surface
        Stalk surface (surf_cont[0]).
    scale_factor : float
        Multiplicative factor for stalk height (e.g., 0.7–1.3).
    """
    ctrlpts = surface.ctrlpts
    if not ctrlpts:
        return

    min_z = min(pt[2] for pt in ctrlpts)

    # Scale every point's Z above the base
    for pt in ctrlpts:
        if pt[2] != min_z:
            pt[2] = min_z + (pt[2] - min_z) * scale_factor

    surface.ctrlpts = ctrlpts


# -----------------------------------
# Leaf vertical positioning on stalk
# -----------------------------------

def apply_inter_leaf_distances(surf_cont, inter_leaf_distances: np.ndarray) -> None:
    """
    Position each leaf along the stalk using relative values in [0,1].

    This version enforces a biologically reasonable attachment band:
      [min_leaf_offset, stalk_height - top_margin]

    and makes attachment positions dependent on the (possibly scaled) stalk height.

    Also, leaves are ordered bottom-to-top by sorting the samples.

    Parameters
    ----------
    surf_cont : geomdl.multi.SurfaceContainer
        Container with stalk at index 0 and leaves at 1:.
    inter_leaf_distances : np.ndarray
        Relative positions in [0,1], one value per leaf.
    """
    stalk_surface = surf_cont[0]
    stalk_pts = stalk_surface.ctrlpts
    if not stalk_pts:
        return

    min_z = min(pt[2] for pt in stalk_pts)
    max_z = max(pt[2] for pt in stalk_pts)
    stalk_height = max_z - min_z

    # ---- Attachment band along stalk (your original logic) ----
    # Leaves will attach no lower than min_leaf_offset above base
    # and no higher than (stalk_height - top_margin) below the top.
    min_leaf_offset = 13.0
    top_margin = 15.0
    max_leaf_offset = stalk_height - top_margin

    # Guard against pathological stalk heights
    if max_leaf_offset <= min_leaf_offset:
        # If stalk is too short to support the margins, fall back to full span
        min_leaf_offset = 0.0
        max_leaf_offset = stalk_height

    available_z_span = max_leaf_offset - min_leaf_offset

    # Sort to enforce bottom-to-top order
    sorted_samples = np.sort(np.asarray(inter_leaf_distances, dtype=float))

    # Map samples [0,1] into [min_leaf_offset, max_leaf_offset] and shift by min_z
    attachment_z = (sorted_samples * available_z_span) + min_leaf_offset + min_z

    # Apply one z_target per leaf (supports any number of leaves via surf_cont[1:])
    # Anchor definition: ctrlpts[6] is treated as the leaf attachment reference point.
    for z_target, leaf_surface in zip(attachment_z, surf_cont[1:]):
        ctrlpts = leaf_surface.ctrlpts
        if not ctrlpts:
            continue

        z_anchor = ctrlpts[6][2]  # anchor control point (index 6)
        z_shift = z_target - z_anchor

        for pt in ctrlpts:
            pt[2] += z_shift

        leaf_surface.ctrlpts = ctrlpts


# ----------------------------
# Curve utilities (length/width)
# ----------------------------

def calculate_curve_length(ctrlpts: Sequence[Sequence[float]]) -> float:
    """
    Approximate arc length of a 3D B-spline curve defined by control points.
    Used here as a 'leaf length' proxy.

    Parameters
    ----------
    ctrlpts : sequence of [x, y, z]
        Control points.

    Returns
    -------
    float
        Curve length.
    """
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = [list(p) for p in ctrlpts]
    curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)
    curve.delta = 0.01
    return operations.length_curve(curve)


def calculate_curve_width(ctrlpts: Sequence[Sequence[float]]) -> float:
    """
    Approximate arc length of a curve used as a 'leaf width' proxy.

    Parameters
    ----------
    ctrlpts : sequence of [x, y, z]
        Control points.

    Returns
    -------
    float
        Curve length.
    """
    curve = BSpline.Curve()
    curve.degree = 2
    curve.ctrlpts = [list(p) for p in ctrlpts]
    curve.knotvector = utilities.generate_knot_vector(curve.degree, curve.ctrlpts_size)
    curve.delta = 0.01
    return operations.length_curve(curve)


# ----------------------------
# Leaf scaling helpers
# ----------------------------

def scale_length(points, multiplier, exclude_indices=(0, 6, 12)):
    """
    Scale leaf length by scaling X coordinate of control points.
    Some indices are excluded to preserve a backbone / reference points.

    Parameters
    ----------
    points : list of ctrlpts
    multiplier : float
    exclude_indices : tuple[int]

    Returns
    -------
    points : list (mutated)
    """
    for idx, pt in enumerate(points):
        if idx not in exclude_indices:
            pt[0] *= multiplier
    return points


def scale_width(points, multiplier, exclude_indices=(1,)):
    """
    Scale leaf width by scaling Y coordinate of control points.
    Some indices are excluded to preserve a reference column.

    Parameters
    ----------
    points : list of ctrlpts (typically a column across rows)
    multiplier : float
    exclude_indices : tuple[int]

    Returns
    -------
    points : list (mutated)
    """
    for idx, pt in enumerate(points):
        if idx not in exclude_indices:
            pt[1] *= multiplier
    return points


def apply_leaf_size(surface, target_length: float, target_width: float) -> None:
    """
    Resize a leaf to target length and width.

    Length reference:
      - Uses ctrlpts[6:12] (your original mid-span curve).

    Width reference:
      - Builds rows of 6 points, and uses the 3rd point in each row (index 2)
        to define a width curve across rows.

    Parameters
    ----------
    surface : geomdl surface
        Leaf surface
    target_length : float
    target_width : float
    """
    ctrlpts = surface.ctrlpts
    if not ctrlpts:
        return

    # ---- Length scaling (X) ----
    current_len = calculate_curve_length(ctrlpts[6:12])
    if current_len > 0:
        scale_length(ctrlpts, target_length / current_len)

    # ---- Width scaling (Y) ----
    v_dir_sets = [ctrlpts[i:i + PTS_PER_ROW] for i in range(0, len(ctrlpts), PTS_PER_ROW)]

    # “Width curve” uses the 3rd control point in each row (index 2)
    current_width = calculate_curve_width([v_dir_set[2] for v_dir_set in v_dir_sets])
    if current_width > 0:
        width_factor = target_width / current_width

        # Scale internal columns j=1..4 (keep edges j=0 and j=5 unchanged)
        for j in range(1, PTS_PER_ROW - 1):
            col_pts = [v_dir_set[j] for v_dir_set in v_dir_sets]
            scaled = scale_width(col_pts, width_factor)
            for k, row in enumerate(v_dir_sets):
                row[j] = scaled[k]

    surface.ctrlpts = [pt for row in v_dir_sets for pt in row]


# ----------------------------
# Leaf curvature / twist
# ----------------------------

def apply_leaf_curvature(surface, curvature: float) -> None:
    """
    Apply curvature by warping Z across leaf width with a smooth sinusoidal profile.

    Parameters
    ----------
    surface : geomdl surface
    curvature : float
        Signed amplitude. Positive arches up; negative arches down.
    """
    ctrlpts = surface.ctrlpts
    if not ctrlpts or abs(curvature) < 1e-9:
        return

    v_dir_sets = [ctrlpts[i:i + PTS_PER_ROW] for i in range(0, len(ctrlpts), PTS_PER_ROW)]

    # Apply across internal width columns j=1..4
    for j in range(1, PTS_PER_ROW - 1):
        arch = curvature * math.sin(math.pi * j / (PTS_PER_ROW - 1))
        for v_dir_set in v_dir_sets:
            v_dir_set[j][2] += arch

    surface.ctrlpts = [pt for row in v_dir_sets for pt in row]


def apply_leaf_twist(surface, twist_value: float) -> None:
    """
    Apply a twist-like deformation (implemented as a Z-warp across width).

    NOTE:
      This is not a true 3D torsional rotation; it is a smooth Z warp whose sign
      controls direction (negative/positive).

    Parameters
    ----------
    surface : geomdl surface
    twist_value : float
        Signed amplitude.
    """
    ctrlpts = surface.ctrlpts
    if not ctrlpts or abs(twist_value) < 1e-6:
        return

    v_dir_sets = [ctrlpts[i:i + PTS_PER_ROW] for i in range(0, len(ctrlpts), PTS_PER_ROW)]

    for j in range(1, PTS_PER_ROW - 1):
        factor = math.sin(math.pi * j / (PTS_PER_ROW - 1))
        for v_dir_set in v_dir_sets:
            v_dir_set[j][2] += twist_value * factor

    surface.ctrlpts = [pt for row in v_dir_sets for pt in row]


# ----------------------------
# Leaf orientation (rotations)
# ----------------------------

def apply_leaf_orientation(surface, z_angle_deg: float, y_angle_deg: float) -> None:
    """
    Rotate leaf surface.

    Rotation order:
      1) Rotate about Y axis (inclination / tilt)
      2) Rotate about Z axis (azimuth around stalk)

    Parameters
    ----------
    surface : geomdl surface
    z_angle_deg : float
        Azimuth rotation about Z (degrees).
    y_angle_deg : float
        Inclination rotation about Y (degrees).
    """
    rotate(surface, angle=y_angle_deg, axis=1, inplace=True)  # Y tilt
    rotate(surface, angle=z_angle_deg, axis=2, inplace=True)  # Z azimuth


# ----------------------------
# Main entry: apply parameters
# ----------------------------

def change_leaf_parameters(surf_cont, **kwargs) -> None:
    """
    Apply all parameters to the NURBS plant container.

    Expected kwargs keys (arrays should be length = num_leaves):
      - stalk_scale: float
      - inter_leaf_distances: array-like in [0,1]
      - lengths: array-like
      - widths: array-like
      - curvatures: array-like
      - twists: array-like
      - thetas: array-like   (inclination / tilt about Y)
      - phis: array-like     (azimuth about Z)
    """
    # num_leaves = len(surf_cont) - 1  # available if you need checks

    # 1) Stalk scaling first (so stalk_height used below reflects the scaled stalk)
    if kwargs.get("stalk_scale") is not None:
        scale_stalk_length(surf_cont[0], float(kwargs["stalk_scale"]))

    # 2) Leaf vertical placement next (depends on scaled stalk height)
    if kwargs.get("inter_leaf_distances") is not None:
        apply_inter_leaf_distances(surf_cont, np.asarray(kwargs["inter_leaf_distances"], dtype=float))

    # 3) Per-leaf trait changes
    for i, surface in enumerate(surf_cont[1:]):

        if kwargs.get("lengths") is not None and kwargs.get("widths") is not None:
            apply_leaf_size(surface, float(kwargs["lengths"][i]), float(kwargs["widths"][i]))

        if kwargs.get("curvatures") is not None:
            apply_leaf_curvature(surface, float(kwargs["curvatures"][i]))

        if kwargs.get("twists") is not None:
            apply_leaf_twist(surface, float(kwargs["twists"][i]))

        if kwargs.get("thetas") is not None and kwargs.get("phis") is not None:
            # IMPORTANT FIX:
            #   thetas = inclination (Y rotation)
            #   phis   = azimuth     (Z rotation)
            #
            # apply_leaf_orientation expects (z_angle_deg, y_angle_deg), so:
            apply_leaf_orientation(
                surface,
                float(kwargs["phis"][i]),    # z_angle_deg (azimuth)
                float(kwargs["thetas"][i]),  # y_angle_deg (inclination)
            )
