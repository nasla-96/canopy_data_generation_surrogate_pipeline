# canopy_from_sobol.py

import os
from typing import Union, Optional, List
import numpy as np
from geomdl import exchange, multi

from canopy_geometry import change_leaf_parameters

# Extra imports for building a full canopy field from a single plant OBJ
import open3d as o3d
import copy
import csv
import random


def map_to_range(x: Union[float, np.ndarray],
                 lo: float,
                 hi: float) -> np.ndarray:
    """
    Map values from [0, 1] → [lo, hi] elementwise.

    Works for scalars or numpy arrays; always returns a numpy array.
    """
    arr = np.asarray(x, dtype=float)
    return lo + arr * (hi - lo)


def create_field(file_path: str,
                 row_spacing: float,
                 plant_spacing: float,
                 rotation_csv_path: Optional[str] = None) -> str:
    """
    Given a *single-plant* OBJ file, build a small canopy/field by tiling,
    randomly rotating each plant a bit, and cropping the central region.

    Parameters
    ----------
    file_path : str
        Path to the single-plant OBJ file (will be overwritten with a
        rescaled, recentered version).
    row_spacing : float
        Distance between rows in meters (y-direction).
    plant_spacing : float
        Distance between plants in a row in meters (x-direction).
    rotation_csv_path : str or None
        If provided, append one row per call with the file_path and
        the random rotation angle for each plant instance.

    Returns
    -------
    str
        Path to the new, cropped *canopy* OBJ file. The name is based on
        ``file_path`` with ``".obj"`` replaced by ``"_field_cropped.obj"``.
    """
    # Read the single-plant mesh
    mesh = o3d.io.read_triangle_mesh(file_path)

    # Scale the mesh to meters (assuming current units are centimeters)
    scale_factor = 0.01
    mesh.scale(scale_factor, center=mesh.get_center())

    # Translate the original mesh so its center is at the origin
    mesh.translate(-mesh.get_center())

    # Overwrite the original file with the rescaled, recentered single plant
    o3d.io.write_triangle_mesh(file_path, mesh)

    field = o3d.geometry.TriangleMesh()
    rotation_angles: List[float] = []

    # Build a 6x6 grid of plants centered around the origin
    for i in range(-3, 3):
        for j in range(-3, 3):
            cloned_mesh = copy.deepcopy(mesh)

            # Apply a small random rotation around the Z axis (in degrees)
            rotation_angle = random.uniform(-10.0, 10.0)
            rotation_radians = np.radians(rotation_angle)
            rotation_matrix = cloned_mesh.get_rotation_matrix_from_axis_angle(
                [0.0, 0.0, rotation_radians]
            )
            cloned_mesh.rotate(rotation_matrix, center=(0.0, 0.0, 0.0))

            # Translate this plant to its grid position
            cloned_mesh.translate(
                np.array([i * plant_spacing, j * row_spacing, 0.0]),
                relative=True,
            )

            field += cloned_mesh
            rotation_angles.append(rotation_angle)

    # Slight overall shift (so the cropped region is nicely centered)
    field.translate((-0.5 * plant_spacing, 0.5 * row_spacing, 0.0))

    # Crop a central region of the field (roughly 4x4 plants)
    cropped_field = field.crop(
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-2.0 * plant_spacing, -2.0 * row_spacing, -np.inf),
            max_bound=(2.0 * plant_spacing, 2.0 * row_spacing, np.inf),
        )
    )

    # Optionally log the rotation angles
    if rotation_csv_path is not None:
        os.makedirs(os.path.dirname(rotation_csv_path), exist_ok=True)
        with open(rotation_csv_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([file_path] + rotation_angles)

    # New canopy filename
    output_filepath = file_path.replace(".obj", "_field_cropped.obj")
    o3d.io.write_triangle_mesh(output_filepath, cropped_field)

    return output_filepath


def generate_canopy_obj_from_sobol(
    sim_id: int,
    sobol_vec: np.ndarray,
    base_json_path: str,
    out_dir: str,
    row_spacing: float = 0.76,    # meters
    plant_spacing: float = 0.20,  # meters
    rotation_csv_path: Optional[str] = None,
) -> str:
    """
    Generate one *canopy* OBJ from a Sobol vector.

    This function does two geometry steps:
      1) Use the Sobol vector and the NURBS template to create a single
         maize plant (sim_XXXXXX.obj).
      2) Tile that single plant into a small field, apply slight random
         rotations, crop the central region, and write a canopy OBJ
         (sim_XXXXXX_field_cropped.obj). The canopy OBJ path is returned.

    Parameters
    ----------
    sim_id : int
        Simulation ID (used for naming the OBJ files).
    sobol_vec : np.ndarray, shape (1 + 7 * num_leaves,)
        1D array in [0,1] with layout:
            [stalk_scale,
             interleaf[0..L-1],
             length[0..L-1],
             width[0..L-1],
             theta[0..L-1],
             phi[0..L-1],
             curvature[0..L-1],
             twist[0..L-1]]
    base_json_path : str
        Path to base NURBS JSON file (your Mo17 template).
    out_dir : str
        Directory to write the OBJ files into.
    row_spacing : float, optional
        Row spacing (m) used when tiling plants into a field.
    plant_spacing : float, optional
        In-row plant spacing (m) used when tiling plants.
    rotation_csv_path : str or None, optional
        If provided, log random per-plant rotation angles to this CSV file.

    Returns
    -------
    str
        Full path to the *canopy* OBJ file (the field-cropped mesh).
    """
    # -------------------------------
    # REPRODUCIBILITY SEED (ADD HERE)
    # -------------------------------
    random.seed(sim_id)
    np.random.seed(sim_id)
    
    # ---- Load base NURBS plant ----
    data = exchange.import_json(base_json_path)
    surf_cont = multi.SurfaceContainer(data)
    num_leaves = len(surf_cont) - 1

    # ---- Validate Sobol vector length ----
    sobol_vec = np.asarray(sobol_vec, dtype=float).ravel()
    expected_len = 1 + 7 * num_leaves
    if sobol_vec.size != expected_len:
        raise ValueError(
            f"Expected sobol_vec length {expected_len} for {num_leaves} leaves, "
            f"got {sobol_vec.size}"
        )

    # ---- Unpack Sobol vector ----
    idx = 0
    raw_stalk = sobol_vec[idx]; idx += 1

    raw_interleaf = sobol_vec[idx:idx + num_leaves]; idx += num_leaves
    raw_length    = sobol_vec[idx:idx + num_leaves]; idx += num_leaves
    raw_width     = sobol_vec[idx:idx + num_leaves]; idx += num_leaves
    raw_theta     = sobol_vec[idx:idx + num_leaves]; idx += num_leaves
    raw_phi       = sobol_vec[idx:idx + num_leaves]; idx += num_leaves
    raw_curv      = sobol_vec[idx:idx + num_leaves]; idx += num_leaves
    raw_twist     = sobol_vec[idx:idx + num_leaves]; idx += num_leaves

    # ---- Map to physical ranges ----
    # Stalk scale factor
    stalk_scale = float(map_to_range(raw_stalk, 0.7, 1.3))

    # Relative inter-leaf positions along the stalk (0.1–1.0 of stalk height)
        # interleaf_distances = map_to_range(raw_interleaf, 0.1, 1.0)
    # Relative inter-leaf positions along the stalk (0.0–1.0); canopy_geometry.py enforces 10–95% band
    # -----------------------------
    # Inter-leaf: enforce min/max SPACING (cm), but still output POSITIONS in [0,1]
    # -----------------------------
    min_spacing_cm = 7.0
    max_spacing_cm = 20.0

    # These must match canopy_geometry.py’s band logic
    min_leaf_offset_cm = 13.0
    top_margin_cm = 15.0

    # Estimate available vertical band using the (about-to-be) scaled stalk height.
    # scale_stalk_length() scales Z about min_z, so the height scales ~linearly.
    stalk_pts = surf_cont[0].ctrlpts
    base_min_z = min(p[2] for p in stalk_pts)
    base_max_z = max(p[2] for p in stalk_pts)
    base_stalk_h = base_max_z - base_min_z
    scaled_stalk_h = base_stalk_h * stalk_scale

    available_z_span = (scaled_stalk_h - top_margin_cm) - min_leaf_offset_cm
    if available_z_span <= 1e-6:
        # Pathological stalk — fall back to unconstrained positions
        interleaf_distances = raw_interleaf
    else:
        n = num_leaves
        target_sum = available_z_span

        # If constraints are infeasible for this stalk, relax safely.
        min_feasible = target_sum / (n - 1)
        if (n - 1) * min_spacing_cm > target_sum:
            min_spacing = min_feasible
            max_spacing = min_feasible
        elif (n - 1) * max_spacing_cm < target_sum:
            min_spacing = min_feasible
            max_spacing = min_feasible
        else:
            min_spacing = min_spacing_cm
            max_spacing = max_spacing_cm

        # Use Sobol values to propose (n-1) spacings in [min_spacing, max_spacing]
        u = np.asarray(raw_interleaf[: n - 1], dtype=float)  # (n-1) gaps
        spacings = min_spacing + u * (max_spacing - min_spacing)
        spacings = np.clip(spacings, min_spacing, max_spacing)

        # Project spacings to exactly sum to target_sum while staying within bounds
        for _ in range(20):
            delta = target_sum - float(np.sum(spacings))
            if abs(delta) < 1e-6:
                break
            if delta > 0:
                room = (max_spacing - spacings)
            else:
                room = (spacings - min_spacing)
            room_sum = float(np.sum(room))
            if room_sum <= 1e-12:
                break
            spacings = spacings + delta * (room / room_sum)
            spacings = np.clip(spacings, min_spacing, max_spacing)

        # Convert spacings -> absolute attachment z’s within the band (relative to band bottom)
        attachment_rel = np.insert(np.cumsum(spacings), 0, 0.0)  # length n, starts at 0
        # Convert absolute (within-band) -> normalized positions in [0,1]
        interleaf_distances = attachment_rel / target_sum
        interleaf_distances = np.clip(interleaf_distances, 0.0, 1.0)

    # Leaf size (cm)
    lengths = map_to_range(raw_length, 40.0, 110.0)    # cm
    widths  = map_to_range(raw_width, 5.0, 14.0)       # cm

    # Leaf orientation (deg)
    thetas  = map_to_range(raw_theta, -40.0, 40.0)     # vertical angle
    phis    = map_to_range(raw_phi, 0.0, 360.0)        # azimuth

    # Curvature (dimensionless-ish) and continuous twist (signed warp amplitude)
    curvatures = map_to_range(raw_curv, 0.0, 2.0)
    twists     = map_to_range(raw_twist, -5.0, 5.0)  # negative / zero / positive

    # ---- Apply parameters to NURBS surfaces ----
    change_leaf_parameters(
        surf_cont,
        stalk_scale=stalk_scale,
        inter_leaf_distances=interleaf_distances,
        lengths=lengths,
        widths=widths,
        curvatures=curvatures,
        twists=twists,
        thetas=thetas,
        phis=phis,
    )

    # ---- Export single-plant OBJ ----
    os.makedirs(out_dir, exist_ok=True)
    single_plant_path = os.path.join(out_dir, f"sim_{sim_id:06d}.obj")
    exchange.export_obj(surf_cont, single_plant_path)

    # ---- Build canopy OBJ by tiling that plant into a field ----
    canopy_path = create_field(
        file_path=single_plant_path,
        row_spacing=row_spacing,
        plant_spacing=plant_spacing,
        rotation_csv_path=rotation_csv_path,
    )

    return canopy_path