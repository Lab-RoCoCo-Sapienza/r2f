# Partial credits to https://github.com/cvg/FrontierNet/blob/main/utils/geometry.py

import numpy as np
import cv2
from typing import Tuple
from scipy.spatial.transform import Rotation as R


# Rotation/Transformation
EPS = 1e-8  # numerical tolerance

def rot2quat(rot):
    quat = R.from_matrix(rot).as_quat()  # xyzw
    return quat

def _safe_normalize(v, eps=EPS):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector encountered.")
    return v / n

def _rot_between(u, v, eps=EPS):
    """
    Small-angle-safe rotation that takes `u` onto `v`
    (both assumed unit).
    """
    dot = np.dot(u, v)
    if dot > 1 - eps:  # already aligned
        return R.identity()
    if dot < -1 + eps:  # opposite; pick any orthogonal axis
        # choose an axis least parallel to u for numerical stability
        helper = (
            np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        axis = _safe_normalize(np.cross(u, helper))
        return R.from_rotvec(axis * np.pi)

    axis = _safe_normalize(np.cross(u, v))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle)


def compute_alignment_transforms(
    origins,
    align_vec,
    align_axis,
    appr_vec,
    appr_axis,
    refine=True,
    eps=EPS,
    reortho=True,
):
    """
    A function to compute a list of 4x4 homogeneous transforms
    that align a set of origins with a specified direction and approach vector.
    The transforms are computed such that its align_axis aligns with the align_vec,
    and its approach axis try to align with the appr_vec as much as possible.
    Parameters:
    ----------
    origins : list[np.ndarray]
        List of 3D points (origins) where the transforms will be applied.
    align_vec : np.ndarray
        The target direction vector to align with.
    align_axis : np.ndarray
        The axis along which the align_vec should be aligned.
    appr_vec : np.ndarray
        The target approach vector to align with as much as possible.
    appr_axis : np.ndarray
        The axis along which the appr_vec should be aligned as much as possible.
    refine : bool, optional
        If True, applies an additional refinement step to align the approach vector.
        Default is True.
    eps : float, optional
        Numerical tolerance for vector normalization and rotation calculations.
        Default is 1e-8.
    reortho : bool, optional
        If True, re-orthonormalizes the resulting rotation matrix to ensure it is orthonormal.
        This can help eliminate numerical drift in the rotation matrix.
        Default is True.

    Returns
    -------
    list[np.ndarray] one 4x4 homogeneous transform per origin.
    """

    # align_axis and appr_axis should be orthonormal unit vectors.
    if not (
        np.isclose(np.linalg.norm(align_axis), 1.0, atol=eps)
        and np.isclose(np.linalg.norm(appr_axis), 1.0, atol=eps)
    ):
        raise ValueError("align_axis and appr_axis must be unit vectors.")
    if not (np.isclose(np.dot(align_axis, appr_axis), 0.0, atol=eps)):
        raise ValueError("align_axis and appr_axis must be orthogonal.")

    # upfront normalisation of target directions
    align_vec = _safe_normalize(align_vec, eps)
    appr_vec = _safe_normalize(appr_vec, eps)
    align_axis = _safe_normalize(align_axis, eps)
    appr_axis = _safe_normalize(appr_axis, eps)

    T_list = []

    # primary rotation: align align_axis → align_vec
    R1 = _rot_between(align_axis, align_vec, eps)

    for origin in origins:
        # Optional refinement: twist about new align_vec so (R1·appr_axis) lines up with appr_vec
        if refine:
            new_appr_axis = R1.apply(appr_axis)

            # project both into the plane ⟂ align_vec
            proj_a = new_appr_axis - np.dot(new_appr_axis, align_vec) * align_vec
            proj_b = appr_vec - np.dot(appr_vec, align_vec) * align_vec

            n_a = np.linalg.norm(proj_a)
            n_b = np.linalg.norm(proj_b)

            if n_a > eps and n_b > eps:
                proj_a /= n_a
                proj_b /= n_b

                dot = np.clip(np.dot(proj_a, proj_b), -1.0, 1.0)
                angle = np.arccos(dot)

                # choose sign so the rotation has the shorter direction
                if np.dot(np.cross(proj_a, proj_b), align_vec) < 0:
                    angle = -angle

                R2 = R.from_rotvec(align_vec * angle)
                R_comb = R2 * R1
            else:
                # projections degenerate ⇒ skip refinement
                R_comb = R1
        else:
            R_comb = R1

        # optional re-orthonormalisation to kill tiny drift
        if reortho:
            U, _, Vt = np.linalg.svd(R_comb.as_matrix())
            R_comb = R.from_matrix(U @ Vt)

        # build homogeneous transform
        T = np.eye(4)
        T[:3, :3] = R_comb.as_matrix()
        T[:3, 3] = origin
        T_list.append(T)

    return T_list


def pose_difference(tf_1, tf_2):
    """
    Compute pairwise translation and rotation differences between two sets of transformation matrices.

    Args:
        tf_1: (N, 4, 4) numpy array
        tf_2: (M, 4, 4) numpy array

    Returns:
        translation_diff: (N, M) numpy array of L2 translation differences
        rotation_diff: (N, M) numpy array of rotation angle differences (in radians)
    """
    assert tf_1.shape[1:] == (4, 4) and tf_2.shape[1:] == (
        4,
        4,
    ), "Input tensors must be of shape (N, 4, 4) and (M, 4, 4)"
    N, M = tf_1.shape[0], tf_2.shape[0]

    # Extract translations
    trans_1 = tf_1[:, :3, 3]  # (N, 3)
    trans_2 = tf_2[:, :3, 3]  # (M, 3)

    # Compute translation differences (N, M)
    trans_diff = np.linalg.norm(trans_1[:, None, :] - trans_2[None, :, :], axis=-1)

    # Extract rotations
    rot_1 = tf_1[:, :3, :3]  # (N, 3, 3)
    rot_2 = tf_2[:, :3, :3]  # (M, 3, 3)

    # Compute relative rotation matrices: R_rel = R2 @ R1.T for all pairs
    rot_1_T = rot_1.transpose(0, 2, 1)  # (N, 3, 3)
    rel_rot = np.einsum("mij,njk->nmik", rot_2, rot_1_T)

    # Flatten to compute rotation differences using scipy
    rel_rot_flat = rel_rot.reshape(-1, 3, 3)
    relative_rotations = R.from_matrix(rel_rot_flat)
    rot_diff = np.linalg.norm(relative_rotations.as_rotvec(), axis=1).reshape(N, M)

    return trans_diff, rot_diff



# Image Processing
def compute_gradient(
    image: np.ndarray, kernel_size: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute horizontal and vertical gradients of a mono image using the Sobel operator.

    Args:
        image (np.ndarray): 2D image.
        kernel_size (int): Size of the extended Sobel kernel; must be odd and positive.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Horizontal (x) and vertical (y) gradients.
    """
    # Validate inputs
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer.")

    image = np.asarray(image, dtype=np.float64)

    grad_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)

    return grad_x, grad_y


def grad_mag_and_direct_from_gradmap(
    grad_x: np.ndarray, grad_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the magnitude and direction (in degrees) of the gradient.

    Args:
        grad_x (np.ndarray): Horizontal (x) gradient.
        grad_y (np.ndarray): Vertical (y) gradient.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradient magnitude and direction in degrees.
    """
    if grad_x.shape != grad_y.shape:
        raise ValueError("grad_x and grad_y must have the same shape.")

    magnitude = np.hypot(
        grad_x, grad_y
    )  # more stable and preferred over sqrt(x**2 + y**2)
    direction = np.degrees(np.arctan2(grad_y, grad_x))  # converts to degrees directly

    return magnitude, direction



# Depth mask / backprojection helpers
def compute_valid_mask(
    depth: np.ndarray,
    near: float,
    far: float,
    top_crop_fraction: float,
) -> np.ndarray:
    """
    Build a boolean mask of pixels with usable depth values.

    A pixel is valid when its depth is finite and within [near, far].
    Optionally, the top rows of the image are masked out to suppress
    ceiling / sky frontiers.

    Returns
    -------
    valid : (H, W) bool ndarray
    """
    valid = np.isfinite(depth) & (depth > near) & (depth < far)

    if top_crop_fraction > 0.0:
        H = depth.shape[0]
        crop_rows = int(H * top_crop_fraction)
        valid[:crop_rows, :] = False

    return valid


def compute_missing_depth_boundary(valid_mask: np.ndarray) -> np.ndarray:
    """
    Find valid pixels that border at least one invalid pixel.

    These "missing-depth boundary" pixels indicate occlusion edges or
    sensor-range limits -- natural frontier indicators.

    Returns
    -------
    boundary : (H, W) uint8 ndarray (0 or 1)
    """
    invalid_mask = (~valid_mask).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_invalid = cv2.dilate(invalid_mask, kernel, iterations=1)

    boundary = (dilated_invalid > 0) & valid_mask
    return boundary.astype(np.uint8)


def backproject_pixel_to_ray(
    u: float,
    v: float,
    K_inv: np.ndarray,
) -> np.ndarray:
    """
    Backproject a single pixel (u, v) into a unit ray in camera frame.

    The returned ray is in **OpenCV pinhole convention** (+X right,
    +Y down, +Z forward).  If the pose rotation matrix R uses the
    Habitat / OpenGL camera convention (+X right, +Y up, −Z forward),
    use :func:`pixel_to_world_bearing` instead, which handles the
    coordinate flip automatically.

    Parameters
    ----------
    u, v : float
        Pixel coordinates (column, row).
    K_inv : (3, 3) ndarray
        Inverse of the camera intrinsic matrix.

    Returns
    -------
    ray : (3,) unit vector in camera frame (OpenCV convention)
    """
    ray = K_inv @ np.array([u, v, 1.0])
    norm = np.linalg.norm(ray)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return ray / norm


# OpenCV (+X right, +Y down, +Z fwd) → OpenGL (+X right, +Y up, −Z fwd)
_CV_TO_GL = np.array([1.0, -1.0, -1.0])


def pixel_to_world_bearing(
    u: float,
    v: float,
    K_inv: np.ndarray,
    R_cam2world: np.ndarray,
) -> np.ndarray:
    """
    Backproject a pixel to a **unit bearing vector in world frame**.

    Handles the OpenCV → OpenGL coordinate flip that is required when
    the camera intrinsic matrix K uses the standard pinhole convention
    (+Z forward) but the pose rotation R transforms from the Habitat /
    OpenGL camera frame (−Z forward, +Y up).

    Parameters
    ----------
    u, v : float
        Pixel coordinates (column, row).
    K_inv : (3, 3) ndarray
        Inverse of the camera intrinsic matrix.
    R_cam2world : (3, 3) ndarray
        Upper-left 3×3 rotation block of the camera-to-world pose
        (Habitat / OpenGL convention).

    Returns
    -------
    bearing : (3,) unit vector in world frame.
    """
    ray_cv = K_inv @ np.array([u, v, 1.0])
    ray_gl = ray_cv * _CV_TO_GL  # flip Y and Z
    bearing = R_cam2world @ ray_gl
    norm = np.linalg.norm(bearing)
    if norm < 1e-8:
        return np.array([0.0, 0.0, -1.0])
    return bearing / norm


def compute_wedge_from_pixels(
    pixels_uv: np.ndarray,
    K_inv: np.ndarray,
    min_half_angle: float,
    max_half_angle: float,
) -> Tuple[float, float]:
    """
    Compute a bearing wedge (theta0, delta) from a set of frontier pixels.

    theta0 is the horizontal angle (azimuth) of the centroid ray in camera
    frame, measured as atan2(x, z) so that 0 = straight ahead.

    delta is the half-angle that covers the angular spread of the component
    pixels, clamped to [min_half_angle, max_half_angle].

    Parameters
    ----------
    pixels_uv : (N, 2) array of (u, v) pixel coords
    K_inv : (3, 3) inverse intrinsic matrix
    min_half_angle, max_half_angle : float (radians)

    Returns
    -------
    (theta0, delta) : tuple of float (radians)
    """
    centroid = pixels_uv.mean(axis=0)
    ray_center = K_inv @ np.array([centroid[0], centroid[1], 1.0])
    theta0 = np.arctan2(ray_center[0], ray_center[2])

    ones = np.ones(len(pixels_uv))
    homog = np.column_stack([pixels_uv[:, 0], pixels_uv[:, 1], ones])  # (N, 3)
    rays = (K_inv @ homog.T).T  # (N, 3)
    azimuths = np.arctan2(rays[:, 0], rays[:, 2])

    diffs = np.abs(azimuths - theta0)
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    max_dev = diffs.max() if len(diffs) > 0 else 0.0

    delta = float(np.clip(max_dev, min_half_angle, max_half_angle))
    return (theta0, delta)