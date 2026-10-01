"""Kinematic model of the Franka Emika Panda, and its numerical primitives.

Provenance
----------
This module is a cleaned-up, dependency-light reimplementation of ``panda.py``
from this branch, which used CasADi throughout.  Two things were changed and
nothing else:

* the transforms are evaluated in NumPy, so importing the library does not drag
  in CasADi and calling the forward kinematics is a few microseconds instead of
  a few hundred.  ``tests/test_model.py`` checks the two against each other
  entry by entry, so the numbers cannot drift;
* the joint limits, the DH table and the tool offset are named and documented
  instead of being literals inside a class.

Conventions
-----------
The link transform is the modified (Craig) Denavit-Hartenberg form

.. math::

    {}^{i-1}T_i = \\mathrm{RotX}(\\alpha_i)\\,\\mathrm{TransX}(a_i)\\,
                   \\mathrm{TransZ}(d_i)\\,\\mathrm{RotZ}(\\theta_i)

with the chain accumulated by right multiplication, ``T = T_1 T_2 ... T_n``.
``TransZ`` and ``RotZ`` commute because they share an axis, so the order between
them is immaterial; ``RotX`` and ``TransX`` do not, and the order above is the
one ``panda.py`` uses.

The table has nine rows but only seven joints.  Row 8 (``d = 0.107``) is the
constant flange offset, and row 9 (``d = 0.1034``) is only used by the tool
frame.  :func:`fk_flange` stops after row 8 -- that is the frame the analytical
inverse kinematics of this repository solves for -- and :func:`fk_tool` adds the
factory ``RotZ(-pi/4)`` that converts the flange into the nominal tool frame.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "NUM_JOINTS",
    "FLANGE_ROW",
    "DH_PARAMETERS",
    "LOWER_LIMITS_DEG",
    "UPPER_LIMITS_DEG",
    "lower_limits",
    "upper_limits",
    "rot_x",
    "rot_y",
    "rot_z",
    "trans_x",
    "trans_z",
    "link_transform",
    "forward_kinematics",
    "fk_flange",
    "fk_tool",
    "jacobian",
    "manipulability",
    "joint_frames",
    "limits",
    "TOOL_ROTATION",
]

#: Number of actuated joints.
NUM_JOINTS = 7

#: Index of the last link that :func:`fk_flange` includes (the constant 0.107 m
#: offset).  The rows after it are only used by :func:`fk_tool`.
FLANGE_ROW = 7

#: Modified DH parameters, one row per link: ``[a, d, alpha, theta_offset]``.
#: Taken verbatim from ``panda.py``; ``tools`` is 0 for every row because no
#: joint of this arm has a constant angle offset, and the ``theta`` column is
#: kept only so that the table can be read against the original.
DH_PARAMETERS = np.array(
    [
        [0.0000, 0.3330, 0.0000, 0.0],
        [0.0000, 0.0000, -np.pi / 2, 0.0],
        [0.0000, 0.3160, np.pi / 2, 0.0],
        [0.0825, 0.0000, np.pi / 2, 0.0],
        [-0.0825, 0.3840, -np.pi / 2, 0.0],
        [0.0000, 0.0000, np.pi / 2, 0.0],
        [0.0880, 0.0000, np.pi / 2, 0.0],
        [0.0000, 0.1070, 0.0000, 0.0],
        [0.0000, 0.1034, 0.0000, 0.0],
    ]
)

#: Joint limits in degrees, as the robot actually reports them.  Joint 4 is
#: one-sided (``[-175, -5]``) and joint 6 is one-sided (``[0, 214]``), which is
#: what makes the angle wrapping in this repository non-trivial.
LOWER_LIMITS_DEG = np.array([-165.0, -100.0, -165.0, -175.0, -165.0, 0.0, -165.0])
UPPER_LIMITS_DEG = np.array([165.0, 100.0, 165.0, -5.0, 165.0, 214.0, 165.0])

#: The factory tool rotation applied on top of the flange frame, in radians.
TOOL_ROTATION = -np.pi / 4


def lower_limits() -> np.ndarray:
    """Joint lower limits in radians."""
    return np.radians(LOWER_LIMITS_DEG)


def upper_limits() -> np.ndarray:
    """Joint upper limits in radians."""
    return np.radians(UPPER_LIMITS_DEG)


# --------------------------------------------------------------------------- #
# elementary transforms
# --------------------------------------------------------------------------- #
def rot_x(angle: float) -> np.ndarray:
    """Homogeneous rotation about ``x`` by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def rot_y(angle: float) -> np.ndarray:
    """Homogeneous rotation about ``y`` by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def rot_z(angle: float) -> np.ndarray:
    """Homogeneous rotation about ``z`` by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def trans_x(distance: float) -> np.ndarray:
    """Homogeneous translation along ``x``."""
    matrix = np.eye(4)
    matrix[0, 3] = distance
    return matrix


def trans_z(distance: float) -> np.ndarray:
    """Homogeneous translation along ``z``."""
    matrix = np.eye(4)
    matrix[2, 3] = distance
    return matrix


def link_transform(row: int, theta: float) -> np.ndarray:
    """Transform of link ``row`` of :data:`DH_PARAMETERS` at joint angle ``theta``.

    Args:
        row: Row of the DH table, zero-based.
        theta: Joint angle in radians (ignored for rows beyond the last joint,
            which are rigid links; the caller passes ``0``).

    Returns:
        The 4x4 transform
        ``RotX(alpha) TransX(a) TransZ(d) RotZ(theta)``.
    """
    # The fourth column (a constant joint offset) is zero for every row of this
    # arm, so it is not applied; it is kept in the table only for traceability.
    a, d, alpha, _offset = DH_PARAMETERS[row]
    return rot_x(alpha) @ trans_x(a) @ trans_z(d) @ rot_z(theta)


# --------------------------------------------------------------------------- #
# forward kinematics
# --------------------------------------------------------------------------- #
def forward_kinematics(q: Sequence[float]) -> np.ndarray:
    """All link frames of a configuration.

    Args:
        q: Seven joint angles in radians.

    Returns:
        Array of shape ``(9, 4, 4)``; entry ``i`` is the base-to-frame ``i``
        transform, so ``[7]`` is the flange frame and ``[8]`` the tool frame
        before the factory tool rotation.

    Raises:
        ValueError: If ``q`` does not have seven elements.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (NUM_JOINTS,):
        raise ValueError(f"expected {NUM_JOINTS} joint angles, got shape {q.shape}")

    frames = np.empty((DH_PARAMETERS.shape[0], 4, 4))
    current = np.eye(4)
    for row in range(DH_PARAMETERS.shape[0]):
        theta = q[row] if row < NUM_JOINTS else 0.0
        current = current @ link_transform(row, theta)
        frames[row] = current
    return frames


def fk_flange(q: Sequence[float]) -> np.ndarray:
    """Flange pose of a configuration -- the frame the analytical IK solves for.

    Args:
        q: Seven joint angles in radians.

    Returns:
        The 4x4 base-to-flange transform.
    """
    return forward_kinematics(q)[FLANGE_ROW]


def fk_tool(q: Sequence[float]) -> np.ndarray:
    """Nominal tool pose, i.e. the flange pose with the factory tool rotation.

    Args:
        q: Seven joint angles in radians.

    Returns:
        The 4x4 base-to-tool transform.
    """
    return forward_kinematics(q)[-1] @ rot_z(TOOL_ROTATION)


# --------------------------------------------------------------------------- #
# differential kinematics
# --------------------------------------------------------------------------- #
def jacobian(q: Sequence[float]) -> np.ndarray:
    """Geometric Jacobian of the flange frame, in the base frame.

    Args:
        q: Seven joint angles in radians.

    Returns:
        The 6x7 Jacobian, the linear part first, obtained by the usual
        axis-cross-product construction.  It is the Jacobian of the **flange**
        frame, so it matches ``original/panda.py``'s ``jacobian_flange`` (to
        5.6e-16), not its ``jacobian``: the latter differentiates the tool point,
        which sits a further ``0.1034 m`` along the flange ``z``, and the two
        differ by that lever arm.
    """
    frames = forward_kinematics(q)
    tip = frames[FLANGE_ROW][:3, 3]
    result = np.zeros((6, NUM_JOINTS))
    for index in range(NUM_JOINTS):
        axis = frames[index][:3, 2]
        offset = tip - frames[index][:3, 3]
        result[:3, index] = np.cross(axis, offset)
        result[3:, index] = axis
    return result


def manipulability(q: Sequence[float]) -> float:
    """Yoshikawa's manipulability measure :math:`\\sqrt{\\det(JJ^T)}`.

    Args:
        q: Seven joint angles in radians.

    Returns:
        The measure, zero at a singular configuration.
    """
    matrix = jacobian(q)
    determinant = float(np.linalg.det(matrix @ matrix.T))
    return float(np.sqrt(determinant)) if determinant > 0.0 else 0.0


def joint_frames(q: Sequence[float]) -> List[np.ndarray]:
    """Origins of the actuated joint frames, base first.

    Args:
        q: Seven joint angles in radians.

    Returns:
        List of seven 3-vectors, useful for drawing the arm.
    """
    frames = forward_kinematics(q)
    return [frames[index][:3, 3].copy() for index in range(NUM_JOINTS)]


def limits() -> Tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` joint limits in radians."""
    return (lower_limits(), upper_limits())
