"""The SRS-equivalent geometry that the Franka analytical IK is built on.

This module holds the geometric core of the method, so that the derivation can
be read on its own and tested on its own.

The reduction in one paragraph
------------------------------
The Panda is *almost* an S-R-S arm.  Its shoulder is offset by
``bias = 0.0825 m`` from the point where the three shoulder axes would intersect
(DH rows 4 and 5 carry ``a = +-0.0825``), and its wrist carries the
``offset = 0.088 m`` of DH row 7.  The method of this repository removes the two
obstructions in turn:

1. **the wrist offset** makes the last link reach the flange at a fixed extra
   angle :math:`\\beta = \\arctan(\\text{offset}/d_{wt})` compared with an S-R-S
   wrist.  Rotating the target orientation by
   :math:`R_z(-q_7)\\,R_y(-\\beta)\\,R_z(q_7)` (:func:`wrist_correction`) makes
   the flange frame behave exactly like an S-R-S flange, at the price of
   replacing :math:`d_{wt}` with :math:`\\sqrt{\\text{offset}^2 + d_{wt}^2}`;
2. **the shoulder bias** shifts the elbow, so joint 4 no longer obeys the plain
   S-R-S elbow law :math:`\\cos\\theta_4 = (\\|x_{sw}\\|^2 - d_{se}^2 -
   d_{ew}^2)/(2\\,d_{se}d_{ew})`.  Solving the same triangle with the bias
   included gives a **quadratic in** :math:`\\tan(\\theta_4/2)` instead -- see
   :func:`q4_roots` -- and that quadratic is where this method's extra
   multiplicity lives.

The result is not one S-R-S arm but a one-parameter *family* of them.  Once
:math:`\\theta_4` is fixed, the equivalent links are

.. math::

    l_{se} = [0,\\; d_{se} + \\delta,\\; 0], \\qquad
    l_{ew} = [0,\\; 0,\\; d_{ew} + \\delta], \\qquad
    \\delta = -\\tan(\\theta_4/2)\\;\\text{bias},

and for each member of the family the standard S-R-S solution of Shimizu et al.
(2008) applies directly, provided the equivalent arm angle :math:`\\phi` is
chosen so that the resulting joint 7 equals the value that was asked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = [
    "EquivalentGeometry",
    "PAPER_GEOMETRY",
    "wrist_offset_angle",
    "effective_wrist_length",
    "rot_x",
    "rot_y",
    "rot_z",
    "wrist_correction",
    "shoulder_to_wrist",
    "q4_coefficients",
    "q4_discriminant",
    "q4_roots",
    "link_offset",
    "equivalent_link_vectors",
    "equivalent_joint_rotation",
]

#: Rotation convention of the reduction: the rotation matrix of joint ``i`` is
#: ``Rz(theta) Rx(alpha_i)`` with the ``alpha`` of the *standard* DH rows 1..7 of
#: :mod:`franka_ik.model`.  It is the convention the published implementation
#: uses, and it differs from the modified-DH transforms of the model module; the
#: two are reconciled numerically in ``tests/test_solver.py``.
DEFAULT_ALPHAS = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]) * (math.pi / 2.0)


@dataclass(frozen=True)
class EquivalentGeometry:
    """Geometric parameters of the reduction.

    Attributes:
        d_bs: Base-to-shoulder offset (DH row 1, ``d``).
        d_se: Shoulder-to-elbow length (DH row 3, ``d``).
        d_ew: Elbow-to-wrist length (DH row 5, ``d``).
        d_wt: Wrist-to-flange length (DH row 8, ``d``).
        offset: The ``a`` of DH row 7, the wrist offset that makes the arm
            non-S-R-S.
        bias: The ``a`` of DH rows 4 and 5, the shoulder offset.
    """

    d_bs: float
    d_se: float
    d_ew: float
    d_wt: float
    offset: float
    bias: float


#: The parameters the published implementation on this branch uses.
PAPER_GEOMETRY = EquivalentGeometry(
    d_bs=0.333, d_se=0.316, d_ew=0.384, d_wt=0.107, offset=0.088, bias=0.0825
)


def wrist_offset_angle(geometry: EquivalentGeometry = PAPER_GEOMETRY) -> float:
    """``beta = atan2(offset, d_wt)``: how far the flange frame is rotated."""
    return math.atan2(geometry.offset, geometry.d_wt)


def effective_wrist_length(geometry: EquivalentGeometry = PAPER_GEOMETRY) -> float:
    """``sqrt(offset^2 + d_wt^2)``: the last link length after step 1."""
    return math.hypot(geometry.offset, geometry.d_wt)


# --------------------------------------------------------------------------- #
# small rotation helpers (3x3, the convention of the reduction)
# --------------------------------------------------------------------------- #
def rot_x(angle: float) -> np.ndarray:
    """Rotation about ``x`` by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    """Rotation about ``y`` by ``angle`` radians.

    Note the sign convention: ``[[c, 0, -s], [0, 1, 0], [s, 0, c]]``, matching
    the ``rot_y`` of the published implementation.  Flipping it silently
    mirrors the wrist correction and every branch returns a wrong pose, which is
    exactly how this bug was found -- the solver reported a pose residual of
    0.865 instead of 1e-16.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    """Rotation about ``z`` by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def equivalent_joint_rotation(
    theta: float, index: int, alphas: np.ndarray = DEFAULT_ALPHAS
) -> np.ndarray:
    """Rotation matrix ``Rz(theta) Rx(alpha_index)`` of joint ``index`` (1-based).

    This is the convention the reduction works in.  It is the *standard* DH
    rotation :math:`{}^{i-1}R_i` of eq. (3) of Shimizu et al., not the modified
    DH transform of :mod:`franka_ik.model`.

    Args:
        theta: Joint angle in radians.
        index: One-based joint index, 1..7.
        alphas: The seven ``alpha`` values.

    Returns:
        The 3x3 rotation matrix.
    """
    return rot_z(theta) @ rot_x(float(alphas[index - 1]))


def wrist_correction(
    q7: float, geometry: EquivalentGeometry = PAPER_GEOMETRY
) -> np.ndarray:
    """Step 1 of the reduction: the rotation applied to the target orientation.

    .. math:: R_{\\text{srs}} = R_{\\text{franka}}\\;R_z(-q_7)\\,R_y(-\\beta)\\,R_z(q_7)

    Args:
        q7: Joint 7 value that parameterises the branch, in radians.
        geometry: Geometric parameters.

    Returns:
        The 3x3 matrix to post-multiply onto the target rotation.
    """
    return rot_z(-q7) @ rot_y(-wrist_offset_angle(geometry)) @ rot_z(q7)


def shoulder_to_wrist(
    position: np.ndarray,
    corrected_rotation: np.ndarray,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
) -> np.ndarray:
    """Shoulder-to-wrist vector of the equivalent S-R-S arm.

    Args:
        position: Tip (flange) position, 3-vector.
        corrected_rotation: The orientation from :func:`wrist_correction` applied
            to the target, i.e. ``R_franka @ wrist_correction(q7)``.
        geometry: Geometric parameters.

    Returns:
        :math:`{}^0x_{sw} = x - [0,0,d_{bs}] - R\\,[0,0,\\ell_{wt}]`.
    """
    last_link = np.array([0.0, 0.0, effective_wrist_length(geometry)])
    base = np.array([0.0, 0.0, geometry.d_bs])
    return np.asarray(position, dtype=float) - base - corrected_rotation @ last_link


# --------------------------------------------------------------------------- #
# the elbow quadratic -- the heart of the reduction, and of its multiplicity
# --------------------------------------------------------------------------- #
def q4_coefficients(
    distance_squared: float, geometry: EquivalentGeometry = PAPER_GEOMETRY
) -> Tuple[float, float, float]:
    """Coefficients ``(a2, a1, a0)`` of the quadratic in ``tan(theta_4 / 2)``.

    The equation is ``a2 x^2 - a1 x + a0 = 0`` with

    .. math::

        a_2 &= 4\\,b^2 + (d_{se} - d_{ew})^2 - \\|x_{sw}\\|^2 \\\\
        a_1 &= 4\\,b\\,(d_{se} + d_{ew}) \\\\
        a_0 &= (d_{se} + d_{ew})^2 - \\|x_{sw}\\|^2

    where :math:`b` is the shoulder bias.  Setting ``b = 0`` collapses this to
    the S-R-S law of eq. (12) of Shimizu et al.

    Args:
        distance_squared: Squared norm of the shoulder-to-wrist vector.
        geometry: Geometric parameters.

    Returns:
        ``(a2, a1, a0)``.
    """
    bias = geometry.bias
    a2 = 4.0 * bias**2 + (geometry.d_se - geometry.d_ew) ** 2 - distance_squared
    a1 = 4.0 * bias * (geometry.d_se + geometry.d_ew)
    a0 = (geometry.d_se + geometry.d_ew) ** 2 - distance_squared
    return (a2, a1, a0)


def q4_discriminant(
    distance_squared: float, geometry: EquivalentGeometry = PAPER_GEOMETRY
) -> float:
    """Discriminant ``a1^2 - 4 a0 a2`` of the elbow quadratic.

    A negative value means the target is outside the reachable shell, i.e. the
    wrist cannot be placed at that distance from the shoulder at all.

    Args:
        distance_squared: Squared norm of the shoulder-to-wrist vector.
        geometry: Geometric parameters.

    Returns:
        The discriminant.
    """
    a2, a1, a0 = q4_coefficients(distance_squared, geometry)
    return a1 * a1 - 4.0 * a0 * a2


def q4_roots(
    distance_squared: float,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    tolerance: float = 1e-12,
) -> List[float]:
    """**Both** solutions of the elbow quadratic, as joint-4 angles.

    This function is the difference between the four-branch solver that this
    repository originally published and the eight-branch solver it ships now.
    The quadratic has two roots and the published code took only the ``+`` one,
    which silently discards half of the solution set -- see
    ``docs/branch_analysis.md``.  The two roots correspond to the elbow being on
    either side of the shoulder-wrist line, exactly as
    :math:`\\theta_4 = \\pm\\arccos(\\cdot)` does in the S-R-S case.

    Args:
        distance_squared: Squared norm of the shoulder-to-wrist vector.
        geometry: Geometric parameters.
        tolerance: Relative tolerance below which the discriminant counts as
            zero and the two roots are reported as one.

    Returns:
        Zero, one or two joint-4 angles in radians.  Empty when the target is
        out of reach.
    """
    a2, a1, a0 = q4_coefficients(distance_squared, geometry)
    discriminant = a1 * a1 - 4.0 * a0 * a2
    scale = max(abs(a1 * a1), abs(4.0 * a0 * a2), 1e-30)
    if discriminant < -tolerance * scale:
        return []
    root = math.sqrt(max(discriminant, 0.0))
    if abs(a2) < 1e-15:  # pragma: no cover - degenerate geometry
        return []
    angles = []
    for sign in (+1.0, -1.0):
        value = (a1 + sign * root) / (2.0 * a2)
        angles.append(2.0 * math.atan(value))
    if abs(root) <= tolerance * max(abs(a1), 1e-30):
        return [angles[0]]
    return angles


def link_offset(theta4: float, geometry: EquivalentGeometry = PAPER_GEOMETRY) -> float:
    """``delta``: how much the equivalent links grow for a given ``theta_4``.

    Args:
        theta4: Joint 4 angle in radians.
        geometry: Geometric parameters.

    Returns:
        ``-tan(theta_4 / 2) * bias``.  Zero when the bias is zero, which is what
        makes the reduction exact for a true S-R-S arm.
    """
    return -math.tan(theta4 / 2.0) * geometry.bias


def equivalent_link_vectors(
    theta4: float, geometry: EquivalentGeometry = PAPER_GEOMETRY
) -> Tuple[np.ndarray, np.ndarray]:
    """The two link vectors of the equivalent S-R-S arm.

    Args:
        theta4: Joint 4 angle in radians.
        geometry: Geometric parameters.

    Returns:
        ``(l_se, l_ew)`` with ``l_se = [0, d_se + delta, 0]`` and
        ``l_ew = [0, 0, d_ew + delta]``.
    """
    delta = link_offset(theta4, geometry)
    return (
        np.array([0.0, geometry.d_se + delta, 0.0]),
        np.array([0.0, 0.0, geometry.d_ew + delta]),
    )
