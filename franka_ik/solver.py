"""The analytical inverse kinematics of the Franka Emika Panda.

The method is the one derived in ``franka解析反解方法.pdf`` on this branch: reduce
the Panda to an equivalent S-R-S arm (:mod:`franka_ik.geometry`), solve that arm
with the closed form of Shimizu et al. (2008), and undo the reduction.  The
redundancy is parameterised by **joint 7** rather than by an arm angle, so the
caller says "reach this pose with joint 7 at this value" and gets the finitely
many configurations that do so.

Branch structure
----------------
Four independent two-way choices appear on the way:

======================  ======================================================
``q4_root``             which root of the elbow quadratic is used, i.e. which
                        side of the shoulder-wrist line the elbow lies on.  This
                        is the one the originally published four-branch solver
                        discarded; see ``docs/branch_analysis.md``.
``phi_root``            which of the two equivalent arm angles satisfies the
                        requested joint 7.
``shoulder_flip``       ``(q1, q2, q3) -> (q1 + pi, -q2, q3 + pi)``: a genuine
                        two-fold symmetry of the arm, verified numerically in
                        ``tests/test_branches.py``.
``wrist_flip``          the second wrist solution ``(q5 + pi, -q6)``, which
                        this implementation selects by checking which candidate
                        reaches the pose rather than by the published
                        ``criteria`` test -- that test carries a factor
                        ``cos(q7)`` and degenerates at ``q7 = +-90 deg``.
========================  ======================================================

That gives eight candidate configurations, of which typically one to eight
reach the pose while satisfying the joint limits.  ``docs/branch_analysis.md``
records the measurement, and the numbers below are from
``analysis.coverage_study`` with targets drawn uniformly inside the real joint
limits (measured at three seeds: 100 % recovery for eight branches in all of
them, 88.5 %, 88.0 % and 86.5 % for the published four over 600 trials each).

Why joint 7 is a parameter at all
---------------------------------
Shimizu et al. argue in Section VI-A of their paper *against* using a joint angle
as the redundancy parameter: the relation between a joint angle and the arm angle
can be **cyclic** rather than monotonic, in which case one joint angle
corresponds to several arm angles and the parameterisation stops being unique.
This repository uses joint 7 anyway -- it is what the Franka controller and the
2020 derivation exposed -- and the two ``phi_root`` values are exactly the
phenomenon that argument warns about: the requested joint 7 is reached at two
different equivalent arm angles of the equivalent S-R-S arm.  Enumerating both,
rather than picking one, is what turns the warning into a non-issue here.  It is
also why the solver cannot be described by a single closed-form function of the
joint 7 value: the caller has to be handed the whole set, which is what
:func:`solve` does.

Joint limits
------------
Two different things are called "wrapping" in the original code and they are
kept apart here:

* :func:`wrap_to_limits` maps an angle into ``[lower, upper]`` of the real
  robot.  It replaces the published ``limit_joints``, whose windows for joints 4
  and 6 were wider than 360 degrees and whose loop could not terminate on a
  ``nan`` -- see ``docs/limitations.md``.
* :func:`solve` returns every candidate, with a flag saying whether it satisfies
  the limits, rather than silently projecting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import model
from .geometry import (
    PAPER_GEOMETRY,
    EquivalentGeometry,
    equivalent_joint_rotation,
    equivalent_link_vectors,
    q4_roots,
    shoulder_to_wrist,
    wrist_correction,
    wrist_offset_angle,
)

__all__ = [
    "IkSolution",
    "wrap_to_limits",
    "solve",
    "solve_branch",
    "solve_closest",
    "branch_solutions",
    "NUM_BRANCHES",
]

#: Number of candidate configurations the solver enumerates.
NUM_BRANCHES = 8


@dataclass
class IkSolution:
    """One analytical inverse-kinematics solution.

    Attributes:
        q: Joint angles in radians, seven of them.
        q4_root: ``+1`` or ``-1``: which root of the elbow quadratic was used.
        phi_root: ``+1`` or ``-1``: which equivalent arm angle was used.
        shoulder_flip: Whether the shoulder-flip symmetry was applied.
        wrist_flipped: Whether the internal ``criteria`` test flipped the wrist.
        pose_error: Largest absolute entrywise error of ``fk_flange(q)`` against
            the target, i.e. a direct measure of whether the branch is right.
        within_limits: Whether every joint is inside the real joint limits.
    """

    q: np.ndarray
    q4_root: int
    phi_root: int
    shoulder_flip: bool
    wrist_flipped: bool
    pose_error: float
    within_limits: bool

    @property
    def label(self) -> str:
        """Short identifier, used in reports and tests."""
        parts = [f"q4{'+' if self.q4_root > 0 else '-'}"]
        parts.append(f"phi{'+' if self.phi_root > 0 else '-'}")
        parts.append("flip" if self.shoulder_flip else "plain")
        return " ".join(parts)

    def describe(self, precision: int = 3) -> str:
        """One-line summary in degrees."""
        angles = np.degrees(self.q)
        text = ", ".join(f"{value:{precision + 4}.{precision}f}" for value in angles)
        return (
            f"{self.label:16s} q = [{text}] deg   "
            f"pose error {self.pose_error:.2e}   "
            f"{'in limits' if self.within_limits else 'OUT OF LIMITS'}"
        )


def wrap_to_limits(
    q: Sequence[float],
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    tolerance: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map joint angles into the real joint ranges, and report what cannot be.

    For every joint the representative ``q + 2*pi*k`` that lies inside
    ``[lower, upper]`` is chosen.  Joints 4 and 6 of the Panda have one-sided
    ranges (``[-175, -5]`` and ``[0, 214]`` degrees) so this is not the usual
    symmetric wrap; a joint whose range cannot contain the angle at all is left
    at the wrapped value closest to its range.

    Args:
        q: Joint angles in radians.
        lower: Lower limits; defaults to the model's.
        upper: Upper limits; defaults to the model's.
        tolerance: Slack used when deciding whether a representative fits.

    Returns:
        ``(wrapped, inside)`` where ``inside`` is a boolean array saying which
        joints ended up inside their range.

    Raises:
        ValueError: If ``q`` does not have seven elements or is not finite.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (model.NUM_JOINTS,):
        raise ValueError(f"expected {model.NUM_JOINTS} joint angles, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        # The published limit_joints() loops forever on a nan; this is where
        # that failure mode is turned into an error, see docs/limitations.md.
        raise ValueError("joint angles must be finite")

    lower = model.lower_limits() if lower is None else np.asarray(lower, dtype=float)
    upper = model.upper_limits() if upper is None else np.asarray(upper, dtype=float)

    wrapped = q.copy()
    inside = np.zeros(model.NUM_JOINTS, dtype=bool)
    two_pi = 2.0 * math.pi
    for index in range(model.NUM_JOINTS):
        lo, hi = float(lower[index]), float(upper[index])
        candidates = np.array([q[index] + two_pi * k for k in (-2, -1, 0, 1, 2)])
        valid = candidates[(candidates >= lo - tolerance) & (candidates <= hi + tolerance)]
        if valid.size:
            wrapped[index] = float(valid[0])
            inside[index] = True
        else:
            # No representative fits: report the closest one and flag it.
            centre = 0.5 * (lo + hi)
            wrapped[index] = float(candidates[np.argmin(np.abs(candidates - centre))])
            inside[index] = False
    return (wrapped, inside)


def solve_branch(
    pose: np.ndarray,
    q7: float,
    q4_root: int = 1,
    phi_root: int = 1,
    shoulder_flip: bool = False,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    check_pose: bool = True,
) -> Optional[IkSolution]:
    """Evaluate one of the eight branches.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 value in radians.
        q4_root: ``+1`` for the first root of the elbow quadratic, ``-1`` for the
            second.
        phi_root: ``+1`` or ``-1`` to select one of the two equivalent arm angles
            that reproduce ``q7``.
        shoulder_flip: Apply the shoulder-flip symmetry afterwards.
        geometry: Geometric parameters.
        check_pose: Evaluate the forward kinematics and record the residual.  The
            residual is what makes a branch trustworthy, so leave this on unless
            you are profiling.

    Returns:
        An :class:`IkSolution`, or ``None`` when the branch has no solution for
        this pose (out of reach, or the arc-sine of eq. (3) below is outside
        ``[-1, 1]``).
    """
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3]
    position = pose[:3, 3]
    if q4_root not in (1, -1) or phi_root not in (1, -1):
        raise ValueError("q4_root and phi_root must be +1 or -1")

    # ---- step 1: make the flange frame look like an S-R-S flange -------------
    corrected = rotation @ wrist_correction(q7, geometry)

    # ---- step 2: the elbow quadratic, both roots ----------------------------
    p_sw = shoulder_to_wrist(position, corrected, geometry)
    distance_squared = float(p_sw @ p_sw)
    if distance_squared <= 1e-18:
        return None
    roots = q4_roots(distance_squared, geometry)
    if not roots:
        return None
    if len(roots) == 1:  # noqa: SIM108  (a nested ternary reads worse here)
        # A double root: both branches coincide, which happens at the elbow
        # singularity and must not be reported as two different configurations.
        q4 = roots[0]
    else:
        q4 = roots[0] if q4_root > 0 else roots[1]

    l_se, l_ew = equivalent_link_vectors(q4, geometry)

    # ---- reference shoulder angles, i.e. the arm plane at q3 = 0 ------------
    x_aux = equivalent_joint_rotation(0.0, 3) @ (
        l_se + equivalent_joint_rotation(q4, 4) @ l_ew
    )
    y_aux = p_sw
    amplitude1 = math.hypot(float(x_aux[0]), float(x_aux[2]))
    amplitude2 = math.hypot(float(y_aux[0]), float(y_aux[1]))
    if amplitude1 < 1e-12 or amplitude2 < 1e-12:
        return None
    ratio1 = -float(x_aux[1]) / amplitude2
    ratio2 = -float(y_aux[2]) / amplitude1
    if abs(ratio1) > 1.0 + 1e-12 or abs(ratio2) > 1.0 + 1e-12:
        return None
    q1_ref = math.asin(max(-1.0, min(1.0, ratio1))) + math.atan2(
        float(y_aux[1]), float(y_aux[0])
    )
    q2_ref = math.asin(max(-1.0, min(1.0, ratio2))) + math.atan2(
        float(x_aux[2]), float(x_aux[0])
    )
    r_03_ref = (
        equivalent_joint_rotation(q1_ref, 1)
        @ equivalent_joint_rotation(q2_ref, 2)
        @ equivalent_joint_rotation(0.0, 3)
    )

    # ---- the A/B/C matrices of eqs. (15) and (19) ---------------------------
    u_sw = p_sw / math.sqrt(distance_squared)
    skew = _skew(u_sw)
    A_s = skew @ r_03_ref
    B_s = -skew @ A_s
    C_s = np.outer(u_sw, u_sw) @ r_03_ref
    R4 = equivalent_joint_rotation(q4, 4)
    A_w = R4.T @ A_s.T @ corrected
    B_w = R4.T @ B_s.T @ corrected
    C_w = R4.T @ C_s.T @ corrected

    # ---- step 3: the equivalent arm angle that reproduces q7 -----------------
    #
    # STEP3 turns "the resulting joint 7 must equal q7" into a sinusoid in the
    # equivalent arm angle.  Written the obvious way -- using the tangent, as the
    # derivation and the published code do -- the three coefficients are
    #
    #     coeff[i] = W[i][2, 1] + tan(q7) * W[i][2, 0],   W = A_w, B_w, C_w
    #
    # which blows up at q7 = +-90 deg, where the tangent is not finite.  There
    # the solver can end up with no solution at all for a perfectly reachable
    # configuration (measured: 2 failures in 1974 solves across the whole joint-7
    # range, both exactly at +-90 deg).
    #
    # Multiplying the whole equation by cos(q7) is exact -- it multiplies all
    # three coefficients by the same non-zero constant, so the roots are
    # unchanged -- and it removes the tangent:
    #
    #     coeff[i] = W[i][2, 1] * cos(q7) + W[i][2, 0] * sin(q7)
    #
    # which at q7 = 90 deg is simply W[i][2, 0], the correct limit.  Formula (1)
    # below is therefore a numerically robust rewriting of the same equation,
    # not a different method.  tests/test_solver.py checks that it agrees with
    # the tangent form everywhere the tangent form is well defined.
    sin_q7, cos_q7 = math.sin(q7), math.cos(q7)
    coeff = np.array(
        [
            A_w[2, 1] * cos_q7 + A_w[2, 0] * sin_q7,
            B_w[2, 1] * cos_q7 + B_w[2, 0] * sin_q7,
            C_w[2, 1] * cos_q7 + C_w[2, 0] * sin_q7,
        ]
    )
    norm = math.hypot(float(coeff[0]), float(coeff[1]))
    if norm < 1e-12:
        return None
    sine = -float(coeff[2]) / norm
    if abs(sine) > 1.0 + 1e-12:
        return None
    arc = math.asin(max(-1.0, min(1.0, sine)))
    delta = math.atan2(float(coeff[1]), float(coeff[0]))

    # The two roots of the sinusoid, eq. (STEP3) of docs/method.md.
    #
    # The coefficients above are the tangent form multiplied by cos(q7).  For
    # cos(q7) < 0 that multiplication negates all three, which sends
    # delta -> delta + pi and sine -> -sine, and the two roots trade places:
    # the root the published `ik_ca` (branch 1) returns would come out of
    # `phi_root = -1` instead of `+1`.  The solution *set* is unaffected -- both
    # roots are always enumerated -- but the labels would no longer mean what
    # they say, and the branch-by-branch comparison against `original/` would
    # only hold for q7 in (-90, 90) degrees.  Folding the sign of cos(q7) into
    # the choice keeps the labels stable; at q7 = +-90 deg both roots come from
    # the same sinusoid anyway, so either branch is correct there.
    take_first = (phi_root > 0) == (cos_q7 >= 0.0)
    phi = math.pi - delta - arc if take_first else -delta + arc

    # ---- step 4: finish the equivalent S-R-S solution ------------------------
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    r_03 = A_s * sin_phi + B_s * cos_phi + C_s
    r_47 = A_w * sin_phi + B_w * cos_phi + C_w

    q1 = math.atan2(r_03[1, 1], r_03[0, 1])
    q2 = math.acos(max(-1.0, min(1.0, float(r_03[2, 1]))))
    q3 = math.atan2(-r_03[2, 2], -r_03[2, 0])
    q5_direct = math.atan2(r_47[1, 2], r_47[0, 2])
    q6_positive = math.acos(max(-1.0, min(1.0, -float(r_47[2, 2]))))
    beta = wrist_offset_angle(geometry)

    # The wrist has a second configuration, (q5 + pi, -q6), and only one of the
    # two reaches the target.  The published code picks between them with
    # `criteria = r_47[2, 0] * cos(q7)`, which is a restatement of the sign of
    # cos(q7) inherited from tan(theta_7) = N/D -- and it therefore carries a
    # factor cos(q7) that vanishes at q7 = +-90 deg.  There the test degenerates:
    # measured on the joint-7 range, exactly two of 1974 solves failed, both at
    # +-90 deg, one of them returning a configuration with a pose residual of
    # 1.86 (it had flipped the wrist the wrong way) and one reporting no solution.
    #
    # This implementation therefore decides the wrist by what actually matters:
    # which of the two candidates reaches the pose.  That is one extra forward
    # kinematics call, and it is exact.  When `check_pose` is switched off the
    # published criterion is used, so that the two remain comparable.
    candidates = []
    for flipped in (False, True):
        wrist5 = q5_direct + (math.pi if flipped else 0.0)
        wrist6 = -q6_positive if flipped else q6_positive
        candidate = np.array(
            [q1, q2, q3, q4, wrist5, wrist6 - beta, q7]
        )
        if shoulder_flip:
            candidate = candidate.copy()
            candidate[0] = math.atan2(-math.sin(q1), -math.cos(q1))
            candidate[1] = -q2
            candidate[2] = math.atan2(-math.sin(q3), -math.cos(q3))
        candidates.append((flipped, candidate))

    error = math.inf
    if check_pose:
        scored = [
            (float(np.max(np.abs(model.fk_flange(candidate) - pose))), flipped, candidate)
            for flipped, candidate in candidates
        ]
        error, wrist_flipped, q = min(scored, key=lambda item: item[0])
    else:
        criteria = float(r_47[2, 0]) * math.cos(q7)
        wrist_flipped, q = candidates[0] if criteria >= 0.0 else candidates[1]

    wrapped, inside = wrap_to_limits(q)
    return IkSolution(
        q=wrapped,
        q4_root=q4_root,
        phi_root=phi_root,
        shoulder_flip=shoulder_flip,
        wrist_flipped=wrist_flipped,
        pose_error=error,
        within_limits=bool(np.all(inside)),
    )


def branch_solutions(
    pose: np.ndarray,
    q7: float,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    check_pose: bool = True,
) -> List[IkSolution]:
    """Evaluate all :data:`NUM_BRANCHES` branches of one pose.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 value in radians.
        geometry: Geometric parameters.
        check_pose: See :func:`solve_branch`.

    Returns:
        The candidates that exist, in a fixed order.  Branches that have no
        solution for this pose are omitted; nothing is deduplicated here.
    """
    results: List[IkSolution] = []
    for q4_root in (1, -1):
        for phi_root in (1, -1):
            for shoulder_flip in (False, True):
                solution = solve_branch(
                    pose,
                    q7,
                    q4_root=q4_root,
                    phi_root=phi_root,
                    shoulder_flip=shoulder_flip,
                    geometry=geometry,
                    check_pose=check_pose,
                )
                if solution is not None:
                    results.append(solution)
    return results


def solve(
    pose: np.ndarray,
    q7: float,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    tolerance: float = 1e-9,
    within_limits_only: bool = False,
    deduplicate: bool = True,
) -> List[IkSolution]:
    """All distinct configurations that reach ``pose`` with the requested joint 7.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 value in radians.
        geometry: Geometric parameters.
        tolerance: A branch counts as correct when its pose residual is below
            this.  Branches are evaluated analytically and then checked against
            the forward kinematics, so a wrong branch is dropped rather than
            returned.
        within_limits_only: Drop candidates with a joint outside its range.
        deduplicate: Merge candidates that describe the same configuration
            (branches can coincide at singular poses).

    Returns:
        The list of solutions, verified against the forward kinematics.
    """
    candidates = [
        solution
        for solution in branch_solutions(pose, q7, geometry=geometry)
        if solution.pose_error <= tolerance
    ]
    if within_limits_only:
        candidates = [solution for solution in candidates if solution.within_limits]
    if not deduplicate:
        return candidates
    return _deduplicate(candidates)


def solve_closest(
    pose: np.ndarray,
    q7: float,
    reference: Sequence[float],
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    within_limits_only: bool = True,
) -> Optional[IkSolution]:
    """The solution whose joints are closest to a reference configuration.

    Useful for tracking: the eight branches are discrete, so a controller that
    simply takes the first one will jump between them.  Choosing the branch
    nearest to the previous command keeps the motion continuous.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 value in radians.
        reference: Configuration to stay close to, typically the previous
            command, in radians.
        geometry: Geometric parameters.
        within_limits_only: See :func:`solve`.

    Returns:
        The closest solution, or ``None`` when the pose has none.
    """
    solutions = solve(
        pose, q7, geometry=geometry, within_limits_only=within_limits_only
    )
    if not solutions:
        return None
    reference = np.asarray(reference, dtype=float)
    return min(solutions, key=lambda item: _deviation(item.q, reference))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _skew(vector: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector."""
    x, y, z = (float(vector[0]), float(vector[1]), float(vector[2]))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest joint-wise angular difference, ignoring whole turns."""
    delta = a - b
    return float(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))))


def _deduplicate(
    solutions: Sequence[IkSolution], tolerance: float = 1e-6
) -> List[IkSolution]:
    """Keep one entry per distinct configuration, preferring the first."""
    unique: List[IkSolution] = []
    for solution in solutions:
        if any(_deviation(solution.q, kept.q) < tolerance for kept in unique):
            continue
        unique.append(solution)
    return unique
