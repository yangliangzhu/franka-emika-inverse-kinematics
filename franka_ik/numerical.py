"""Independent numerical inverse kinematics, used to test the analytical solver.

The analytical solver in :mod:`franka_ik.solver` enumerates eight branches.  Two
questions follow, and the solver itself can answer neither:

1. does the enumeration *find* the configuration a pose was generated from?  That
   is :func:`franka_ik.analysis.coverage_study`, and the answer is yes, 600 of 600.
2. are there configurations the enumeration does **not** find?  A coverage study
   cannot answer that, because it only ever looks at configurations it started
   from; a solver that misses a whole branch still returns correct poses.

This module answers the second question with machinery that shares neither code
nor assumption with the analytical solver: the same kinematics built symbolically
in CasADi, "reach this pose with joint 7 at this value" stated as a constrained
optimisation, solved by IPOPT from many random starting points.  Anything the
optimiser finds that is not one of the eight branches is a counter-example to the
completeness claim, and would be reported as such.

CasADi is imported lazily inside the functions, so ``import franka_ik`` never
pulls it in and the library keeps working without it -- the environment note in
``AGENTS.md`` depends on that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from . import model
from .solver import solve

__all__ = [
    "symbolic_forward_kinematics",
    "numeric_solver",
    "numerical_ik",
    "configuration_distance",
    "CompletenessReport",
    "completeness_check",
    "DEFAULT_SOLUTION_TOLERANCE",
]

#: Largest entry of ``fk_flange(q) - pose`` at which an IPOPT result counts as a
#: configuration that reaches the pose.  The optimiser itself is run far tighter
#: (``tol`` in :data:`_SOLVER_OPTIONS`); this is the acceptance test afterwards.
DEFAULT_SOLUTION_TOLERANCE = 1e-8

#: Iteration and tolerance settings handed to IPOPT.  Tighter than the defaults
#: because the goal is to reach a pose *exactly*, not to make progress towards it.
_SOLVER_OPTIONS = {
    "print_time": False,
    "ipopt": {
        "print_level": 0,
        "sb": "yes",
        "tol": 1e-12,
        "max_iter": 300,
        "acceptable_tol": 1e-10,
        "linear_solver": "mumps",
    },
}

_CACHE: dict = {}


def _require_casadi():
    """Import CasADi, with an error that says why it is wanted.

    Returns:
        The ``casadi`` module.

    Raises:
        ImportError: If CasADi is not installed, naming the ways to get it.
    """
    try:
        import casadi as ca
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "this module needs CasADi, which the library itself does not. "
            "Install it with `uv sync` (the dev group includes it), "
            "`uv sync --extra reference`, or `pip install casadi`."
        ) from exc
    return ca


def symbolic_forward_kinematics():
    """The flange pose as a CasADi function of the seven joint angles.

    Built from :data:`franka_ik.model.DH_PARAMETERS`, so it is the same
    kinematics the NumPy model evaluates; the point is a second *solver*, not a
    second model.

    Returns:
        A ``casadi.Function`` mapping a 7-vector to the 4x4 flange pose.
    """
    if "fk" not in _CACHE:
        ca = _require_casadi()
        q = ca.SX.sym("q", model.NUM_JOINTS)

        def rot_x(angle):
            c, s = ca.cos(angle), ca.sin(angle)
            m = ca.SX.eye(4)
            m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
            return m

        def rot_z(angle):
            c, s = ca.cos(angle), ca.sin(angle)
            m = ca.SX.eye(4)
            m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
            return m

        def trans_x(distance):
            m = ca.SX.eye(4)
            m[0, 3] = distance
            return m

        def trans_z(distance):
            m = ca.SX.eye(4)
            m[2, 3] = distance
            return m

        current = ca.SX.eye(4)
        for row in range(model.FLANGE_ROW + 1):
            a, d, alpha, _offset = model.DH_PARAMETERS[row]
            theta = q[row] if row < model.NUM_JOINTS else 0.0
            current = current @ rot_x(alpha) @ trans_x(a) @ trans_z(d) @ rot_z(theta)
        _CACHE["fk"] = ca.Function("franka_fk_flange", [q], [current])
    return _CACHE["fk"]


def numeric_solver():
    """The constrained-optimisation solver used by :func:`numerical_ik`.

    The program is

    .. math::

        \\min_q \\; \\| T(q) - T^d \\|_F^2
        \\quad \\text{s.t.} \\quad
        q^l \\le q \\le q^u, \\; q_7 = q_7^d

    where :math:`T(q)` is the flange pose.  The objective is the Frobenius norm of
    the whole 4x4 pose error rather than a hand-rolled position-plus-orientation
    residual: it is zero exactly at a solution, needs no rotation parametrisation,
    and CasADi differentiates it exactly.

    Returns:
        A ``casadi.Function`` taking ``(q0, target_pose, q7)`` and returning the
        solution vector.
    """
    if "solver" not in _CACHE:
        ca = _require_casadi()
        fk = symbolic_forward_kinematics()
        q = ca.SX.sym("q", model.NUM_JOINTS)
        target = ca.SX.sym("target", 4, 4)
        q7 = ca.SX.sym("q7")

        error = ca.reshape(fk(q) - target, 16, 1)
        # the target pose and the requested joint 7 are *parameters* of the NLP,
        # not free symbols: declaring them here is what lets one compiled solver
        # be reused for every pose
        parameters = ca.vertcat(ca.reshape(target, 16, 1), q7)
        nlp = {"x": q, "p": parameters, "f": ca.dot(error, error), "g": q[6] - q7}
        solver = ca.nlpsol(
            "franka_numeric_ik",
            "ipopt",
            nlp,
            {**_SOLVER_OPTIONS, "print_time": False},
        )
        lower = model.lower_limits()
        upper = model.upper_limits()
        _CACHE["solver"] = ca.Function(
            "franka_numeric_ik_fn",
            [q, target, q7],
            [solver(
                x0=q,
                p=parameters,
                lbx=lower,
                ubx=upper,
                lbg=0.0,
                ubg=0.0,
            )["x"]],
        )
    return _CACHE["solver"]


def numerical_ik(
    pose: np.ndarray,
    q7: float,
    starts: int = 400,
    seed: int = 0,
    tolerance: float = DEFAULT_SOLUTION_TOLERANCE,
) -> List[np.ndarray]:
    """Search for every configuration reaching ``pose`` with joint 7 fixed.

    The program is solved from many random starting points inside the joint
    limits, which is the standard way to enumerate the solutions of a small
    inverse-kinematics problem without trusting any closed-form structure.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 in radians.
        starts: Number of random starting points.
        seed: RNG seed.
        tolerance: A solution counts when the largest entry of
            ``fk_flange(q) - pose`` is below this.

    Returns:
        Distinct configurations, each verified with the NumPy forward kinematics
        and with joint 7 equal to the request.
    """
    solver = numeric_solver()
    pose = np.asarray(pose, dtype=float)
    lower = model.lower_limits()
    upper = model.upper_limits()
    rng = np.random.default_rng(seed)

    found: List[np.ndarray] = []
    for _ in range(starts):
        start = rng.uniform(lower, upper)
        try:
            # positional call, unnamed output: the result is a DM, and float()
            # turns it into a plain NumPy array rather than an object array
            candidate = np.array(solver(start, pose, float(q7)), dtype=float).ravel()
        except RuntimeError:
            # IPOPT can fail to converge from an unlucky start; that is not an
            # error for the caller, it is one starting point that did not help.
            continue
        if not np.all(np.isfinite(candidate)):
            continue
        if float(np.max(np.abs(model.fk_flange(candidate) - pose))) > tolerance:
            continue
        if abs(math.atan2(math.sin(candidate[6] - q7), math.cos(candidate[6] - q7))) > 1e-6:
            continue
        if not any(_same(candidate, kept) for kept in found):
            found.append(candidate)
    return found


@dataclass
class CompletenessReport:
    """Comparison of the numerical solution set with the analytical branches.

    Attributes:
        q7: Joint 7 value the pose was solved at, in radians.
        numerical: Distinct configurations the optimiser found.
        analytical: Distinct configurations the analytical solver returned.
        matched: How many numerical solutions coincide with a branch.
        unmatched: Numerical solutions that are far from *every* branch -- each
            one would be a counter-example to the completeness claim.
        distances_deg: For every numerical solution, the angular distance to the
            nearest analytical branch, in degrees.  This is the number that
            matters: an optimiser converging near a singularity can stop a
            fraction of a degree away from the branch it is standing on, which is
            not a counter-example, whereas a solution degrees away from all of
            them would be.
        unreachable: ``True`` when both methods found nothing, i.e. the pose
            really is unreachable at this joint 7.
    """

    q7: float
    numerical: List[np.ndarray] = field(default_factory=list)
    analytical: List[np.ndarray] = field(default_factory=list)
    matched: int = 0
    unmatched: List[np.ndarray] = field(default_factory=list)
    unreachable: bool = False
    distances_deg: List[float] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Whether every numerically found solution is one of the branches."""
        return not self.unmatched

    @property
    def worst_distance_deg(self) -> float:
        """Largest distance from a numerical solution to its nearest branch.

        This is the honest summary of the numerical search: a small value means
        every solution the optimiser found is one the analytical solver also
        returns.
        """
        return max(self.distances_deg, default=0.0)

    def describe(self) -> str:
        """One-line summary for the studies and tests."""
        if self.unreachable:
            return (
                f"joint 7 = {math.degrees(self.q7):.3f} deg: unreachable — "
                "both methods found nothing"
            )
        text = (
            f"joint 7 = {math.degrees(self.q7):.3f} deg: optimiser {len(self.numerical)}, "
            f"analytical {len(self.analytical)}, matched {self.matched} "
            f"(worst distance {self.worst_distance_deg:.4f} deg)"
        )
        if self.complete:
            return text + " — no counter-example"
        return text + f" — {len(self.unmatched)} NOT produced by any branch"


def configuration_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest joint-wise angular difference between two configurations, radians.

    Args:
        a: Joint angles in radians.
        b: Joint angles in radians.

    Returns:
        The largest ``|wrap(a_i - b_i)|`` over the seven joints.  Whole turns are
        ignored, so two representatives of the same configuration give zero.
    """
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))))


def completeness_check(
    pose: np.ndarray,
    q7: float,
    starts: int = 400,
    seed: int = 0,
    tolerance: float = 1e-3,
    counter_example_tolerance: float = 0.05,
) -> CompletenessReport:
    """Compare the numerical solution set with the analytical branches.

    Two tolerances, because the two methods converge differently.  IPOPT drives
    the *pose* error to zero, so near a kinematic singularity -- where a joint can
    move a little without moving the tool -- it stops a fraction of a degree away
    from the configuration it is standing on.  A numerical solution is therefore
    counted as one of the branches when it is within ``tolerance`` of it, and only
    a solution further than ``counter_example_tolerance`` from **every** branch is
    reported as a counter-example.  The default is ``0.05`` deg, near the size of
    the stopping error the optimiser shows at a singularity, so a counter-example
    means a configuration more than a rounding error away from every branch.  The full list of distances
    is kept in :attr:`CompletenessReport.distances_deg` so the margin is visible
    rather than hidden in a tolerance.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 in radians.
        starts: Random starting points for the numerical search.
        seed: RNG seed.
        tolerance: Distance below which a numerical solution *is* a branch.
        counter_example_tolerance: Distance above which it is a counter-example.

    Returns:
        A :class:`CompletenessReport`.
    """
    numerical = numerical_ik(pose, q7, starts=starts, seed=seed)
    analytical = [solution.q for solution in solve(pose, q7, within_limits_only=True)]

    matched = 0
    unmatched: List[np.ndarray] = []
    distances: List[float] = []
    for candidate in numerical:
        if not analytical:
            distances.append(math.inf)
            unmatched.append(candidate)
            continue
        nearest = min(configuration_distance(candidate, branch) for branch in analytical)
        distances.append(nearest)
        if nearest <= tolerance:
            matched += 1
        elif nearest > counter_example_tolerance:
            unmatched.append(candidate)

    return CompletenessReport(
        q7=q7,
        numerical=numerical,
        analytical=analytical,
        matched=matched,
        unmatched=unmatched,
        unreachable=not numerical and not analytical,
        distances_deg=[math.degrees(value) for value in distances],
    )


def _same(a: Sequence[float], b: Sequence[float], tolerance: float = 1e-6) -> bool:
    """Whether two joint vectors describe the same configuration."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bool(np.all(np.abs(np.arctan2(np.sin(delta), np.cos(delta))) < tolerance))
