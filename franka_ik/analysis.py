"""Studies of the solution set: how many solutions exist, and are they all found?

This module turns the open question of the repository -- *how many inverse
kinematic solutions does the Panda have, and does the published solver find
them?* -- into measurements that can be re-run and re-checked.

The two studies it provides answer two different questions:

:func:`coverage_study`
    For randomly sampled configurations, is the configuration itself among the
    solutions the solver returns?  A solver that misses a branch fails this even
    though every solution it does return is correct -- which is exactly the
    failure mode of a solver that takes one root of a quadratic.

:func:`solution_count_study`
    How many distinct configurations actually reach the same pose with the same
    joint 7?  This is a property of the arm, not of the solver, so it is the
    number the solver's branch count should be compared against.

Both are also exposed as a tiny command line in ``scripts/study_branches.py``,
and the numbers they produce are what ``docs/branch_analysis.md`` reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import model
from .geometry import PAPER_GEOMETRY, EquivalentGeometry, q4_discriminant
from .solver import IkSolution, branch_solutions, solve

__all__ = [
    "BranchOutcome",
    "PoseStudy",
    "study_pose",
    "CoverageReport",
    "coverage_study",
    "SolutionCountReport",
    "solution_count_study",
    "reachable_distance_range",
    "classify_failure",
    "PUBLISHED_Q4_ROOT",
]

#: The elbow-quadratic root the originally published four-branch solver kept.
#: ``q4_roots`` returns the ``+`` root first, so restricting to ``q4_root == +1``
#: reproduces the published branch set exactly.  ``tests/test_branches.py``
#: checks that claim against the original CasADi implementation branch by branch.
PUBLISHED_Q4_ROOT = 1


@dataclass
class BranchOutcome:
    """What one branch did for one pose.

    Attributes:
        label: Branch identifier, e.g. ``"q4+ phi- flip"``.
        exists: Whether the branch produced a configuration at all.
        pose_error: Residual against the target pose, ``inf`` when it does not
            exist.
        within_limits: Whether the configuration respects the joint limits.
        recovers_target: Whether it reproduces the target configuration (only
            meaningful when a target is known).
    """

    label: str
    exists: bool
    pose_error: float = math.inf
    within_limits: bool = False
    recovers_target: bool = False


@dataclass
class PoseStudy:
    """Per-branch detail for a single pose.

    Attributes:
        q7: The joint 7 value the pose was solved at, in radians.
        outcomes: One :class:`BranchOutcome` per branch.
        n_distinct: Number of distinct in-limit configurations found.
        n_published: How many of those the published four-branch solver finds.
    """

    q7: float
    outcomes: List[BranchOutcome] = field(default_factory=list)
    n_distinct: int = 0
    n_published: int = 0

    @property
    def exists_count(self) -> int:
        """How many branches produced a configuration."""
        return sum(1 for outcome in self.outcomes if outcome.exists)

    def describe(self, precision: int = 3) -> str:
        """Multi-line summary, one line per branch."""
        lines = [f"joint 7 = {math.degrees(self.q7):.3f} deg"]
        for outcome in self.outcomes:
            if not outcome.exists:
                lines.append(f"  {outcome.label:16s} no solution for this pose")
                continue
            flags = []
            if outcome.within_limits:
                flags.append("in limits")
            if outcome.recovers_target:
                flags.append("TARGET")
            lines.append(
                f"  {outcome.label:16s} pose error {outcome.pose_error:.2e}   "
                + ", ".join(flags)
            )
        lines.append(
            f"  distinct in-limit solutions: {self.n_distinct} "
            f"(published four-branch subset finds {self.n_published})"
        )
        return "\n".join(lines)


def study_pose(
    pose: np.ndarray,
    q7: float,
    target: Optional[Sequence[float]] = None,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    tolerance: float = 1e-9,
) -> PoseStudy:
    """Evaluate every branch for one pose and summarise what each one produced.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 in radians.
        target: The configuration the pose came from, when it is known.  Enables
            the ``recovers_target`` flag.
        geometry: Geometric parameters.
        tolerance: Pose residual below which a branch counts as correct.

    Returns:
        A :class:`PoseStudy`.
    """
    outcomes: List[BranchOutcome] = []
    valid: List[IkSolution] = []
    published: List[IkSolution] = []

    for solution in branch_solutions(pose, q7, geometry=geometry):
        recovers = False
        if target is not None:
            delta = solution.q - np.asarray(target, dtype=float)
            recovers = bool(
                np.all(np.abs(np.arctan2(np.sin(delta), np.cos(delta))) < 1e-6)
            )
        outcomes.append(
            BranchOutcome(
                label=solution.label,
                exists=True,
                pose_error=solution.pose_error,
                within_limits=solution.within_limits,
                recovers_target=recovers,
            )
        )
        if solution.pose_error <= tolerance and solution.within_limits:
            valid.append(solution)
            if solution.q4_root == PUBLISHED_Q4_ROOT:
                published.append(solution)

    for label in _all_labels():
        if not any(outcome.label == label for outcome in outcomes):
            outcomes.append(BranchOutcome(label=label, exists=False))
    outcomes.sort(key=lambda outcome: outcome.label)

    return PoseStudy(
        q7=q7,
        outcomes=outcomes,
        n_distinct=len(_distinct(valid)),
        n_published=len(_distinct(published)),
    )


@dataclass
class CoverageReport:
    """Result of :func:`coverage_study`.

    Attributes:
        samples: Number of configurations sampled.
        recovered_full: How many were found by the full branch set.
        recovered_published: How many were found when restricted to the elbow
            root the published solver keeps.
        mean_solutions: Average number of distinct in-limit solutions per pose.
        min_solutions: Smallest number of distinct solutions seen.
        max_solutions: Largest number of distinct solutions seen.
    """

    samples: int
    recovered_full: int
    recovered_published: int
    mean_solutions: float
    min_solutions: int
    max_solutions: int

    @property
    def full_rate(self) -> float:
        """Fraction of configurations recovered by the full branch set."""
        return self.recovered_full / self.samples if self.samples else 0.0

    @property
    def published_rate(self) -> float:
        """Fraction recovered when restricted to the published elbow root."""
        return self.recovered_published / self.samples if self.samples else 0.0

    def describe(self, precision: int = 3) -> str:
        """Report used by the study script and the documentation."""
        return (
            f"{self.samples} random configurations\n"
            f"  published four-branch subset recovers the target: "
            f"{self.recovered_published}/{self.samples} "
            f"({100 * self.published_rate:.1f} %)\n"
            f"  full eight-branch solver recovers the target:     "
            f"{self.recovered_full}/{self.samples} "
            f"({100 * self.full_rate:.1f} %)\n"
            f"  distinct in-limit solutions per pose: mean {self.mean_solutions:.2f}, "
            f"range {self.min_solutions}-{self.max_solutions}"
        )


def coverage_study(
    samples: int = 300,
    seed: int = 0,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    use_real_limits: bool = True,
) -> CoverageReport:
    """Sample configurations and measure how often the solver finds them back.

    Args:
        samples: Number of configurations.
        seed: RNG seed.
        geometry: Geometric parameters.
        use_real_limits: Sample uniformly inside the real joint limits.  With
            ``False`` the sample is normal-distributed and wrapped, which
            concentrates it near the middle of the range.

    Returns:
        A :class:`CoverageReport`.
    """
    rng = np.random.default_rng(seed)
    lower, upper = model.lower_limits(), model.upper_limits()

    recovered_full = 0
    recovered_published = 0
    counts: List[int] = []

    for _ in range(samples):
        if use_real_limits:
            target = rng.uniform(lower, upper)
        else:
            target = np.arctan2(np.sin(rng.normal(size=7)), np.cos(rng.normal(size=7)))
            target = np.clip(target, lower, upper)
        pose = model.fk_flange(target)
        q7 = float(target[6])

        solutions = solve(pose, q7, geometry=geometry, within_limits_only=True)
        counts.append(len(solutions))

        if any(_same_configuration(solution.q, target) for solution in solutions):
            recovered_full += 1
        if any(
            solution.q4_root == PUBLISHED_Q4_ROOT
            and _same_configuration(solution.q, target)
            for solution in solutions
        ):
            recovered_published += 1

    return CoverageReport(
        samples=samples,
        recovered_full=recovered_full,
        recovered_published=recovered_published,
        mean_solutions=float(np.mean(counts)) if counts else 0.0,
        min_solutions=int(min(counts)) if counts else 0,
        max_solutions=int(max(counts)) if counts else 0,
    )


@dataclass
class SolutionCountReport:
    """Histogram of how many solutions a pose has.

    Attributes:
        samples: Number of poses sampled.
        histogram: Mapping from solution count to how many poses had it.
        mean: Mean solution count.
    """

    samples: int
    histogram: Dict[int, int]
    mean: float

    def describe(self) -> str:
        """Report used by the study script and the documentation."""
        lines = [
            f"distinct in-limit solutions for {self.samples} random poses "
            f"(mean {self.mean:.2f}):"
        ]
        for count in sorted(self.histogram):
            bar = "#" * int(round(60 * self.histogram[count] / self.samples))
            lines.append(f"  {count} solution(s): {self.histogram[count]:5d}  {bar}")
        return "\n".join(lines)


def solution_count_study(
    samples: int = 300,
    seed: int = 0,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
) -> SolutionCountReport:
    """Measure the distribution of the number of solutions per pose.

    Args:
        samples: Number of poses.
        seed: RNG seed.
        geometry: Geometric parameters.

    Returns:
        A :class:`SolutionCountReport`.
    """
    rng = np.random.default_rng(seed)
    lower, upper = model.lower_limits(), model.upper_limits()
    histogram: Dict[int, int] = {}
    total = 0
    for _ in range(samples):
        target = rng.uniform(lower, upper)
        pose = model.fk_flange(target)
        count = len(solve(pose, float(target[6]), geometry=geometry, within_limits_only=True))
        histogram[count] = histogram.get(count, 0) + 1
        total += count
    return SolutionCountReport(
        samples=samples, histogram=histogram, mean=total / samples if samples else 0.0
    )


def reachable_distance_range(
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
) -> Tuple[float, float]:
    """Range of ``||x_sw||`` for which the elbow quadratic has real roots.

    The discriminant is a quadratic in :math:`D = \\|x_{sw}\\|^2`,

    .. math::

        \\operatorname{disc}(D) = -D^2 + (A + B + C)D - AB, \\qquad
        A = (d_{se}+d_{ew})^2,\\quad B = (d_{se}-d_{ew})^2,\\quad C = 4b^2,

    so the reachable shell is available in closed form.  Setting the bias to zero
    recovers the familiar S-R-S shell ``[|d_se - d_ew|, d_se + d_ew]``.

    The bias *widens* the shell on both ends: for the Panda the inner radius
    drops from 0.068 m to 0.0662 m and the outer rises from 0.700 m to 0.7194 m,
    because the offset lets the elbow fold slightly further in and reach slightly
    further out.

    Args:
        geometry: Geometric parameters.

    Returns:
        ``(minimum, maximum)`` of ``||x_sw||`` in metres.
    """
    outer = (geometry.d_se + geometry.d_ew) ** 2
    inner = (geometry.d_se - geometry.d_ew) ** 2
    bias_term = 4.0 * geometry.bias**2

    # -D^2 + (outer + inner + bias_term) D - outer*inner = 0
    linear = outer + inner + bias_term
    constant = outer * inner
    discriminant = linear * linear - 4.0 * constant
    if discriminant < 0.0:  # pragma: no cover - cannot happen for real geometry
        return (math.nan, math.nan)
    root = math.sqrt(discriminant)
    low = max(0.5 * (linear - root), 0.0)
    high = 0.5 * (linear + root)
    return (math.sqrt(low), math.sqrt(high))


def classify_failure(
    pose: np.ndarray,
    q7: float,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
) -> str:
    """Say why a pose has no solution, when it has none.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Requested joint 7 in radians.
        geometry: Geometric parameters.

    Returns:
        ``"solvable"``, or one of ``"shoulder_wrist_distance_zero"``,
        ``"outside_reachable_shell"``, ``"arm_angle_singular"`` and
        ``"no_valid_branch"``.  Every one of those corresponds to a ``return
        None`` in :func:`franka_ik.solver.solve_branch`, so a reader can trace it.

    Raises:
        ValueError: If ``pose`` or ``q7`` contains a non-finite value.  Every
            *finite* input is classified, however absurd it is -- poses with
            entries of 1e300 and non-orthonormal rotations included -- but a
            ``nan`` cannot be classified, and propagating one silently is exactly
            the failure mode that made the published ``limit_joints`` hang.  The
            guard lives in :func:`franka_ik.solver.wrap_to_limits`.
    """
    from .geometry import shoulder_to_wrist, wrist_correction

    rotation = np.asarray(pose, dtype=float)[:3, :3]
    position = np.asarray(pose, dtype=float)[:3, 3]
    corrected = rotation @ wrist_correction(q7, geometry)
    p_sw = shoulder_to_wrist(position, corrected, geometry)
    distance_squared = float(p_sw @ p_sw)
    if distance_squared <= 1e-18:
        return "shoulder_wrist_distance_zero"
    if q4_discriminant(distance_squared, geometry) < 0.0:
        return "outside_reachable_shell"
    if not branch_solutions(pose, q7, geometry=geometry):
        return "arm_angle_singular"
    if not solve(pose, q7, geometry=geometry, within_limits_only=True):
        return "no_valid_branch"
    return "solvable"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _same_configuration(a: np.ndarray, b: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Whether two joint vectors describe the same configuration."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bool(np.all(np.abs(np.arctan2(np.sin(delta), np.cos(delta))) < tolerance))


def _distinct(solutions: Sequence[IkSolution]) -> List[IkSolution]:
    """Deduplicate solutions by configuration."""
    unique: List[IkSolution] = []
    for solution in solutions:
        if any(_same_configuration(solution.q, kept.q) for kept in unique):
            continue
        unique.append(solution)
    return unique


def _all_labels() -> List[str]:
    """Every branch label, in a stable order."""
    labels = []
    for q4_root in ("+", "-"):
        for phi_root in ("+", "-"):
            for flip in ("plain", "flip"):
                labels.append(f"q4{q4_root} phi{phi_root} {flip}")
    return labels
