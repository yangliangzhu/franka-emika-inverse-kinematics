"""The eight branches -- and which of them the published solver throws away.

Four independent two-way choices appear in the derivation: ``q4_root`` (which root
of the elbow quadratic, i.e. which side of the shoulder-wrist line the elbow is
on), ``phi_root`` (which equivalent arm angle reproduces the requested joint 7),
``shoulder_flip`` (``(q1, q2, q3) -> (q1 + pi, -q2, q3 + pi)``) and the internal
``wrist_flip``.  That is eight candidate configurations.  The published solver kept
only the ``+`` root of the quadratic (``PUBLISHED_Q4_ROOT``), so it enumerates
four -- all correct, but not all of them.

For the default pose all eight branches exist and give eight *distinct* in-limit
solutions; the published subset returns four of them and misses the configuration
the pose was generated from, which only a ``q4-`` branch recovers.  The example
closes with :func:`franka_ik.solver.solve_closest`, which returns the branch
nearest a reference configuration instead of the first one -- the tool a tracking
controller needs, since the branches are discrete.

Run from the repository root::

    MPLBACKEND=Agg python3 examples/03_branches.py --save-dir /tmp/branches
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import (  # noqa: E402
    NUM_BRANCHES,
    fk_flange,
    lower_limits,
    solve,
    solve_closest,
    upper_limits,
)

# ``branch_solutions`` is re-exported; the published root constant and the per-pose
# study are only in ``franka_ik.analysis``.
from franka_ik.analysis import PUBLISHED_Q4_ROOT, study_pose  # noqa: E402
from franka_ik.solver import IkSolution, branch_solutions  # noqa: E402

LOGGER = logging.getLogger("examples.03_branches")

#: Default seed: eight distinct in-limit solutions, all eight branches present, and
#: a target that only a ``q4-`` branch recovers.
DEFAULT_SEED = 53

#: Reference configuration for ``solve_closest``, in degrees.
REFERENCE_DEG = np.array([0.0, -30.0, 0.0, -60.0, 0.0, 90.0, 0.0])


def import_pyplot(headless: bool) -> Any:
    """Import ``matplotlib.pyplot``, selecting a backend before it is imported.

    Args:
        headless: Use ``Agg`` so that saving works without a display.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finish(figure: Any, save_dir: Optional[Path], stem: str) -> Optional[Path]:
    """Save the figure when ``--save-dir`` was given, otherwise show it.

    Args:
        figure: The figure to render.
        save_dir: Destination directory, or ``None`` for an interactive window.
        stem: Figure file name without extension.

    Returns:
        The written path, or ``None`` when the figure was shown instead.
    """
    plt = import_pyplot(save_dir is not None)
    if save_dir is None:
        plt.show()
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{stem}.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    LOGGER.info("wrote %s", path)
    return path


def deviation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest joint-wise angular difference in degrees, ignoring whole turns.

    Args:
        a: First configuration in radians.
        b: Second configuration in radians.

    Returns:
        The deviation in degrees.
    """
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.degrees(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))))


def print_branches(target: np.ndarray, solutions: Sequence[IkSolution]) -> List[bool]:
    """Print every branch through ``IkSolution.describe`` and mark the target.

    Args:
        target: The configuration the pose was generated from, in radians.
        solutions: The branches that exist for this pose.

    Returns:
        One flag per branch, true where it recovers ``target``.
    """
    print(f"\nAll branches that exist for this pose ({len(solutions)} of {NUM_BRANCHES}):")
    matches = [deviation(solution.q, target) < 1e-6 for solution in solutions]
    for solution, match in zip(solutions, matches):
        print(f"  {solution.describe()}{'  <-- TARGET CONFIGURATION' if match else ''}")
    return matches


def print_solve_closest(pose: np.ndarray, q7: float, target: np.ndarray) -> None:
    """Show ``solve_closest`` choosing the branch nearest a reference.

    Args:
        pose: The 4x4 flange pose.
        q7: Requested joint 7 in radians.
        target: The configuration the pose came from, in radians.
    """
    reference = np.radians(REFERENCE_DEG)
    solutions = solve(pose, q7, within_limits_only=True)
    chosen = solve_closest(pose, q7, reference, within_limits_only=True)
    if chosen is None:
        raise RuntimeError("the pose has no in-limit solution")
    print(f"\nsolve_closest against a reference of {REFERENCE_DEG} deg:")
    print(f"  {'branch':>16} {'deviation [deg]':>16} {'pose error':>12}   note")
    for solution in sorted(solutions, key=lambda item: deviation(item.q, reference)):
        notes = []
        # solve_closest builds its own copies, so compare configurations, not ids.
        if deviation(solution.q, chosen.q) < 1e-9:
            notes.append("chosen by solve_closest")
        if deviation(solution.q, target) < 1e-6:
            notes.append("is the target")
        print(f"  {solution.label:>16} {deviation(solution.q, reference):>16.3f} "
              f"{solution.pose_error:>12.2e}   {', '.join(notes)}")
    print(f"  the naive 'first solution' rule would answer {solutions[0].label} at "
          f"{deviation(solutions[0].q, reference):.3f} deg away")


def build_figure(plt: Any, target: np.ndarray, solutions: Sequence[IkSolution]) -> Any:
    """Draw the branch configurations and the target as a heat map.

    Args:
        plt: The imported ``matplotlib.pyplot`` module.
        target: The configuration the pose came from, in radians.
        solutions: The branches that exist for this pose.

    Returns:
        The matplotlib figure.
    """
    figure, axes = plt.subplots(figsize=(9.0, 4.2))
    rows = [np.degrees(solution.q) for solution in solutions] + [np.degrees(target)]
    labels = [
        solution.label + (" (published)" if solution.q4_root == PUBLISHED_Q4_ROOT else "")
        for solution in solutions
    ] + ["target configuration"]
    image = axes.imshow(np.asarray(rows), aspect="auto", cmap="twilight_shifted")
    axes.set_xticks(range(7), [f"q{index}" for index in range(1, 8)])
    axes.set_yticks(range(len(rows)), labels, fontsize=8)
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            axes.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=7)
    axes.set_title("branch configurations in degrees (one pose, eight solutions)")
    figure.colorbar(image, ax=axes, label="joint angle [deg]")
    figure.tight_layout()
    return figure


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (see ``--help``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=None, help="write the figure here")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed of the pose ({DEFAULT_SEED} has eight solutions)")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO or DEBUG")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Enumerate the branches of one pose and draw the figure.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    target = np.random.default_rng(args.seed).uniform(lower_limits(), upper_limits())
    q7 = float(target[6])
    pose = fk_flange(target)
    print("The eight branches of the analytical inverse kinematics")
    print(f"seed = {args.seed}, target configuration = {np.round(np.degrees(target), 3)} deg")
    print(f"joint 7 prescribed to {np.degrees(q7):.6f} deg, "
          f"flange position {np.round(pose[:3, 3], 6)} m")

    solutions = branch_solutions(pose, q7)
    matches = print_branches(target, solutions)
    study = study_pose(pose, q7, target=target)
    published_recovers = any(
        match and solution.q4_root == PUBLISHED_Q4_ROOT
        for match, solution in zip(matches, solutions)
    )
    print("\nSummary from franka_ik.analysis.study_pose:")
    print(f"  distinct in-limit solutions : {study.n_distinct}")
    print(f"  published subset (q4_root == +{PUBLISHED_Q4_ROOT}) : {study.n_published}")
    print(f"  branches recovering the target configuration: {sum(matches)}")
    if not published_recovers:
        print("  the published subset MISSES the configuration the pose came from; its four")
        print("  answers are all correct, they are just not all of them")
    print("  largest pose residual over the branches: "
          f"{max(solution.pose_error for solution in solutions):.3e}")

    print_solve_closest(pose, q7, target)
    plt = import_pyplot(args.save_dir is not None)
    finish(build_figure(plt, target, solutions), args.save_dir, "03_branches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
