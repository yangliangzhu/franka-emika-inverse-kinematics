"""How many solutions does the Panda have, and does the solver find them all?

The two studies of :mod:`franka_ik.analysis` answer two different questions and
this example runs both on the same seeded sample:

:func:`franka_ik.analysis.coverage_study`
    For each random configuration, is *that configuration* among the solutions the
    solver returns?  A solver that misses a branch fails this even when every answer
    it does return is correct -- the failure mode of taking one root of the elbow
    quadratic.
:func:`franka_ik.analysis.solution_count_study`
    How many distinct in-limit configurations reach the same pose with the same
    joint 7?  A property of the arm, not of the solver.

The example also counts what the published four-branch subset finds for the same
poses, so the two distributions can be compared directly, and reports the
reachable shell of ``||x_sw||``, one failure classification and the cost of a full
solve.  The figure is the histogram plus the recovery-rate comparison.

Run from the repository root::

    MPLBACKEND=Agg python3 examples/05_coverage_study.py --save-dir /tmp/coverage
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import (  # noqa: E402
    PAPER_GEOMETRY,
    EquivalentGeometry,
    fk_flange,
    lower_limits,
    reachable_distance_range,
    solve,
    upper_limits,
)
from franka_ik.analysis import (  # noqa: E402
    PUBLISHED_Q4_ROOT,
    CoverageReport,
    classify_failure,
    coverage_study,
    solution_count_study,
)

LOGGER = logging.getLogger("examples.05_coverage_study")


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


def unpublished_counts(samples: int, seed: int) -> Dict[int, int]:
    """Count, per pose, the solutions the published four-branch subset finds.

    The sampling matches :func:`franka_ik.analysis.solution_count_study` at the same
    seed, so the two histograms describe the same poses.

    Args:
        samples: Number of poses.
        seed: RNG seed.

    Returns:
        Mapping from solution count to how many poses had it.
    """
    rng = np.random.default_rng(seed)
    lower, upper = lower_limits(), upper_limits()
    histogram: Dict[int, int] = {}
    for _ in range(samples):
        target = rng.uniform(lower, upper)
        solutions = solve(fk_flange(target), float(target[6]), within_limits_only=True)
        count = sum(1 for solution in solutions if solution.q4_root == PUBLISHED_Q4_ROOT)
        histogram[count] = histogram.get(count, 0) + 1
    return histogram


def print_reachable_shell() -> None:
    """Print the reachable shell of ``||x_sw||`` with and without the bias."""
    lower, upper = reachable_distance_range()
    unbiased = EquivalentGeometry(
        d_bs=PAPER_GEOMETRY.d_bs,
        d_se=PAPER_GEOMETRY.d_se,
        d_ew=PAPER_GEOMETRY.d_ew,
        d_wt=PAPER_GEOMETRY.d_wt,
        offset=PAPER_GEOMETRY.offset,
        bias=0.0,
    )
    srs_lower, srs_upper = reachable_distance_range(unbiased)
    print("\nReachable shell of ||x_sw|| (where the elbow discriminant is non-negative):")
    print(f"  with the shoulder bias : {lower:.4f} m to {upper:.4f} m")
    print(f"  bias = 0, plain S-R-S  : {srs_lower:.4f} m to {srs_upper:.4f} m "
          f"(|d_se - d_ew| to d_se + d_ew)")
    print("  the bias widens the shell at both ends: with it the elbow folds slightly")
    print("  further in and reaches slightly further out")


def print_failure_classification() -> None:
    """Show ``classify_failure`` naming why a pose cannot be solved."""
    rng = np.random.default_rng(0)
    target = rng.uniform(lower_limits(), upper_limits())
    pose = fk_flange(target)
    far = pose.copy()
    far[2, 3] += 1.0
    print("\nclassify_failure on a solvable pose and on one 1 m too far away:")
    print(f"  the sampled pose               -> {classify_failure(pose, float(target[6]))}")
    print(f"  the same pose shifted 1 m in z -> {classify_failure(far, float(target[6]))}")


def build_figure(
    plt: Any, full: Dict[int, int], published: Dict[int, int], report: CoverageReport
) -> Any:
    """Draw the two histograms and the recovery-rate comparison.

    Args:
        plt: The imported ``matplotlib.pyplot`` module.
        full: Histogram of the full eight-branch solver.
        published: Histogram of the published four-branch subset.
        report: The coverage report.

    Returns:
        The matplotlib figure.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    counts = sorted(set(full) | set(published))
    positions, width = np.arange(len(counts)), 0.4
    left.bar(positions - width / 2, [full.get(count, 0) for count in counts], width,
             label="full eight-branch solver")
    left.bar(positions + width / 2, [published.get(count, 0) for count in counts], width,
             label="published four-branch subset")
    left.set_xticks(positions, [str(count) for count in counts])
    left.set_xlabel("distinct in-limit solutions for one pose")
    left.set_ylabel("number of poses")
    left.set_title(f"solution count over {report.samples} poses")
    left.legend(fontsize=8)
    left.grid(axis="y", alpha=0.3)

    bars = right.bar(
        ["published four\n(q4_root = +1)", "full eight\nbranches"],
        [100.0 * report.published_rate, 100.0 * report.full_rate],
        color=["tab:orange", "tab:blue"],
    )
    for bar, count in zip(bars, (report.recovered_published, report.recovered_full), strict=True):
        right.annotate(f"{count}/{report.samples}\n{bar.get_height():.1f} %",
                       (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                       textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    right.set_ylim(0.0, 115.0)
    right.set_ylabel("configurations recovered [%]")
    right.set_title("does the solver find the configuration it was given?")
    right.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (see ``--help``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=None, help="write the figure here")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed of both studies")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO or DEBUG")
    parser.add_argument("--samples", type=int, default=300, help="number of random poses")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run both studies, print them and draw the figure.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    print("Coverage and solution count of the analytical inverse kinematics")
    print(f"samples = {args.samples}, seed = {args.seed}\n")
    report = coverage_study(samples=args.samples, seed=args.seed)
    print("1. Does the solver return the configuration it was given?")
    print(report.describe())

    counts = solution_count_study(samples=args.samples, seed=args.seed)
    print("\n2. How many distinct in-limit solutions does a pose have?")
    print(counts.describe())

    published = unpublished_counts(samples=args.samples, seed=args.seed)
    print("\n3. The same poses, restricted to the published four-branch subset:")
    print(f"  {'solutions':>10} {'full':>7} {'published':>10}")
    for count in sorted(set(counts.histogram) | set(published)):
        print(f"  {count:>10} {counts.histogram.get(count, 0):>7} {published.get(count, 0):>10}")
    full_total = sum(count * poses for count, poses in counts.histogram.items())
    published_total = sum(count * poses for count, poses in published.items())
    print(f"  the published subset returns {published_total} configurations where the full "
          f"solver returns {full_total} ({full_total - published_total} fewer, "
          f"{100.0 * (full_total - published_total) / full_total:.1f} %)")
    print(f"  poses where the published subset finds no in-limit solution at all: "
          f"{published.get(0, 0)}")

    print_reachable_shell()
    print_failure_classification()

    start = time.perf_counter()
    for target in [np.random.default_rng(args.seed).uniform(lower_limits(), upper_limits())
                   for _ in range(100)]:
        solve(fk_flange(target), float(target[6]), within_limits_only=True)
    LOGGER.info("all eight branches for one pose: %.3f ms (timing, not a study result)",
                1000.0 * (time.perf_counter() - start) / 100)

    plt = import_pyplot(args.save_dir is not None)
    finish(build_figure(plt, counts.histogram, published, report), args.save_dir, "05_coverage_study")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
