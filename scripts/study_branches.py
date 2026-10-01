#!/usr/bin/env python3
"""Run the branch studies of :mod:`franka_ik.analysis` and print the numbers.

This is the script the documentation's measurements come from, so its output is
deliberately plain, labelled and stable: the same command prints the same
numbers, and nothing that varies between machines (timings in particular) is part
of the report.  ``--json`` dumps the same numbers in a form a document or a test
can be generated from.

Two questions are answered, with the same seeded sample of configurations:

* **coverage** -- for a random configuration, does the solver return *that*
  configuration?  Reported for the published four-branch subset (the ``+`` root of
  the elbow quadratic) and for the full eight branches.
* **solution count** -- how many distinct in-limit configurations reach the same
  pose with the same joint 7?  This is a property of the arm, not of the solver.

Examples::

    python3 scripts/study_branches.py
    python3 scripts/study_branches.py --samples 300 --seed 0 --json /tmp/branches.json
    python3 scripts/study_branches.py --quiet --json /tmp/branches.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    coverage_study,
    solution_count_study,
)

LOGGER = logging.getLogger("scripts.study_branches")

#: Decimal places kept in the JSON dump, so that the file is stable.
ROUNDING = 6


def published_counts(samples: int, seed: int) -> Dict[int, int]:
    """Count, per pose, the in-limit solutions the published subset finds.

    Args:
        samples: Number of poses.
        seed: RNG seed; the sampling matches
            :func:`franka_ik.analysis.solution_count_study`, so the histograms
            describe the same poses.

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


def unbiased_geometry() -> EquivalentGeometry:
    """The same arm with ``bias = 0``, i.e. a plain S-R-S arm.

    Returns:
        The geometry with the shoulder offset removed.
    """
    return EquivalentGeometry(
        d_bs=PAPER_GEOMETRY.d_bs,
        d_se=PAPER_GEOMETRY.d_se,
        d_ew=PAPER_GEOMETRY.d_ew,
        d_wt=PAPER_GEOMETRY.d_wt,
        offset=PAPER_GEOMETRY.offset,
        bias=0.0,
    )


def total_configurations(histogram: Dict[int, int]) -> int:
    """Total number of configurations described by a histogram.

    Args:
        histogram: Mapping from solution count to number of poses.

    Returns:
        The sum of ``count * poses``.
    """
    return sum(count * poses for count, poses in histogram.items())


def run_studies(samples: int, seed: int) -> Dict[str, Any]:
    """Run both studies and collect every number the report needs.

    Args:
        samples: Number of random poses.
        seed: RNG seed.

    Returns:
        A JSON-serialisable dictionary of results.
    """
    coverage = coverage_study(samples=samples, seed=seed)
    counts = solution_count_study(samples=samples, seed=seed)
    published = published_counts(samples=samples, seed=seed)
    shell_low, shell_high = reachable_distance_range()
    srs_low, srs_high = reachable_distance_range(unbiased_geometry())

    return {
        "samples": samples,
        "seed": seed,
        "coverage": {
            "recovered_full": coverage.recovered_full,
            "recovered_published": coverage.recovered_published,
            "full_rate": round(coverage.full_rate, ROUNDING),
            "published_rate": round(coverage.published_rate, ROUNDING),
            "mean_solutions": round(coverage.mean_solutions, ROUNDING),
            "min_solutions": coverage.min_solutions,
            "max_solutions": coverage.max_solutions,
        },
        "solution_counts": {str(key): counts.histogram[key] for key in sorted(counts.histogram)},
        "published_solution_counts": {
            str(key): published[key] for key in sorted(published)
        },
        "totals": {
            "full": total_configurations(counts.histogram),
            "published": total_configurations(published),
            "poses_with_no_published_solution": published.get(0, 0),
        },
        "reachable_shell": {
            "with_bias": [round(shell_low, ROUNDING), round(shell_high, ROUNDING)],
            "bias_zero": [round(srs_low, ROUNDING), round(srs_high, ROUNDING)],
        },
    }


def report(result: Dict[str, Any]) -> None:
    """Print the human-readable report.

    Args:
        result: The dictionary returned by :func:`run_studies`.
    """
    coverage = result["coverage"]
    print("franka-ik branch study")
    print(f"  samples = {result['samples']}, seed = {result['seed']}, "
          f"sampling = uniform inside the real joint limits")
    print()
    print("1. Coverage: is the configuration the pose came from among the solutions?")
    print(f"  published four-branch subset (q4_root == +{PUBLISHED_Q4_ROOT}): "
          f"{coverage['recovered_published']}/{result['samples']} "
          f"({100.0 * coverage['published_rate']:.1f} %)")
    print(f"  full eight-branch solver:     "
          f"{coverage['recovered_full']}/{result['samples']} "
          f"({100.0 * coverage['full_rate']:.1f} %)")
    print()
    print("2. Solution count per pose (distinct in-limit configurations):")
    print(f"  mean {coverage['mean_solutions']:.2f}, "
          f"range {coverage['min_solutions']}-{coverage['max_solutions']}")
    print(f"  {'solutions':>10} {'full':>7} {'published':>10}   histogram (full)")
    for key in sorted(int(item) for item in result["solution_counts"]):
        full = result["solution_counts"].get(str(key), 0)
        published = result["published_solution_counts"].get(str(key), 0)
        bar = "#" * int(round(40 * full / result["samples"]))
        print(f"  {key:>10} {full:>7} {published:>10}   {bar}")
    totals = result["totals"]
    print(f"  total configurations: {totals['full']} with eight branches, "
          f"{totals['published']} with the published four "
          f"({totals['full'] - totals['published']} fewer, "
          f"{100.0 * (totals['full'] - totals['published']) / totals['full']:.1f} %)")
    print(f"  poses where the published subset finds nothing at all: "
          f"{totals['poses_with_no_published_solution']}")
    print()
    print("3. Reachable shell of ||x_sw|| (elbow discriminant non-negative):")
    shell = result["reachable_shell"]
    print(f"  with the shoulder bias: {shell['with_bias'][0]:.4f} m to "
          f"{shell['with_bias'][1]:.4f} m")
    print(f"  bias = 0 (plain S-R-S): {shell['bias_zero'][0]:.4f} m to "
          f"{shell['bias_zero'][1]:.4f} m")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list, ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=300, help="number of random poses")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed of both studies")
    parser.add_argument(
        "--json", type=Path, default=None, help="write the numbers to this file as JSON"
    )
    parser.add_argument("--quiet", action="store_true", help="do not print the report")
    parser.add_argument("--log-level", default="WARNING", help="logging level, e.g. WARNING or INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the studies.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    if args.samples < 1:
        raise SystemExit("--samples must be positive")

    result = run_studies(args.samples, args.seed)
    if not args.quiet:
        report(result)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("wrote %s", args.json)
        if not args.quiet:
            print(f"\nnumbers written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
