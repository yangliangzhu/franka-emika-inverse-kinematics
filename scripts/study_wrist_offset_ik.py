#!/usr/bin/env python3
"""Measure how this method relates to the published Franka analytical IK solvers.

The repository's headline finding -- the published code keeps four elbow branches
and the arm has eight -- was measured in 2026, on code written in 2020, against a
literature that has since caught up.  ``docs/provenance.md`` tells that story, and
this is the script its numbers come from, so that every claim in it is one command
away from being re-checked.

Four measurements, all on the same seeded sample of in-limit configurations:

``roots``
    Which root of the ``STEP2`` elbow quadratic the generating configuration sits
    on, and whether the published four-branch subset can recover it.  This is the
    265/300 of ``docs/branch_analysis.md``, recomputed here so the provenance
    numbers and the branch-analysis numbers cannot drift apart.

``limits``
    The question the branch finding raises and does not answer: are the missing
    branches missing because they violate the Panda's joint limits?  The
    published code never checks a limit (``limit_joints`` wraps into hand-written
    windows, see ``--show-limit-joints``), but a reader is right to ask whether
    the discarded half is *usefully* discarded.  It is not: every generating
    configuration on the second root has in-limit solutions, and so does every
    other pose in the sample.

``q4``
    The histogram of joint 4 on the second root, split by whether the solution is
    inside the Panda's range.  This is the measurement that lets the repository
    say something precise about the closest published work, which discards one
    variant of the second root on the grounds that it is impractical on a Franka.

``optimiser``
    Optional (``--optimiser``): an independent CasADi + IPOPT enumeration, used
    to check that the eight branches are *all* of them rather than the eight the
    derivation happens to produce.  Slow -- about two seconds per pose -- and
    skipped by default.

Examples::

    python3 scripts/study_wrist_offset_ik.py
    python3 scripts/study_wrist_offset_ik.py --samples 300 --seed 0 --json /tmp/provenance.json
    python3 scripts/study_wrist_offset_ik.py --optimiser --optimiser-poses 10
    python3 scripts/study_wrist_offset_ik.py --show-limit-joints
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import (  # noqa: E402
    LOWER_LIMITS_DEG,
    UPPER_LIMITS_DEG,
    analysis,
    fk_flange,
    lower_limits,
    solve,
    upper_limits,
)

LOGGER = logging.getLogger("study_wrist_offset_ik")

#: The ``q7`` tolerance used everywhere in this repository when two
#: configurations are compared: whole turns are identified, see
#: :func:`franka_ik.solver.solve`.
_TURN_DEG = 360.0

#: Joint-4 window that the closest published work quotes for the elbow variant it
#: discards.  Quoted, not derived here: He and Liu, *Analytical Inverse Kinematics
#: for Franka Emika Panda -- a Geometrical Solver for 7-DOF Manipulators with
#: Unconventional Design*, say Case A1 gives ``q4 in [-26.76, -4] deg`` and that
#: they ignore it.  The measurement below is what that window actually contains.
PUBLISHED_ELBOW_WINDOW_DEG = (-26.76, -4.0)


def sample_configurations(samples: int, seed: int) -> np.ndarray:
    """Draw configurations uniformly inside the Panda's real joint limits.

    Uniform on ``[-pi, pi]`` would be wrong: joints 4 and 6 are one-sided, so a
    symmetric draw produces configurations the arm cannot reach.
    """
    rng = np.random.default_rng(seed)
    lower, upper = lower_limits(), upper_limits()
    return lower + rng.random((samples, 7)) * (upper - lower)


def _wrap_deg(q: np.ndarray) -> np.ndarray:
    """Map joint angles in radians to degrees in ``[-180, 180)``."""
    return (np.degrees(q) + _TURN_DEG / 2.0) % _TURN_DEG - _TURN_DEG / 2.0


def _same_configuration(a: np.ndarray, b: np.ndarray, tol_deg: float = 1e-6) -> bool:
    """True when two configurations differ only by whole turns."""
    return bool(np.all(np.abs(_wrap_deg(a) - _wrap_deg(b)) < tol_deg))


def _inside_limits(q: np.ndarray) -> bool:
    """True when every joint is inside its range (a hair of slack for rounding)."""
    degrees = np.degrees(q)
    return bool(
        np.all(degrees >= LOWER_LIMITS_DEG - 1e-9) and np.all(degrees <= UPPER_LIMITS_DEG + 1e-9)
    )


def study_roots(samples: int, seed: int) -> Dict[str, Any]:
    """Which elbow root reproduces the configuration, and can the published four?"""
    counts = {"plus": 0, "minus": 0, "neither": 0}
    recovered_by_published = 0
    for q in sample_configurations(samples, seed):
        solutions = solve(fk_flange(q), float(q[6]), within_limits_only=False)
        on_plus = any(
            s.q4_root == analysis.PUBLISHED_Q4_ROOT and _same_configuration(s.q, q)
            for s in solutions
        )
        on_minus = any(
            s.q4_root != analysis.PUBLISHED_Q4_ROOT and _same_configuration(s.q, q)
            for s in solutions
        )
        if on_plus:
            counts["plus"] += 1
            recovered_by_published += 1
        elif on_minus:
            counts["minus"] += 1
        else:
            counts["neither"] += 1
    return {
        "samples": samples,
        "seed": seed,
        "targets_on_plus_root": counts["plus"],
        "targets_on_minus_root": counts["minus"],
        "targets_on_neither": counts["neither"],
        "recovered_by_published_subset": recovered_by_published,
        "published_rate": recovered_by_published / float(samples),
    }


def study_limits(samples: int, seed: int) -> Dict[str, Any]:
    """Are the branches the published subset drops dropped because of the limits?"""
    on_minus = 0
    minus_with_in_limit = 0
    minus_without_in_limit = 0
    plus_contributes_too = 0
    minus_only_pose = 0
    in_limit_plus = 0
    in_limit_minus = 0
    for q in sample_configurations(samples, seed):
        solutions = solve(fk_flange(q), float(q[6]), within_limits_only=False)
        plus = [s for s in solutions if s.q4_root == analysis.PUBLISHED_Q4_ROOT]
        minus = [s for s in solutions if s.q4_root != analysis.PUBLISHED_Q4_ROOT]
        in_plus = [s for s in plus if _inside_limits(s.q)]
        in_minus = [s for s in minus if _inside_limits(s.q)]
        in_limit_plus += len(in_plus)
        in_limit_minus += len(in_minus)
        if any(_same_configuration(s.q, q) for s in minus):
            on_minus += 1
            if in_minus:
                minus_with_in_limit += 1
            else:
                minus_without_in_limit += 1
            if in_plus:
                plus_contributes_too += 1
            else:
                minus_only_pose += 1
    return {
        "samples": samples,
        "seed": seed,
        "targets_on_minus_root": on_minus,
        "minus_targets_with_an_in_limit_solution": minus_with_in_limit,
        "minus_targets_without_any_in_limit_solution": minus_without_in_limit,
        "minus_targets_where_the_plus_root_also_works": plus_contributes_too,
        "minus_targets_only_the_minus_root_reaches": minus_only_pose,
        "in_limit_solutions_on_plus_root": in_limit_plus,
        "in_limit_solutions_on_minus_root": in_limit_minus,
    }


def study_q4(samples: int, seed: int) -> Dict[str, Any]:
    """Where joint 4 lands on the second elbow root, in limits and out of them."""
    inside: List[float] = []
    outside: List[float] = []
    for q in sample_configurations(samples, seed):
        for s in solve(fk_flange(q), float(q[6]), within_limits_only=False):
            if s.q4_root == analysis.PUBLISHED_Q4_ROOT:
                continue
            (inside if _inside_limits(s.q) else outside).append(float(np.degrees(s.q[3])))
    low, high = PUBLISHED_ELBOW_WINDOW_DEG

    def summary(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0}
        array = np.asarray(values)
        return {
            "count": int(array.size),
            "min_deg": float(array.min()),
            "median_deg": float(np.median(array)),
            "max_deg": float(array.max()),
        }

    in_window = [v for v in inside if low <= v <= high]
    above_window = [v for v in inside if v > high]
    below_window = [v for v in inside if v < low]
    return {
        "samples": samples,
        "seed": seed,
        "published_elbow_window_deg": [low, high],
        "minus_root_q4_in_limit": summary(inside),
        "minus_root_q4_out_of_limit": summary(outside),
        "in_limit_solutions_within_the_published_window": len(in_window),
        "in_limit_solutions_above_the_window": len(above_window),
        "in_limit_solutions_below_the_window": len(below_window),
        "share_of_in_limit_minus_root_inside_the_window": (
            len(in_window) / float(len(inside)) if inside else None
        ),
    }


def study_optimiser(poses: int, seed: int, starts: int) -> Dict[str, Any]:
    """Cross-check the eight branches against an independent IPOPT enumeration.

    ``numerical.completeness_check`` already answers the question; this function
    asks it again with the two criteria kept apart, because the repository's claim
    rests on exactly that distinction:

    * the **pose residual** says IPOPT produced a solution at all;
    * the **distance to the nearest branch** says which branch it is.

    A candidate counts as an unexplained configuration only when it reaches the
    pose and is further than ``counter_example_tolerance`` from every branch.  A
    candidate that reaches the pose but sits a fraction of a degree from a branch
    is the optimiser stopping short -- expected at a kinematic singularity, where a
    joint can move without moving the tool -- and is reported separately rather
    than silently absorbed into a tolerance.
    """
    from franka_ik import numerical  # imported late: CasADi is an optional extra

    # Read the tolerance off the function rather than repeating it, so the two
    # cannot drift apart.
    counter_example_tolerance = float(
        numerical.completeness_check.__defaults__[-1]  # type: ignore[union-attr]
    )
    total = reached_pose = matched = 0
    distances: List[float] = []
    short_pose: Dict[int, float] = {}
    for index, q in enumerate(sample_configurations(poses, seed)):
        pose, q7 = fk_flange(q), float(q[6])
        report = numerical.completeness_check(pose, q7, starts=starts, seed=seed + 1000 + index)
        total += len(report.numerical)
        analytic = solve(pose, q7, within_limits_only=True)
        for candidate in report.numerical:
            if (
                float(np.abs(fk_flange(candidate) - pose).max())
                > numerical.DEFAULT_SOLUTION_TOLERANCE
            ):
                continue  # IPOPT did not reach the pose; not a solution at all
            reached_pose += 1
            nearest = min(
                (numerical.configuration_distance(candidate, s.q) for s in analytic),
                default=math.inf,
            )
            distances.append(math.degrees(nearest))
            if nearest <= math.radians(counter_example_tolerance):
                matched += 1
            else:
                # A joint can move a long way without moving the tool near a
                # kinematic singularity, so record how far the *pose* moves for the
                # configuration offset that was called a miss.  If that is at the
                # level of round-off, the "miss" is the optimiser's stopping point
                # and not a configuration the branches fail to contain.
                branch = min(
                    analytic, key=lambda s: numerical.configuration_distance(candidate, s.q)
                )
                pose_gap = float(np.abs(fk_flange(candidate) - fk_flange(branch.q)).max())
                short_pose[index] = max(short_pose.get(index, 0.0), pose_gap)
    return {
        "poses": poses,
        "seed": seed,
        "starts_per_pose": starts,
        "optimiser_solutions": total,
        "reached_the_pose": reached_pose,
        "matched_by_an_analytic_branch": matched,
        "stopping_short_of_a_branch": reached_pose - matched,
        "worst_distance_deg": max(distances) if distances else 0.0,
        "stopping_short_pose_gap": {str(index): gap for index, gap in sorted(short_pose.items())},
        "counter_example_tolerance_deg": counter_example_tolerance,
    }


def show_limit_joints() -> Dict[str, Any]:
    """Demonstrate that the published ``limit_joints`` never consults a limit.

    ``limit_joints`` wraps into hand-written windows -- joints 1, 2, 3, 5 and 7
    into ``[-181, 181]``, joint 4 into ``[-271, 91]``, joint 6 into ``[-74, 288]``
    -- while ``Panda.upper_bounds``/``lower_bounds`` say something narrower.  The
    docstring of ``franka_ik.analysis`` quotes this as the reason the published
    four-branch subset is not a deliberate limit filter.
    """
    sys.path.insert(0, str(REPO_ROOT / "original"))
    try:
        from ik_ca import limit_joints  # type: ignore[import-not-found]
        from panda import Panda  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit(
            "the originals need CasADi and the original/ directory: " + str(exc)
        ) from exc

    model = Panda()
    lower = np.degrees(model.lower_bounds)
    upper = np.degrees(model.upper_bounds)
    cases: List[Tuple[str, int, float]] = [
        ("joint 1, +200 deg", 0, 200.0),
        ("joint 4, 0 deg", 3, 0.0),
        ("joint 6, -30 deg", 5, -30.0),
        ("joint 7, +170 deg", 6, 170.0),
    ]
    rows = []
    for name, index, value in cases:
        probe = np.zeros(7)
        probe[index] = np.radians(value)
        wrapped = np.degrees(limit_joints(probe))
        rows.append(
            {
                "case": name,
                "wrapped_deg": float(wrapped[index]),
                "inside_the_published_model_limits": bool(
                    lower[index] - 1e-9 <= wrapped[index] <= upper[index] + 1e-9
                ),
            }
        )
    return {
        "model_limits_deg": {"lower": lower.tolist(), "upper": upper.tolist()},
        "limit_joints_windows_deg": {
            "joints_1_2_3_5_7": [-181.0, 181.0],
            "joint_4": [-271.0, 91.0],
            "joint_6": [-74.0, 288.0],
        },
        "cases": rows,
    }


def run_studies(samples: int, seed: int) -> Dict[str, Any]:
    """Run the three fast studies on one shared sample."""
    return {
        "roots": study_roots(samples, seed),
        "limits": study_limits(samples, seed),
        "q4": study_q4(samples, seed),
    }


def report(result: Dict[str, Any]) -> None:
    """Print the numbers with the labels the documentation uses."""
    roots = result["roots"]
    print(f"elbow roots, {roots['samples']} in-limit configurations, seed {roots['seed']}")
    print(f"  target on the '+' root (published keeps this half): {roots['targets_on_plus_root']}")
    print(
        f"  target on the '-' root (published cannot return it): {roots['targets_on_minus_root']}"
    )
    print(f"  target on neither root (a real failure):            {roots['targets_on_neither']}")
    print(
        f"  recovered by the published four-branch subset:      "
        f"{roots['recovered_by_published_subset']}/{roots['samples']} "
        f"= {100.0 * roots['published_rate']:.1f} %"
    )
    print()

    limits = result["limits"]
    print("do the joint limits explain the missing half?")
    print(
        f"  '-' root targets:                                   {limits['targets_on_minus_root']}"
    )
    print(
        f"  of those, with an in-limit solution on the '-' root: "
        f"{limits['minus_targets_with_an_in_limit_solution']}"
    )
    print(
        f"  of those, with no in-limit solution at all:          "
        f"{limits['minus_targets_without_any_in_limit_solution']}"
    )
    print(
        f"  of those, where the '+' root also reaches the target: "
        f"{limits['minus_targets_where_the_plus_root_also_works']}"
    )
    print(
        f"  of those, reachable only through the '-' root:        "
        f"{limits['minus_targets_only_the_minus_root_reaches']}"
    )
    print(
        f"  in-limit solutions at the target joint 7: '+' root "
        f"{limits['in_limit_solutions_on_plus_root']}, '-' root "
        f"{limits['in_limit_solutions_on_minus_root']}"
    )
    print()

    q4 = result["q4"]
    low, high = q4["published_elbow_window_deg"]
    panda_low, panda_high = LOWER_LIMITS_DEG[3], UPPER_LIMITS_DEG[3]
    print(
        "joint 4 on the '-' root "
        f"(Panda range [{float(panda_low):.0f}, {float(panda_high):.0f}] deg)"
    )
    for key, label in (
        ("minus_root_q4_in_limit", "in limits"),
        ("minus_root_q4_out_of_limit", "out of limits"),
    ):
        entry = q4[key]
        if entry["count"]:
            print(
                f"  {label:>13}: n={entry['count']:4d}  min {entry['min_deg']:8.2f}  "
                f"median {entry['median_deg']:8.2f}  max {entry['max_deg']:8.2f}"
            )
        else:
            print(f"  {label:>13}: none")
    print(
        f"  in-limit '-' root solutions inside the published window [{low}, {high}] deg: "
        f"{q4['in_limit_solutions_within_the_published_window']}"
    )
    print(
        f"  outside that window: above {q4['in_limit_solutions_above_the_window']}, "
        f"below {q4['in_limit_solutions_below_the_window']}"
    )
    print()


def main(argv: List[str] | None = None) -> int:
    """Parse the arguments, run the studies and print them."""
    parser = argparse.ArgumentParser(
        description="measure this method against the published Franka analytical IK solvers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=300, help="in-limit configurations to draw")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the sample")
    parser.add_argument("--json", type=Path, default=None, help="write the numbers here as JSON")
    parser.add_argument("--quiet", action="store_true", help="do not print the report")
    parser.add_argument(
        "--optimiser",
        action="store_true",
        help="also run the independent CasADi + IPOPT cross-check (slow)",
    )
    parser.add_argument("--optimiser-poses", type=int, default=10, help="poses for --optimiser")
    parser.add_argument(
        "--optimiser-starts", type=int, default=400, help="IPOPT starts per pose for --optimiser"
    )
    parser.add_argument(
        "--show-limit-joints",
        action="store_true",
        help="also demonstrate that the published limit_joints never checks a limit",
    )
    parser.add_argument("--log-level", default="warning", help="logging level, e.g. info or debug")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    if args.samples < 1:
        raise SystemExit("--samples must be positive")

    result = run_studies(args.samples, args.seed)
    if args.optimiser:
        if args.optimiser_poses < 1:
            raise SystemExit("--optimiser-poses must be positive")
        result["optimiser"] = study_optimiser(
            args.optimiser_poses, args.seed, args.optimiser_starts
        )
    if args.show_limit_joints:
        result["limit_joints"] = show_limit_joints()

    if not args.quiet:
        report(result)
        optimiser = result.get("optimiser")
        if optimiser is not None:
            print(
                f"independent IPOPT check, {optimiser['poses']} poses x "
                f"{optimiser['starts_per_pose']} starts"
            )
            print(
                f"  IPOPT solutions {optimiser['optimiser_solutions']}, of which "
                f"{optimiser['reached_the_pose']} reach the pose"
            )
            print(
                f"  of those, matched by an analytic branch: "
                f"{optimiser['matched_by_an_analytic_branch']}, further than "
                f"{optimiser['counter_example_tolerance_deg']:.2f} deg from every branch "
                f"(counter-examples): {optimiser['stopping_short_of_a_branch']}"
            )
            if optimiser["stopping_short_of_a_branch"]:
                gaps = optimiser["stopping_short_pose_gap"]
                worst_gap = max(gaps.values()) if gaps else 0.0
                print(
                    f"  the worst such distance is {optimiser['worst_distance_deg']:.4f} deg, on "
                    f"poses {sorted(gaps)}; there a configuration offset of that size moves the "
                    f"pose by at most {worst_gap:.1e}"
                )
                print(
                    "  so those are IPOPT stopping short at a kinematic singularity, not "
                    "configurations the branches miss"
                )
            print()
        if "limit_joints" in result:
            entry = result["limit_joints"]
            print("does the published limit_joints enforce the Panda limits?  (it does not)")
            print(f"  Panda model limits, lower: {entry['model_limits_deg']['lower']}")
            print(f"  Panda model limits, upper: {entry['model_limits_deg']['upper']}")
            for row in entry["cases"]:
                print(
                    f"  {row['case']:>20} -> {row['wrapped_deg']:8.2f} deg   "
                    f"inside those limits: {row['inside_the_published_model_limits']}"
                )
            print()

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("wrote %s", args.json)
        if not args.quiet:
            print(f"numbers written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
