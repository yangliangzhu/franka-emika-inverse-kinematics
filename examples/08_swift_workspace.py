#!/usr/bin/env python3
"""Example 08 -- the workspace shell, and why 88 % coverage fails at the edges.

``franka_ik.analysis.reachable_distance_range()`` gives the shell the wrist can
reach in closed form: ``‖x_sw‖`` lies in ``[0.06617, 0.71935]`` m.  This example
draws both surfaces -- the outer boundary and the hole around the shoulder -- as
two point clouds of markers, with the arm inside them, so "out of reach" becomes a
distance you can see rather than an error string.

The slider is the **elbow**, and that is a measurement rather than a choice:
joint 4 is the only single joint that changes ``‖x_sw‖`` at all.  Joints 1, 2, 3, 5,
6 and 7 rotate the wrist about the shoulder or about itself, and a full sweep of any
of them leaves the distance unchanged at 0.54405 m -- printed on start-up, so the
claim is checkable.  Sweeping joint 4 across its one-sided ``[-175, -5]`` degrees
takes the wrist from 0.20685 m to 0.71935 m, and **0.71935 m is the outer radius
itself**: at joint 4 = -26.7573 deg the arm is exactly straight, and ``‖x_sw‖`` there
is 0.719354203404 m, the closed-form outer radius to the last digit (the start-up
line prints the difference: ``0.0e+00``).

That angle is a property of the **arm**, not of the pose -- measured ``-26.7573``
degrees for every pose tried, because full extension is where the elbow stops
contributing -- and it is where the in-limit band of the second elbow root begins
(`docs/provenance.md` section 3).

Two things are worth looking at, and both are measured rather than asserted:

* **the shell is a property of the arm, not of a pose.** No sampling is involved:
  the radius is the closed form from the elbow quadratic's discriminant, and every
  marker sits exactly on it.
* **where the published four-branch subset fails is measurable.** ``--scan``
  samples configurations band by band across the shell and reports how often that
  subset recovers the generating configuration.  Measured (30 per band, seed 0):
  every band from 0.20 m out recovers 30 of 30 except the outermost -- the arm at
  full stretch -- which recovers 23 of 30 (76.7 %), and the mean number of
  in-limit solutions rises with reach, 2.2 to 4.1.  Stretched out, the second elbow
  root is the in-limit one, which is why the 88.3 % headline is an average and not
  a uniform rate.

  The two innermost bands come back empty at ``seed 0``: the shell does extend to
  0.06617 m, but uniform sampling of joint space essentially never lands there --
  that region is the arm folded back onto its own shoulder.  A band with no samples
  prints ``nan``, which is the honest answer; it is not "100 % coverage".

Run it with::

    python examples/08_swift_workspace.py
    python examples/08_swift_workspace.py --markers 600 --inner
    python examples/08_swift_workspace.py --headless --scan
    python examples/08_swift_workspace.py --pose near_singular --model skeleton

The viewer needs the optional ``viz`` extra (``uv sync --extra viz``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import franka_ik as fk  # noqa: E402
from franka_ik import analysis, geometry, model, swift_app  # noqa: E402
from franka_ik.swift_viz import (  # noqa: E402
    add_cloud,
    add_shapes,
    apply_camera,
    launch_env,
    reachable_shell_markers,
)

#: Radius bands used by ``--scan``, as fractions of the outer reach.
_SCAN_BANDS = 10

#: The joint the workspace slider drives: the elbow.  See the module docstring --
#: it is the only one that moves ``‖x_sw‖``.
_ELBOW = 3


def shoulder_wrist_distance(q: Sequence[float]) -> float:
    """``‖x_sw‖``: the distance from the shoulder to the equivalent wrist.

    This is the quantity the reachable shell bounds.  It is computed the way the
    solver computes it -- flange position, the step-1 wrist correction, then the
    vector from the shoulder -- so a value printed here is the value the elbow
    quadratic's discriminant actually sees.

    Args:
        q: Seven joint angles in radians.

    Returns:
        The distance in metres.
    """
    pose = model.fk_flange(q)
    corrected = pose[:3, :3] @ geometry.wrist_correction(float(q[6]))
    return float(np.linalg.norm(geometry.shoulder_to_wrist(pose[:3, 3], corrected)))


def with_elbow(q0: Sequence[float], degrees: float) -> np.ndarray:
    """``q0`` with joint 4 set to ``degrees``, everything else untouched."""
    moved = np.asarray(q0, dtype=float).copy()
    moved[_ELBOW] = np.radians(degrees)
    return moved


def elbow_sweep(
    q0: Sequence[float], *, samples: int = 401, iterations: int = 80
) -> Tuple[float, float, float]:
    """How ``‖x_sw‖`` depends on the elbow.

    The coarse scan is what makes this a measurement rather than a claim about the
    closed form, and the ternary search is what makes the *angle* a measurement too.
    The peak is flat: at 401 samples the best sample sits 0.4 degrees from the true
    full-extension angle while its ``‖x_sw‖`` is already right to 1e-9 m, so reporting
    the sampled angle would understate the resolution by four orders of magnitude.
    Measured, at the refined angle ``‖x_sw‖`` is 0.719354203404 m, which is the
    closed-form outer radius to the last digit.

    The angle is a property of the **arm**, not of the pose: measured ``−26.7573``
    degrees for every pose tried, because full extension is where the elbow stops
    contributing.  It is where the in-limit band of the second elbow root begins
    (``docs/provenance.md`` section 3).

    Args:
        q0: The configuration to hold every other joint at, radians.
        samples: Joint-4 values in the coarse scan across the joint's own range.
        iterations: Ternary-search steps refining the bracketed peak.

    Returns:
        ``(smallest, largest, degrees at largest)``: the two ends of the sweep, in
        metres, and the joint-4 angle at which the arm is at full stretch.
    """
    low, high = fk.LOWER_LIMITS_DEG[_ELBOW], fk.UPPER_LIMITS_DEG[_ELBOW]
    grid = np.linspace(low, high, samples)
    distances = [shoulder_wrist_distance(with_elbow(q0, float(degrees))) for degrees in grid]
    peak = int(np.argmax(distances))
    left = float(grid[max(peak - 1, 0)])
    right = float(grid[min(peak + 1, len(grid) - 1)])
    for _ in range(iterations):
        third = (right - left) / 3.0
        middle_left, middle_right = left + third, right - third
        if shoulder_wrist_distance(with_elbow(q0, middle_left)) < shoulder_wrist_distance(
            with_elbow(q0, middle_right)
        ):
            left = middle_left
        else:
            right = middle_right
    stretch = (left + right) / 2.0
    return min(distances), shoulder_wrist_distance(with_elbow(q0, stretch)), float(stretch)


def scan_coverage(samples: int = 30, seed: int = 0) -> None:
    """Print the published-subset coverage as a function of reach.

    For each band of ``‖x_sw‖`` between the inner and outer surface, sample
    configurations whose wrist distance falls in the band and report how often the
    published four-branch subset recovers the generating configuration.  The point
    is to show that the 88 % headline is an average: the failure concentrates where
    the arm is stretched or folded.

    Args:
        samples: Configurations to sample per band.
        seed: RNG seed.
    """
    low, high = analysis.reachable_distance_range()
    rng = np.random.default_rng(seed)
    lower, upper = model.lower_limits(), model.upper_limits()
    print(f"reachable shell: ‖x_sw‖ in [{low:.5f}, {high:.5f}] m")
    print()
    print(
        f"{'band (m)':>18}  {'samples':>7}  {'published recovers':>19}  {'mean in-limit solutions':>24}"
    )
    edges = np.linspace(low, high, _SCAN_BANDS + 1)
    for index in range(_SCAN_BANDS):
        found = 0
        recovered = 0
        total = 0
        attempts = 0
        while found < samples and attempts < 2000 * samples:
            attempts += 1
            q = rng.uniform(lower, upper)
            if not edges[index] <= shoulder_wrist_distance(q) < edges[index + 1]:
                continue
            found += 1
            pose = model.fk_flange(q)
            solutions = fk.solve(pose, float(q[6]), within_limits_only=True)
            total += len(solutions)
            delta = np.degrees(np.asarray([s.q for s in solutions], dtype=float) - q)
            delta = (delta + 180.0) % 360.0 - 180.0
            target_on_first = False
            if len(delta):
                hit = np.all(np.abs(delta) < 1e-6, axis=1)
                target_on_first = bool(
                    np.any(
                        hit
                        & np.asarray(
                            [s.q4_root == analysis.PUBLISHED_Q4_ROOT for s in solutions], dtype=bool
                        )
                    )
                )
            recovered += int(target_on_first)
        share = 100.0 * recovered / found if found else float("nan")
        mean = total / found if found else float("nan")
        print(
            f"[{edges[index]:.4f}, {edges[index + 1]:.4f})  {found:7d}  "
            f"{recovered:8d} ({share:5.1f} %)  {mean:24.2f}"
        )


def main(argv: List[str] | None = None) -> int:
    """Run the example."""
    parser = swift_app.common_parser(
        __doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--markers", type=int, default=400, help="markers per shell surface")
    parser.add_argument(
        "--inner", action="store_true", help="also draw the inner surface at start-up"
    )
    parser.add_argument(
        "--scan", action="store_true", help="also print coverage against reach, then exit"
    )
    args = parser.parse_args(argv)
    swift_app.configure_logging(args)

    if args.scan:
        scan_coverage()
        return 0

    q, q7 = swift_app.resolve_pose(args)
    low, high = analysis.reachable_distance_range()
    distance = shoulder_wrist_distance(q)
    smallest, largest, stretch = elbow_sweep(q)
    print(f"shell           : ‖x_sw‖ in [{low:.5f}, {high:.5f}] m")
    print(
        f"this pose       : ‖x_sw‖ = {distance:.5f} m  ({100.0 * distance / high:.1f} % of reach)"
    )
    print(
        f"elbow sweep     : ‖x_sw‖ {smallest:.5f} to {largest:.5f} m over joint "
        f"{_ELBOW + 1}, full stretch at {stretch:.4f} deg"
    )
    print(
        f"stretch check   : ‖x_sw‖ there is {largest:.12f} m, the closed-form outer "
        f"radius {high:.12f} m (difference {high - largest:.1e})"
    )
    print(f"drawing         : {args.markers} markers per surface, {args.model} arm")
    print(f"failure reason  : {fk.classify_failure(model.fk_flange(q), q7)}")

    from franka_ik.swift_viz import check_kinematics

    arm, urdf = swift_app.load_arm_for(args)
    worst = check_kinematics(arm) if (urdf is not None and arm is not None) else None
    env = launch_env(headless=args.headless, browser=args.browser)
    swift_app.print_model_summary(arm, urdf, worst=worst, model=args.model)

    state = {"q": np.asarray(q, dtype=float).copy(), "inner": bool(args.inner)}
    arm_shapes = swift_app.build_arm_shapes(args, arm=arm)
    arm_shapes.update(state["q"])
    outer = reachable_shell_markers(count=args.markers, radius=0.005, inner=False)
    inner = reachable_shell_markers(count=args.markers, radius=0.005, inner=True)

    # The arm moves, so it goes in one shape at a time; the two shells never move, so
    # each goes in as a single assembly -- one message instead of 400, which is the
    # difference between a viewer that opens and one that takes minutes to
    # (docs/browser_debugging.md section 2.5).
    add_shapes(env, arm_shapes.shapes)
    add_cloud(env, outer)
    inner_cloud = None
    if state["inner"]:
        inner_cloud = add_cloud(env, inner)
    apply_camera(env, args.camera)

    def readout_lines(values: np.ndarray) -> List[str]:
        """The measurement, for the readout: reach and what the solver says.

        ASCII only: these strings cross a websocket into an HTML label, and a
        degrees sign that arrives mangled is a readout nobody can check.
        """
        current = shoulder_wrist_distance(values)
        solutions = fk.solve(model.fk_flange(values), float(values[6]), within_limits_only=True)
        return [
            f"elbow (joint 4) = {np.degrees(values[_ELBOW]):.2f} deg",
            f"|x_sw| = {current:.5f} m &nbsp; {100.0 * current / high:.1f} % of reach",
            f"inside the outer surface by {high - current:.5f} m &nbsp; (it is {high:.5f} m, "
            f"the inner one {low:.5f} m)",
            f"in-limit solutions at this joint 7 : {len(solutions)} &nbsp; solver says "
            f"{fk.classify_failure(model.fk_flange(values), float(values[6]))}",
        ]

    readout: object = None
    slider: object = None

    def on_elbow(value: float) -> None:
        """Move the elbow, re-measure, and re-label."""
        state["q"][_ELBOW] = np.radians(value)
        arm_shapes.update(state["q"])
        if readout is not None:
            swift_app.set_readout(readout, readout_lines(state["q"]))

    def on_inner(value: object) -> None:
        """Add or remove the inner surface, which is drawn or it is not."""
        nonlocal inner_cloud
        flags = swift_app.checkbox_flags(value)
        wanted = bool(flags[0]) if flags else state["inner"]
        if wanted == state["inner"]:
            return
        state["inner"] = wanted
        if wanted:
            inner_cloud = add_cloud(env, inner)
        elif inner_cloud is not None:
            env.remove(inner_cloud)
            inner_cloud = None

    if args.headless:
        print()
        print(
            f"headless: placed {len(outer)} markers on the outer surface"
            + (f" and {len(inner)} on the inner one" if state["inner"] else "")
            + f", and a {arm_shapes.kind} arm, nothing drawn."
        )
    else:
        elements = swift_app.require_viz().swift.Elements
        readout = swift_app.add_readout(env, readout_lines(state["q"]))
        slider = swift_app.add_slider(
            env,
            low=float(fk.LOWER_LIMITS_DEG[_ELBOW]),
            high=float(fk.UPPER_LIMITS_DEG[_ELBOW]),
            value=float(np.degrees(state["q"][_ELBOW])),
            label=f"joint {_ELBOW + 1} (elbow)",
            step=0.5,
            unit="deg",
            precision=2,
            name="elbow",
            elements=elements,
        )
        env.add_ui(
            elements.Checkbox(on_inner, label="surface", options=["inner"], checked=[False]),
            name="inner",
        )
        swift_app.add_camera_radio(env, initial=args.camera, elements=elements)
        print()
        print(
            f"interactive: elbow slider {fk.LOWER_LIMITS_DEG[_ELBOW]:.0f} to "
            f"{fk.UPPER_LIMITS_DEG[_ELBOW]:.0f} deg, an 'inner' checkbox for the second "
            "surface, a camera radio,"
        )
        print("             and a readout of the reach the picture shows.")

    swift_app.interaction_loop(
        env,
        slider=slider,
        initial=float(np.degrees(state["q"][_ELBOW])),
        on_change=on_elbow,
        steps=args.steps,
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
