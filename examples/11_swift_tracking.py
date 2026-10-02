#!/usr/bin/env python3
"""Example 11 -- tracking a straight line, and the joint-2 crossing that fails.

``franka_ik.solver.solve_closest`` is the entry point a controller wants: give it
the previous configuration and it returns the solution nearest to it, so a
trajectory does not jump between branches halfway.  This example follows the tool
along a straight Cartesian line and animates the arm, with the manipulability
ellipsoid as the readout -- when it flattens, the arm is about to need a large joint
velocity for a small tool motion.

The plan is computed once, before anything is drawn, and the browser then scrubs it:
a **progress** slider picks a step, **play/pause** runs it at the rate the **speed**
slider asks for (in trajectory steps per second, not frames), and the readout follows
the selected step.  Scrubbing is the point of precomputing: the whole trajectory --
including where it fails -- is available to look at backwards, which an animation
that only runs forward is not.

The default line is chosen to sweep joint 2 **through zero**, which is the
coordinate singularity of ``docs/limitations.md`` §12: there the closed form cannot
reconstruct the shoulder, and at the right configuration nothing the solver returns
is both correct and inside the joint limits.

Why "at the right configuration" is doing work in that sentence: measured, the
crossing is usually uneventful, and finer steps get closer without breaking.  Tracking
this line with 80 plan steps passes ``0.21°`` away from ``q₂ = 0``, with 1000 steps it
passes ``6.3e-3°`` away, and in both cases every step solves with a pose residual of
about ``1e-13``.  The singularity bites when the configuration is *exactly* on it:
sampling 200 configurations near ``q₂ = 0``, 70 % of those with ``q₂ = 0`` exactly
have no in-limit solution, and the rate falls to 7 % within ``1e-4`` degrees and 0 %
by ``1e-3``.

So this example is the counterpart to ``examples/10_swift_singularities.py``, not a
repeat of it: it shows the crossing a controller does hundreds of times an hour, and
prints the distance to zero so the claim is checkable.  A step that does fail still
stops the animation and names the reason -- and ``--plan-steps`` is how finely the
line is sampled, while ``--steps`` is how many frames to run.

Three numbers are tracked per step and printed when the run ends:

``pose residual``
    ``‖fk_flange(q) − pose‖∞`` (the solver's frame, not the tool's); 1e-15 or so
    when the step succeeded.
``step size``
    largest per-joint move from the previous step, in degrees.  A continuous
    branch keeps this small; a branch switch shows up as a jump.
``manipulability``
    the ellipsoid's overall size, and the quantity that tells you a step is about
    to become expensive.

Run it with::

    python examples/11_swift_tracking.py
    python examples/11_swift_tracking.py --plan-steps 120 --travel 0.30
    python examples/11_swift_tracking.py --no-cross          # avoid the singularity
    python examples/11_swift_tracking.py --headless --steps 200

The viewer needs the optional ``viz`` extra (``uv sync --extra viz``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import franka_ik as fk  # noqa: E402
from franka_ik import model, swift_app  # noqa: E402
from franka_ik.swift_viz import (  # noqa: E402
    add_shapes,
    apply_camera,
    flange_axes,
    launch_env,
    manipulability_ellipsoid,
    tool_axes,
    tool_stem,
    update_manipulability_ellipsoid,
    update_tool_stem,
)

#: Cartesian offset of the line's end from its start, in metres.  Straight down:
#: for the ``ready`` pose this sweeps joint 2 from −17.2° to +0.38° and so crosses
#: ``q₂ = 0``.  Measured at the default 80 plan steps: closest approach 2.1e-1°, all
#: 80 steps solved, worst pose residual 2.2e-14.  At 1000 steps the approach is
#: 6.3e-3° and it still solves.
_DEFAULT_TRAVEL = (0.0, 0.0, -0.35)

#: Trajectory steps per second the speed slider starts at, and its range.  Twelve
#: steps a second walks the default 80-step line in under seven seconds.
_DEFAULT_SPEED = 12.0
_SPEED_RANGE = (2.0, 60.0)

#: The frame interval.  ``dt`` is both the render cadence and the clock the speed
#: slider is measured against, so it has to be the same number the loop is given.
_FRAME_DT = 0.05


def _same(a: np.ndarray, b: np.ndarray, tol_deg: float = 1e-6) -> bool:
    """Whether two configurations differ only by whole turns."""
    delta = np.degrees(np.asarray(a)) - np.degrees(np.asarray(b))
    return bool(np.all(np.abs((delta + 180.0) % 360.0 - 180.0) < tol_deg))


def plan(
    q0: np.ndarray,
    travel: Tuple[float, float, float],
    *,
    steps: int = 80,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[float]]:
    """Follow a straight tool path, keeping the solution nearest the last one.

    Each step is solved independently: the target pose comes from the line, and
    ``solve_closest`` picks the branch closest to the previous configuration.  A
    step that returns nothing is recorded as ``None`` and the plan *continues* --
    the caller decides whether to stop, and the failure count is itself the result.

    Args:
        q0: Starting configuration, radians, whose tool position anchors the line.
        travel: Cartesian offset of the line's end from its start, metres.
        steps: Number of steps along the line.

    Returns:
        ``(targets, solutions, manipulabilities)``: the requested tool poses, the
        configuration found at each step (``None`` where there was none), and the
        manipulability of each solution.
    """
    # The solver's pose is the *flange* pose, so the line is built there.  Because
    # the tool orientation is held fixed, moving the flange position moves the tool
    # point by the same vector -- the 0.1034 m tool offset is a constant rotation of
    # a constant length -- so this is the tool following the line, expressed in the
    # frame the solver works in.  Building the pose with ``fk_tool`` instead would
    # ask for a different point 0.7 m away, which is the mistake this comment exists
    # to prevent.
    anchor = model.fk_flange(q0)
    targets = []
    solutions: List[Optional[np.ndarray]] = []
    manipulabilities: List[float] = []
    reference = np.asarray(q0, dtype=float)
    for alpha in np.linspace(0.0, 1.0, steps):
        pose = anchor.copy()
        pose[:3, 3] = anchor[:3, 3] + alpha * np.asarray(travel, dtype=float)
        targets.append(pose)
        found = fk.solve_closest(pose, float(reference[6]), reference=reference)
        if found is None:
            solutions.append(None)
            manipulabilities.append(float("nan"))
            continue
        reference = np.asarray(found.q, dtype=float)
        solutions.append(reference)
        manipulabilities.append(model.manipulability(reference))
    return targets, solutions, manipulabilities


def report(
    targets: List[np.ndarray],
    solutions: List[Optional[np.ndarray]],
    manipulabilities: List[float],
) -> None:
    """Print the trajectory's measured behaviour."""
    residuals = []
    steps = []
    last: Optional[np.ndarray] = None
    for pose, solution in zip(targets, solutions, strict=False):
        if solution is None:
            continue
        residuals.append(float(np.abs(model.fk_flange(solution) - pose).max()))
        if last is not None:
            delta = np.degrees(solution) - np.degrees(last)
            steps.append(float(np.abs((delta + 180.0) % 360.0 - 180.0).max()))
        last = solution
    missing = sum(1 for solution in solutions if solution is None)
    finite = [m for m in manipulabilities if np.isfinite(m)]
    solved = np.asarray([s for s in solutions if s is not None], dtype=float)
    print()
    if len(solved):
        q2 = np.degrees(solved[:, 1])
        crosses = bool(np.any(q2[:-1] * q2[1:] < 0))
        closest = float(np.min(np.abs(q2)))
        print(
            f"joint 2          : from {q2[0]:+.3f}° to {q2[-1]:+.3f}°, "
            f"closest approach to zero {closest:.2e}°"
        )
        print(
            f"crosses q2 = 0   : {crosses}"
            + (
                "  (and the tracking survived it: see docs/limitations.md §12 for when it does not)"
                if crosses
                else ""
            )
        )
    print(f"steps            : {len(targets)}")
    print(f"solved           : {len(residuals)}   failed: {missing}")
    if residuals:
        print(f"pose residual    : median {np.median(residuals):.2e}  worst {max(residuals):.2e}")
    if steps:
        print(f"joint step       : median {np.median(steps):.3f}°  worst {max(steps):.3f}°")
    if finite:
        print(
            f"manipulability   : min {min(finite):.3e}  max {max(finite):.3e}  "
            f"at the end {finite[-1]:.3e}"
        )
    if missing:
        first = next(index for index, s in enumerate(solutions) if s is None)
        pose = targets[first]
        print()
        print(f"the plan has {missing} unsolved step(s); the first is step {first}.")
        print(
            f"reason: {fk.classify_failure(pose, float(solved[-1][6]) if len(solved) else 0.0)!r}"
        )
        print("If it is 'no_valid_branch' and the step sits on q2 = 0, that is the")
        print("coordinate singularity of docs/limitations.md §12: the branches that are")
        print("exact are outside the joint limits and the ones inside are wrong.")


def main(argv: List[str] | None = None) -> int:
    """Run the example."""
    parser = swift_app.common_parser(
        __doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--travel",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=list(_DEFAULT_TRAVEL),
        help="Cartesian offset of the line's end from its start, in metres",
    )
    parser.add_argument(
        "--plan-steps",
        type=int,
        default=80,
        metavar="N",
        help="steps along the line; --steps is how many frames to run, this is how "
        "finely the line is sampled",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=_DEFAULT_SPEED,
        metavar="HZ",
        help="trajectory steps per second the speed slider starts at",
    )
    parser.add_argument(
        "--no-cross",
        action="store_true",
        help="nudge joint 2 off zero in the start pose, avoiding the singularity",
    )
    args = parser.parse_args(argv)
    swift_app.configure_logging(args)

    q0, q7 = swift_app.resolve_pose(args)
    if args.no_cross and abs(np.degrees(q0[1])) < 1.0:
        q0[1] += np.radians(2.0)
    travel = tuple(float(v) for v in args.travel)

    targets, solutions, manipulabilities = plan(q0, travel, steps=int(args.plan_steps))
    print(
        f"line             : from {np.round(model.fk_tool(q0)[:3, 3], 4).tolist()} "
        f"to {np.round(model.fk_tool(q0)[:3, 3] + np.asarray(travel), 4).tolist()} m"
    )
    print(f"start pose       : {args.pose}   joint 7 = {np.degrees(q7):.3f} deg")
    report(targets, solutions, manipulabilities)

    from franka_ik.swift_viz import check_kinematics

    arm, urdf = swift_app.load_arm_for(args)
    worst = check_kinematics(arm) if (urdf is not None and arm is not None) else None
    env = launch_env(headless=args.headless, browser=args.browser)
    swift_app.print_model_summary(arm, urdf, worst=worst, model=args.model)

    arm_shapes = swift_app.build_arm_shapes(args, arm=arm)
    arm_shapes.update(q0)
    ellipsoid = manipulability_ellipsoid(q0, scale=0.12)
    # The frame follows the **flange**, which is the pose ``plan`` solves for and the
    # frame the readout's residual is measured in; the tool point 0.1034 m further along
    # is drawn with its distance, because the meshes stop at the flange
    # (docs/browser_debugging.md section 3).
    frame = flange_axes(q0, length=0.12)
    tool_frame = tool_axes(q0, length=0.055)
    stem = tool_stem(q0)
    add_shapes(env, arm_shapes.shapes)
    add_shapes(env, [ellipsoid, frame, tool_frame, stem])
    apply_camera(env, args.camera)

    # The plan is drawn up to its first failure: a step that returned no solution has
    # no configuration, and continuing past it would draw a discontinuity as if it
    # were a trajectory.
    failed = next((index for index, s in enumerate(solutions) if s is None), None)
    last = len(solutions) - 1 if failed is None else failed - 1
    if last < 0:
        print()
        print("the first step of the plan has no solution; nothing to draw.")
        env.close()
        return 0
    residuals = [
        float(np.abs(model.fk_flange(solutions[index]) - targets[index]).max())
        for index in range(last + 1)
    ]

    state: Dict[str, object] = {
        "index": 0,
        # Headless there is no button to press, so the plan runs itself; with a
        # browser the reader starts it.
        "playing": bool(args.headless),
        "clock": 0.0,
        "speed": float(args.speed),
    }
    readout: Optional[object] = None
    progress_slider: Optional[object] = None
    speed_slider: Optional[object] = None

    def readout_lines(index: int) -> List[str]:
        """The measurement at step ``index``: where the arm is and how expensive.

        ASCII only: the text crosses a websocket into an HTML label.
        """
        values = solutions[index]
        previous = solutions[index - 1] if index else None
        joint_step = (
            float(np.abs(np.degrees(values) - np.degrees(previous)).max()) if previous is not None else 0.0
        )
        lines = [
            f"step {index + 1} of {len(solutions)} &nbsp; playing: {state['playing']} "
            f"&nbsp; speed {float(state['speed']):.1f} steps/s",
            f"joint 2 = {np.degrees(values[1]):+.4f} deg &nbsp; distance to zero "
            f"{abs(np.degrees(values[1])):.2e} deg",
            f"pose residual = {residuals[index]:.1e} &nbsp; manipulability = "
            f"{model.manipulability(values):.3e}",
            f"joint step from the previous frame = {joint_step:.4f} deg",
            "frames: the flange (the pose the plan solves for) and the tool point "
            "0.1034 m along its z",
        ]
        if failed is not None and index >= last:
            lines.append(
                f"the plan stops here: step {failed + 1} returned no solution "
                f"({fk.classify_failure(targets[failed], float(values[6]))})"
            )
        else:
            lines.append(f"q (deg) = [{swift_app.degrees(values, 1).strip('[]')}]")
        return lines

    def show(index: int) -> None:
        """Draw the plan's step ``index``, and say what it measured."""
        index = max(0, min(int(round(index)), last))
        state["index"] = index
        values = solutions[index]
        arm_shapes.update(values)
        update_manipulability_ellipsoid(ellipsoid, values)
        frame.T = model.fk_flange(values)
        tool_frame.T = model.fk_tool(values)
        update_tool_stem(stem, values)
        if readout is not None:
            swift_app.set_readout(readout, readout_lines(index))

    def on_frame(_current: float) -> Optional[float]:
        """Advance on the speed slider's clock, and never past the failure."""
        if not state["playing"]:
            return None
        if speed_slider is not None:
            state["speed"] = swift_app.read_slider(speed_slider, float(state["speed"]))
        state["clock"] = float(state["clock"]) + _FRAME_DT * float(state["speed"])
        if float(state["clock"]) < 1.0:
            return None
        state["clock"] = float(state["clock"]) - 1.0
        nxt = int(state["index"]) + 1
        if nxt > last:
            if args.headless:
                # Nothing to look at and no button: finish the pass and stop.
                state["playing"] = False
                return None
            nxt = 0
        return float(nxt)

    if args.headless:
        print()
        print("headless: walked the plan, nothing drawn.")
    else:
        elements = swift_app.require_viz().swift.Elements
        readout = swift_app.add_readout(env, readout_lines(0))
        progress_slider = swift_app.add_slider(
            env,
            low=0.0,
            high=float(last),
            value=0.0,
            label="step",
            step=1.0,
            precision=0,
            name="step",
            elements=elements,
        )
        # The speed slider has no ``on_change``: ``on_frame`` polls it, because the
        # loop follows one slider and this one steers the clock rather than the scene.
        speed_slider = swift_app.add_slider(
            env,
            low=_SPEED_RANGE[0],
            high=_SPEED_RANGE[1],
            value=float(state["speed"]),
            label="speed",
            step=0.5,
            unit="steps/s",
            precision=1,
            name="speed",
            elements=elements,
        )

        def toggle_play(_value: object = None) -> None:
            """Start or stop the run, and say which state the button is in."""
            state["playing"] = not state["playing"]
            if state.get("button") is not None:
                state["button"].label = "pause" if state["playing"] else "play"

        state["button"] = swift_app.add_button(env, "play", toggle_play, elements=elements)
        swift_app.add_camera_radio(env, initial=args.camera, elements=elements)
        print()
        print(
            f"interactive: step slider 1 to {last + 1}, play/pause, a speed slider "
            f"{_SPEED_RANGE[0]:.0f} to {_SPEED_RANGE[1]:.0f} steps/s,"
        )
        print("             a camera radio, and a per-step readout of the trajectory.")

    swift_app.interaction_loop(
        env,
        slider=progress_slider,
        initial=0.0,
        on_change=show,
        on_frame=on_frame,
        steps=args.steps,
        dt=_FRAME_DT,
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
