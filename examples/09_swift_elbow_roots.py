#!/usr/bin/env python3
"""Example 09 -- the two elbow roots, side by side.

The Panda's elbow equation is a quadratic in ``tan(θ₄/2)``, and its two roots are
*not* mirror images: the 0.0825 m shoulder offset biases them.  At the sampled
pose in ``docs/branch_analysis.md`` §3.1 the two roots give ``θ₄ = −70.38°`` and
``+16.87°``, where a true S-R-S arm would give ``±34.79°``.  This example draws one
configuration from each root at the same pose and the same joint 7, plus every
in-limit solution in between, so the asymmetry is something you look at.

``--show all`` puts every in-limit solution on screen at once (up to eight);
``--show roots`` keeps one per root, which is the readable version; ``--sweep``
instead walks joint 7 across its in-limit window, which is the self-motion at a
fixed pose.

Interactively the joint-7 slider re-solves, the ``show`` radio switches between the
two views, and the readout prints ``θ₄`` for every configuration drawn, so the
asymmetry is a number before it is a picture.  Dragging joint 7 is what makes the
point of the example sharp: the *pose does not move*, and yet the number of
in-limit configurations -- and which roots they sit on -- changes as it goes.

The numbers printed for each drawn configuration are its pose residual -- computed
with the forward kinematics, not assumed -- so "all of these reach the same pose"
is verified on screen rather than claimed.

Run it with::

    python examples/09_swift_elbow_roots.py --show roots
    python examples/09_swift_elbow_roots.py --show all --pose eight_branches
    python examples/09_swift_elbow_roots.py --sweep --steps 200
    python examples/09_swift_elbow_roots.py --headless

The viewer needs the optional ``viz`` extra (``uv sync --extra viz``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import franka_ik as fk  # noqa: E402
from franka_ik import analysis, swift_app  # noqa: E402
from franka_ik.swift_viz import (  # noqa: E402
    add_cloud,
    add_shapes,
    apply_camera,
    arm_plane_outline,
    launch_env,
)

#: One colour per elbow root, so the two halves of the branch set are told apart
#: by eye: the first root the published code keeps, the second it drops.
_ROOT_COLOURS = {
    analysis.PUBLISHED_Q4_ROOT: [0.20, 0.45, 0.85],
    0: [0.85, 0.35, 0.15],
}


def _root_colour(q4_root: int) -> List[float]:
    """Colour for an elbow root, defaulting to the second-root colour."""
    return _ROOT_COLOURS.get(int(q4_root), _ROOT_COLOURS[0])


def _root_name(q4_root: int) -> str:
    """``q4+`` for the root the published code keeps, ``q4-`` for the other."""
    return "q4+" if int(q4_root) == analysis.PUBLISHED_Q4_ROOT else "q4-"


def self_motion(
    pose: np.ndarray,
    reference: np.ndarray,
    q7_span: float,
    *,
    steps: int = 61,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sample joint 7 across the window that reaches ``pose`` in limits.

    The pose is fixed, so joint 7 is the only free parameter: this returns the
    values to sweep and the configuration at each one.  The window is not assumed
    -- it is found by asking the solver, which is the point, because it depends on
    the pose and is usually much narrower than the joint-7 range.

    The configuration at each value comes from :func:`franka_ik.solver.solve_closest`
    with the generating configuration as the reference, so the sweep tracks one
    continuous branch instead of jumping whenever a second solution appears.

    Args:
        pose: 4x4 target flange pose.
        reference: Configuration to stay close to, usually the one that made the
            pose; joint angles in radians.
        q7_span: Half-width to search around ``reference[6]``, in radians.
        steps: Sample count along that interval.

    Returns:
        ``(values, configurations)``.  ``values`` are in radians and may be shorter
        than ``steps`` where joint 7 leaves the limits or the pose becomes
        unreachable on this branch.
    """
    reach_lower, reach_upper = fk.LOWER_LIMITS_DEG, fk.UPPER_LIMITS_DEG
    values: List[float] = []
    configurations: List[np.ndarray] = []
    for q7 in np.linspace(reference[6] - q7_span, reference[6] + q7_span, steps):
        if not np.radians(reach_lower[6]) <= q7 <= np.radians(reach_upper[6]):
            continue
        closest = fk.solve_closest(pose, float(q7), reference=reference)
        if closest is None:
            continue
        values.append(float(q7))
        configurations.append(np.asarray(closest.q, dtype=float))
    return np.asarray(values), configurations


def one_per_root(solutions: Sequence[object]) -> List[object]:
    """The first solution on each elbow root, which is the readable view.

    Kept separate from the drawing code because it is also the definition the
    printed table uses: ``--show all`` is the measurement, ``--show roots`` is what
    a reader can actually see, and the two have to agree about which solutions the
    second one is a subset of.
    """
    grouped: Dict[int, List[object]] = {}
    for solution in solutions:
        grouped.setdefault(int(solution.q4_root), []).append(solution)
    return [group[0] for group in grouped.values()]


def main(argv: List[str] | None = None) -> int:
    """Run the example."""
    parser = swift_app.common_parser(
        __doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--show",
        choices=("roots", "all"),
        default="roots",
        help="one configuration per elbow root, or every in-limit solution",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="walk joint 7 across its in-limit window instead of drawing the roots",
    )
    parser.add_argument(
        "--q7-span",
        type=float,
        default=15.0,
        metavar="DEG",
        help="half-width of the joint-7 window used by --sweep",
    )
    args = parser.parse_args(argv)
    swift_app.configure_logging(args)

    q, q7 = swift_app.resolve_pose(args)
    pose = fk.fk_flange(q)

    from franka_ik.swift_viz import check_kinematics

    arm, urdf = swift_app.load_arm_for(args)
    worst = check_kinematics(arm) if (urdf is not None and arm is not None) else None
    env = launch_env(headless=args.headless, browser=args.browser)
    swift_app.print_model_summary(arm, urdf, worst=worst, model=args.model)
    print()

    if args.sweep:
        return _sweep(args, env, arm, pose, q)

    solutions = fk.solve(pose, q7, within_limits_only=True)
    if not solutions:
        print(f"no in-limit solution at joint 7 = {np.degrees(q7):.3f} deg")
        env.close()
        return 0

    print(f"pose            : {args.pose}, joint 7 = {np.degrees(q7):.3f} deg")
    first_root = len([s for s in solutions if s.q4_root == analysis.PUBLISHED_Q4_ROOT])
    print(
        f"in-limit solutions: {len(solutions)}  "
        f"(first root {first_root}, second root {len(solutions) - first_root})"
    )
    print()
    print(f"{'drawn':>5}  {'label':22s} {'q4 (deg)':>9}  {'pose residual':>13}")
    initial_drawn = one_per_root(solutions) if args.show == "roots" else list(solutions)
    for index, solution in enumerate(initial_drawn):
        residual = float(np.abs(fk.fk_flange(solution.q) - pose).max())
        print(
            f"{index:5d}  {solution.label:22s} {np.degrees(solution.q[3]):9.3f}  {residual:13.2e}"
        )
    if len(initial_drawn) > 1:
        angles = [float(np.degrees(s.q[3])) for s in initial_drawn]
        print()
        print(
            f"θ₄ spans {min(angles):.2f} to {max(angles):.2f} deg across the drawn "
            "configurations; a true S-R-S arm would give a symmetric pair here."
        )

    # ------------------------------------------------------------ the scene
    # The drawn set changes size as joint 7 moves, and its colours come from the
    # elbow root each solution sits on.  Swift does not re-send a shape whose colour
    # changed underneath it, so the pool is rebuilt when the *keys* change -- count
    # or root -- and only moved (``update``) when the same configurations reshuffle
    # to new angles.  That keeps a mesh-mode rebuild off the slider's hot path.
    state: Dict[str, object] = {
        "show": args.show,
        "q7": float(q7),
        "solutions": solutions,
        "arms": [],
        "keys": [],
        "planes": [],
        # The handle of the group the drawn arms live in, or ``None`` before the first
        # ``sync``.  A group is one message to add and one call to remove, which is
        # what keeps a rebuild off the slider's hot path
        # (``docs/browser_debugging.md`` §2.5).
        "group": None,
    }

    def drawn_from(values: Sequence[object]) -> List[object]:
        """The solutions ``--show`` / the radio currently asks for."""
        if state["show"] == "roots":
            return one_per_root(values)
        return list(values)

    def readout_lines(chosen: Sequence[object]) -> List[str]:
        """The readout: joint 7, the solution count, and theta 4 for each drawn arm.

        ASCII only, because the text crosses a websocket into an HTML label.
        """
        values = state["solutions"]
        lines = [
            f"joint 7 = {np.degrees(state['q7']):.2f} deg &nbsp; in-limit: {len(values)} "
            f"&nbsp; on the published root: "
            f"{len([s for s in values if s.q4_root == analysis.PUBLISHED_Q4_ROOT])}",
            f"drawn: {len(chosen)} ({state['show']})",
        ]
        for index, solution in enumerate(chosen):
            lines.append(
                f"{_root_name(solution.q4_root)} #{index}: q4 = "
                f"{np.degrees(solution.q[3]):+.2f} deg &nbsp; residual "
                f"{float(np.abs(fk.fk_flange(solution.q) - pose).max()):.1e}"
            )
        angles = [float(np.degrees(s.q[3])) for s in chosen]
        if len(angles) > 1:
            lines.append(
                f"q4 spans {min(angles):+.2f} to {max(angles):+.2f} deg; a true S-R-S "
                "arm would be symmetric here"
            )
        return lines

    def sync() -> None:
        """Draw the current solution set, rebuilding only when it changed shape."""
        chosen = drawn_from(state["solutions"])
        keys = [int(solution.q4_root) for solution in chosen]
        if keys != state["keys"]:
            if state["group"] is not None:
                env.remove(state["group"])
                state["group"] = None
            state["arms"] = []
            state["keys"] = keys
            for solution in chosen:
                one = swift_app.build_arm_shapes(args, arm=arm)
                if one.kind == "skeleton":
                    for shape in one.shapes:
                        if hasattr(shape, "color"):
                            shape.color = [*_root_colour(solution.q4_root), 1.0]
                        shape.opacity = 1.0 if len(chosen) == 1 else 0.45
                state["arms"].append(one)
            shapes = [shape for one in state["arms"] for shape in one.shapes]
            if shapes:
                state["group"] = add_cloud(env, shapes, name="arms")
        for one, solution in zip(state["arms"], chosen, strict=False):
            one.update(solution.q)
        # The arm plane is what the two roots differ in, so it follows the arm; one
        # per drawn configuration, rebuilt rather than pooled because there are four
        # points in it.
        for plane in state["planes"]:
            env.remove(plane)
        state["planes"] = []
        for solution in chosen:
            plane = arm_plane_outline(solution.q)
            if plane is not None:
                add_shapes(env, [plane])
                state["planes"].append(plane)
        if readout is not None:
            swift_app.set_readout(readout, readout_lines(chosen))

    readout: Optional[object] = None
    slider: Optional[object] = None
    if args.headless:
        sync()
        print()
        print(
            f"headless: built {len(state['arms'])} {state['arms'][0].kind} arms for one pose, "
            "nothing drawn."
        )
    else:
        elements = swift_app.require_viz().swift.Elements
        readout = swift_app.add_readout(env, readout_lines(drawn_from(solutions)))
        sync()

        def on_q7(value: float) -> None:
            """Re-solve at a new joint 7 and redraw."""
            state["q7"] = float(np.radians(value))
            state["solutions"] = fk.solve(pose, state["q7"], within_limits_only=True)
            sync()

        def on_show(index: int) -> None:
            """Switch between one configuration per root and the whole set."""
            state["show"] = ("roots", "all")[index]
            sync()

        low, high = swift_app.joint7_window(pose, fallback=float(q7))
        slider = swift_app.add_slider(
            env,
            low=low,
            high=high,
            value=float(np.degrees(q7)),
            label="joint 7",
            step=0.5,
            unit="deg",
            precision=2,
            name="q7",
            elements=elements,
        )
        swift_app.add_radio(
            env,
            label="show",
            options=["roots", "all"],
            on_select=on_show,
            checked=("roots", "all").index(args.show),
            name="show",
            elements=elements,
        )
        swift_app.add_camera_radio(env, initial=args.camera, elements=elements)
        print()
        print(
            f"interactive: joint-7 slider {low:.1f} to {high:.1f} deg, a roots/all radio, "
            "a camera radio,"
        )
        print("             and a readout of theta 4 for every configuration drawn.")

    apply_camera(env, args.camera)
    swift_app.interaction_loop(
        env,
        slider=slider,
        initial=float(np.degrees(q7)),
        on_change=on_q7 if not args.headless else None,
        steps=args.steps,
    )
    env.close()
    return 0


def _sweep(
    args: argparse.Namespace, env: object, arm: object, pose: np.ndarray, q: np.ndarray
) -> int:
    """``--sweep``: walk joint 7 across its window at a fixed pose, one step per frame."""
    values, configurations = self_motion(pose, q, float(np.radians(args.q7_span)))
    if len(values) < 2:
        print(
            f"self-motion at one pose: the in-limit joint-7 window is too narrow to "
            f"sweep ({len(values)} sample(s)); try --q7-span 30"
        )
        env.close()
        return 0
    print(
        f"self-motion at one pose: joint 7 from {np.degrees(values[0]):.2f} to "
        f"{np.degrees(values[-1]):.2f} deg in {len(values)} steps"
    )
    arm_shapes = swift_app.build_arm_shapes(args, arm=arm)
    add_shapes(env, arm_shapes.shapes)
    apply_camera(env, args.camera)

    readout: Optional[object] = None
    if not args.headless:
        readout = swift_app.add_readout(env, ["sweep: ..."])
        swift_app.add_camera_radio(env, initial=args.camera)
        print("interactive: a camera radio and a per-step readout of the sweep.")

    residuals = [float(np.abs(fk.fk_flange(c) - pose).max()) for c in configurations]
    if args.headless:
        print(
            f"headless: {len(configurations)} configurations, worst residual {max(residuals):.2e}"
        )
    state: Dict[str, object] = {"index": 0, "plane": None}

    def on_frame(_value: float) -> None:
        """Advance one configuration per frame, and say where the sweep is."""
        index = int(state["index"]) % len(configurations)
        chosen = configurations[index]
        arm_shapes.update(chosen)
        # The arm plane is rebuilt rather than moved: ``sg.PolyLine`` says nothing
        # about where its points are, so there is no pose to write.  Four points per
        # frame is not worth pooling.
        if state["plane"] is not None:
            env.remove(state["plane"])
            state["plane"] = None
        plane = arm_plane_outline(chosen)
        if plane is not None:
            add_shapes(env, [plane])
            state["plane"] = plane
        if readout is not None:
            swift_app.set_readout(
                readout,
                [
                    f"step {index + 1} of {len(configurations)} &nbsp; joint 7 = "
                    f"{np.degrees(values[index]):.2f} deg",
                    f"q4 = {np.degrees(chosen[3]):+.2f} deg &nbsp; residual "
                    f"{residuals[index]:.1e} &nbsp; manipulability "
                    f"{fk.manipulability(chosen):.3e}",
                ],
            )
        state["index"] = index + 1

    swift_app.interaction_loop(env, initial=0.0, on_frame=on_frame, steps=args.steps, dt=0.08)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
