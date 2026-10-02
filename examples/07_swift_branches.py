#!/usr/bin/env python3
"""Example 07 -- all eight branches in 3D, and the four the published code keeps.

The repository's headline finding is that the arm has eight kinematic branches and
the published implementation returns four, because it evaluates one root of the
``STEP2`` elbow quadratic.  ``demos/branches.html`` shows that as a fan of stick
figures in the browser; this is the same measurement with the real robot geometry,
where the two elbow roots are visibly *not* mirror images of each other -- the
Panda's 0.0825 m shoulder offset is what breaks the symmetry.

What it draws:

* the **generating configuration** in solid colour;
* every other in-limit solution for the same pose and the same joint 7, as a
  low-alpha ghost, coloured by which elbow root it came from;
* the arm plane of the selected solution, so the two roots can be compared by eye;
* the pose marker, which never moves: every drawn configuration reaches it.

The readout is the measurement, not decoration: for the pose on screen it prints
how many solutions exist in total, how many are on the first elbow root, and
whether the generating configuration is among them -- which, for the
``eight_branches`` and ``second_root`` poses, it is not.

Run it with::

    python examples/07_swift_branches.py
    python examples/07_swift_branches.py --pose second_root
    python examples/07_swift_branches.py --headless --pose near_singular
    python examples/07_swift_branches.py --pose eight_branches --joint-axes

The viewer needs the optional ``viz`` extra (``uv sync --extra viz``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import franka_ik as fk  # noqa: E402
from franka_ik import analysis, model, swift_app  # noqa: E402
from franka_ik.swift_viz import (  # noqa: E402
    ArmSkeleton,
    add_cloud,
    add_shapes,
    apply_camera,
    arm_plane_outline,
    launch_env,
    tool_axes,
)

#: Ghost opacity for the solutions that are not the generating configuration.
_GHOST_ALPHA = 0.22

#: Colour of a solution on the first (``+``) and second (``-``) elbow root, and
#: of the generating configuration, as ``[r, g, b]`` in ``[0, 1]``.
_COLOUR_PLUS = [0.20, 0.45, 0.85]
_COLOUR_MINUS = [0.85, 0.35, 0.15]
_COLOUR_TARGET = [0.15, 0.65, 0.30]


def _degrees_twice_turns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint difference in degrees, with whole turns removed."""
    delta = np.degrees(np.asarray(a)) - np.degrees(np.asarray(b))
    return (delta + 180.0) % 360.0 - 180.0


def _is_same(q: np.ndarray, target: np.ndarray, tol_deg: float = 1e-6) -> bool:
    """Whether two configurations differ only by whole turns."""
    return bool(np.all(np.abs(_degrees_twice_turns(q, target)) < tol_deg))


def main(argv: List[str] | None = None) -> int:
    """Run the example."""
    parser = swift_app.common_parser(
        __doc__.splitlines()[0],
        formatter_class=__import__("argparse").ArgumentDefaultsHelpFormatter,
    )
    args = parser.parse_args(argv)
    swift_app.configure_logging(args)

    q, q7 = swift_app.resolve_pose(args)
    pose = fk.fk_flange(q)
    solutions = fk.solve(pose, q7, within_limits_only=True)
    on_first_root = [s for s in solutions if s.q4_root == analysis.PUBLISHED_Q4_ROOT]
    others = [s for s in solutions if s.q4_root != analysis.PUBLISHED_Q4_ROOT]
    recovered = any(_is_same(s.q, q) for s in solutions)
    recovered_by_published = any(_is_same(s.q, q) for s in on_first_root)

    print(f"pose    : {args.pose}   joint 7 = {np.degrees(q7):.3f} deg")
    print(f"q       : {swift_app.degrees(q)} deg")
    print(f"in-limit solutions at this joint 7 : {len(solutions)}")
    print(f"  on the first ('+') elbow root    : {len(on_first_root)}")
    print(f"  on the second ('-') elbow root   : {len(others)}")
    print(f"  the generating configuration is among them  : {recovered}")
    print(f"  ... and the published four would return it  : {recovered_by_published}")
    print(f"manipulability : {fk.manipulability(q):.3e}")
    if not recovered_by_published:
        print()
        print("The published four-branch subset cannot reach this configuration:")
        print("it sits on the second root of the elbow quadratic, which that code")
        print("never evaluates.  This is the finding in docs/branch_analysis.md.")

    arm, urdf = swift_app.load_arm_for(args)
    from franka_ik.swift_viz import check_kinematics

    worst = check_kinematics(arm) if (urdf is not None and arm is not None) else None
    env = launch_env(headless=args.headless, browser=args.browser)
    swift_app.print_model_summary(arm, urdf, worst=worst, model=args.model)

    # The generating configuration, drawn solid, plus one ghost per other
    # solution.  Ghosts exist only to be looked at, so they are thinner and
    # transparent; their radii are set at construction, because Swift's headless
    # client does not acknowledge a radius written afterwards.
    # The main arm follows --model; the ghosts stay skeletons, because several
    # overlapping collision shells would be unreadable and colouring them by elbow
    # root is the whole point.
    target = swift_app.build_arm_shapes(args, arm=arm)
    target.update(q)
    ghosts = []
    for solution in solutions:
        if _is_same(solution.q, q):
            continue
        ghost = ArmSkeleton(
            radius_scale=0.8 * args.radius_scale,
            alpha=_GHOST_ALPHA,
            colour=_COLOUR_PLUS
            if solution.q4_root == analysis.PUBLISHED_Q4_ROOT
            else _COLOUR_MINUS,
        )
        ghost.update(solution.q)
        ghosts.append(ghost)

    pose_marker = tool_axes(q, length=0.14)
    plane = arm_plane_outline(q)

    # The solid arm moves, so it goes in one shape at a time; the ghosts are a fan
    # that is rebuilt whenever joint 7 moves, and adding ~120 shapes one at a time is
    # ~120 blocking round trips (docs/browser_debugging.md section 2.5) -- so the fan
    # is one group, added and removed in one call each.
    add_shapes(env, target.shapes)
    ghost_shapes = [shape for ghost in ghosts for shape in ghost.shapes]
    # ``--pose ready`` has one in-limit solution, so there is no fan to draw; ``add_cloud``
    # refuses an empty group rather than adding nothing quietly.
    ghost_group = add_cloud(env, ghost_shapes, name="ghosts") if ghost_shapes else None
    add_shapes(env, [pose_marker])
    if plane is not None:
        add_shapes(env, [plane])

    if args.headless:
        print()
        print(
            f"headless: built a {target.kind} arm and {len(ghosts)} ghost skeletons "
            "for the other solutions, nothing drawn."
        )

    # ---------------------------------------------------------------- UI
    # The browser owns the interaction: a slider sweeps joint 7 across its in-limit
    # window and the arm, the ghosts and the readout all follow.  A solution is
    # chosen by *index* into the current solution list, because the list changes
    # length as joint 7 moves.
    state = {
        "index": 0,
        "playing": False,
        "solutions": solutions,
        "min": None,
        "max": None,
        "q7": float(q7),
    }

    def describe(index: int, values) -> None:
        """Update the readout for solution ``index`` of ``values``."""
        if readout is None or not values:
            return
        index = max(0, min(index, len(values) - 1))
        solution = values[index]
        first = solution.q4_root == analysis.PUBLISHED_Q4_ROOT
        swift_app.set_readout(
            readout,
            [
                f"q7 = {np.degrees(solution.q[6]):.2f} deg",
                f"solution {index + 1} of {len(values)} &nbsp; elbow root "
                f"<b>{'+' if first else '-'}</b> &nbsp;"
                f" {'published four would return it' if first else 'published four MISS it'}",
                f"q (deg) = [{swift_app.degrees(solution.q, 1).strip('[]')}]",
                f"pose residual = {float(np.abs(fk.fk_flange(solution.q) - pose).max()):.1e}"
                f" &nbsp; manipulability = {fk.manipulability(solution.q):.4e}",
            ],
        )

    def show(index: int) -> None:
        """Draw solution ``index`` on the solid arm and re-label the ghosts."""
        values = state["solutions"]
        if not values:
            # A joint 7 nothing reaches inside the limits: no arm to draw, and saying
            # so beats leaving the previous solution on screen as if it were current.
            if readout is not None:
                swift_app.set_readout(readout, [f"q7 = {np.degrees(state['q7']):.2f} deg", "no in-limit solution"])
            return
        index = max(0, min(index, len(values) - 1))
        state["index"] = index
        chosen = values[index].q
        target.update(chosen)
        # ``state["ghosts"]``, not the list built at start-up: ``resync`` replaces it
        # when the solution set changes, and zipping the old one against the new
        # solutions raises ``AttributeError: 'IkSolution' object has no attribute 'q'``
        # the first time the slider moves.  (Found by driving the slider in a browser.)
        for ghost, solution in zip(state["ghosts"], values, strict=False):
            ghost.update(solution.q)
        # The pose marker and the arm plane move with the selection: the plane is
        # what the two elbow roots differ in, so it has to follow.
        pose_marker.T = model.fk_tool(chosen)
        plane = arm_plane_outline(chosen)
        if plane is not None:
            add_shapes(env, [plane])
            state["planes"].append(plane)
        while len(state["planes"]) > 1:
            env.remove(state["planes"].pop(0))
        describe(index, values)

    def resync(q7_value: float) -> None:
        """Re-solve at a new joint 7 and rebuild the ghosts for that set."""
        nonlocal ghost_group
        state["q7"] = float(np.radians(q7_value))
        values = fk.solve(pose, state["q7"], within_limits_only=True)
        state["solutions"] = values
        if ghost_group is not None:
            env.remove(ghost_group)
            ghost_group = None
        state["ghosts"] = []
        for solution in values:
            if not values or _is_same(solution.q, values[0].q):
                continue
            ghost = ArmSkeleton(
                radius_scale=0.8 * args.radius_scale,
                alpha=_GHOST_ALPHA,
                colour=_COLOUR_PLUS
                if solution.q4_root == analysis.PUBLISHED_Q4_ROOT
                else _COLOUR_MINUS,
            )
            ghost.update(solution.q)
            state["ghosts"].append(ghost)
        shapes = [shape for ghost in state["ghosts"] for shape in ghost.shapes]
        if shapes:
            ghost_group = add_cloud(env, shapes, name="ghosts")
        show(0)

    def _toggle_play(_value: object = None) -> None:
        """Start or stop the sweep, and say which state the button is in."""
        state["playing"] = not state["playing"]
        # ``state``, not a closure over the local: Swift calls a button's callback
        # once as the page attaches, and the name is not bound until ``add_button``
        # returns.
        if state.get("button") is not None:
            state["button"].label = "pause" if state["playing"] else "play"

    def _advance(current: float) -> float | None:
        """The next joint 7 while the sweep is running, or ``None`` when paused."""
        if not state["playing"]:
            return None
        wanted = current + 0.75
        if wanted > (state["max"] or wanted):
            wanted = state["min"] or wanted
        return wanted

    state["ghosts"] = ghosts
    state["planes"] = [plane] if plane is not None else []
    readout = None
    slider = None
    if not args.headless:
        elements = swift_app.require_viz().swift.Elements
        low, high = swift_app.joint7_window(pose, fallback=float(q7))
        state["min"], state["max"] = low, high
        readout = swift_app.add_readout(env, ["q7 = ..."])
        # ``add_slider`` reads its own ``.value`` back and writes the initial value
        # to the browser (whose range input clamps the value it is built with), so no
        # example has to know about either trap.
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
        state["button"] = swift_app.add_button(env, "play", _toggle_play, elements=elements)
        swift_app.add_camera_radio(env, initial=args.camera, elements=elements)
        apply_camera(env, args.camera)
        describe(0, solutions)
        print()
        print(f"interactive: joint-7 slider {low:.1f} to {high:.1f} deg, play/pause button,")
        print("             camera radio, and a readout of the selected solution.")

    swift_app.interaction_loop(
        env,
        slider=slider,
        initial=float(np.degrees(q7)),
        on_change=resync,
        on_frame=_advance,
        steps=args.steps,
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
