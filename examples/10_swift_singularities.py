#!/usr/bin/env python3
"""Example 10 -- joint 2 at zero, the singularity the solver cannot cross.

At ``q₂ = 0`` joints 1 and 3 become coaxial: the shoulder loses a degree of freedom,
the pose no longer determines them separately, and the closed form's shoulder
reconstruction collapses -- it reads joint 1 from ``r₀₃[1,1] = −sin θ₁ sin θ₂``,
which is zero for *any* ``θ₁`` when ``sin θ₂ = 0``.  Every branch is then checked
against the forward kinematics before it is returned, so the wrong ones are dropped
rather than returned; if they were the only in-limit ones, the solver reports no
solution for a pose the arm demonstrably reaches.  ``docs/limitations.md`` §12 is
the full account, and this example is its interactive form.

The slider is the **joint-2 offset**, in a range narrow enough to find the failure
window by hand: ``±0.01`` degrees with a step of ``1e-5``, so the window -- ``1e-4``
degrees wide, measured -- is about one percent of the slider's travel and one
arrow-key press is a tenth of it.  The four rows printed below bracket it, the
``offset`` radio jumps straight to each of them, and the readout prints ``q₂`` to
five decimals, because three would show the same number either side of the edge.
Two arms are on screen: the singular configuration, faded, and the perturbed one,
which is the one that moves.

Three things are drawn at once:

* the **generating configuration**, which is inside the joint limits and whose pose
  is exact;
* the same configuration perturbed by the slider's **joint-2 offset**, so the
  transition from "no solution" to "two solutions" is visible rather than asserted;
* a **manipulability ellipsoid** at the wrist.

Those last two belong together, and the readout is what keeps them apart, because
they are *different kinds* of singularity and the numbers say so.  ``q₂ = 0`` is a
**coordinate** singularity: manipulability there is 7.5e-2, as healthy as the
``ready`` pose, because the arm is nowhere near losing a direction -- what collapses
is the derivation's own reconstruction of the shoulder.  The sample pose named
``near_singular`` is the other kind (**kinematic**): manipulability 2.7e-4 and
smallest singular value 6.0e-4, where the Jacobian really is about to lose a
direction and branches collapse onto each other.  Run both and compare the
ellipsoids; ``--case sample --pose near_singular`` is the second.

The readout for the drawn configuration is manipulability, the smallest singular
value of the Jacobian, and what the solver returns.

Run it with::

    python examples/10_swift_singularities.py
    python examples/10_swift_singularities.py --delta 1e-4
    python examples/10_swift_singularities.py --case sample --pose near_singular
    python examples/10_swift_singularities.py --headless

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
    add_cloud,
    add_shapes,
    apply_camera,
    launch_env,
    manipulability_ellipsoid,
    tool_axes,
    update_manipulability_ellipsoid,
)

#: The configuration from ``docs/limitations.md`` §12: inside the joint limits,
#: with joint 2 exactly zero.  Solving for its own pose returns nothing.
Q2_ZERO_POSE_DEG = [36.0, 0.0, -24.0, -100.0, 18.0, 126.0, 48.0]

#: Joint-2 offsets to **print**, in degrees.  ``0.0`` is the singular configuration
#: itself.  ``1e-5`` is *inside* the failure window (measured: 0 in-limit solutions),
#: ``1e-4`` is the first value outside it that was measured (2 solutions), and
#: ``0.35`` is far enough away to look like an ordinary pose.  Printing them
#: together is what makes "knife edge" a measurement rather than a phrase.
_DEFAULT_OFFSETS = (0.0, 1e-5, 1e-4, 0.35)

#: The offsets the radio jumps to, which have to fit inside the slider's range.
_PRESET_OFFSETS = (0.0, 1e-5, 1e-4, 1e-3)

#: Half-width of the slider, in degrees.  Measured: the failure window is ``1e-4``
#: degrees wide, so ``0.01`` puts it at about one percent of the travel -- small
#: enough to drag through, large enough that a stray pixel is not the whole window.
_SLIDER_HALF_WIDTH_DEG = 0.01

#: Colours per drawn configuration: the faded reference, then the live arm.
_COLOUR_REFERENCE = [0.55, 0.55, 0.60]
_COLOUR_LIVE = [0.85, 0.25, 0.15]


def report(q: np.ndarray, label: str) -> None:
    """Print what the solver returns for the pose of ``q``.

    Args:
        q: In-limit configuration, radians.
        label: Name of the configuration, for the row.
    """
    pose = model.fk_flange(q)
    q7 = float(q[6])
    in_limit = fk.solve(pose, q7, within_limits_only=True)
    every = fk.solve(pose, q7, within_limits_only=False)
    singular = np.linalg.svd(model.jacobian(q), compute_uv=False)
    delta = (
        np.degrees(np.asarray([s.q for s in every], dtype=float) - q) if every else np.zeros((0, 7))
    )
    delta = (delta + 180.0) % 360.0 - 180.0
    hits = int(np.count_nonzero(np.all(np.abs(delta) < 1e-6, axis=1))) if len(delta) else 0
    # Five decimals, not three: the failure window is 1e-4 degrees wide, so a
    # readout that rounds to milli-degrees would print the same number for the
    # singular configuration and for one just outside it.
    print(
        f"{label:>18}  q2={np.degrees(q[1]):9.5f}°  manip={model.manipulability(q):.3e}  "
        f"σ_min={singular[-1]:.2e}  in-limit={len(in_limit)}  candidates={len(every)}  "
        f"recovers target={bool(hits)}"
    )


def solve_row(q: np.ndarray) -> Tuple[int, int, bool, float, float]:
    """What the solver returns for the pose of ``q``, as numbers.

    The readout and the printed table have to agree, so both go through the same
    measurement: ``(in-limit, candidates, recovers the target, manipulability,
    smallest singular value)``.
    """
    pose = model.fk_flange(q)
    q7 = float(q[6])
    every = fk.solve(pose, q7, within_limits_only=False)
    in_limit = fk.solve(pose, q7, within_limits_only=True)
    singular = np.linalg.svd(model.jacobian(q), compute_uv=False)
    delta = (
        np.degrees(np.asarray([s.q for s in every], dtype=float) - q) if every else np.zeros((0, 7))
    )
    delta = (delta + 180.0) % 360.0 - 180.0
    hits = int(np.count_nonzero(np.all(np.abs(delta) < 1e-6, axis=1))) if len(delta) else 0
    return len(in_limit), len(every), bool(hits), model.manipulability(q), float(singular[-1])


def main(argv: List[str] | None = None) -> int:
    """Run the example."""
    parser = swift_app.common_parser(
        __doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        metavar="DEG",
        help="joint-2 offset the slider starts at; default 0, the singularity itself",
    )
    parser.add_argument(
        "--ellipsoid-scale",
        type=float,
        default=0.10,
        metavar="M",
        help="metres per unit singular value for the manipulability ellipsoids",
    )
    parser.add_argument(
        "--case",
        choices=("q2_zero", "sample"),
        default="q2_zero",
        help="'q2_zero' for the singularity in docs/limitations.md §12, "
        "'sample' for the configuration chosen by --pose",
    )
    args = parser.parse_args(argv)
    swift_app.configure_logging(args)

    if args.case == "q2_zero":
        base = np.radians(np.asarray(Q2_ZERO_POSE_DEG, dtype=float))
    else:
        base, _ = swift_app.resolve_pose(args)

    print(
        f"configuration : joint 2 = {np.degrees(base[1]):.3f}°"
        + ("  (docs/limitations.md §12)" if args.case == "q2_zero" else f"  (pose {args.pose})")
    )
    print(f"pose reached  : {np.round(model.fk_flange(base)[:3, 3], 4).tolist()} m")
    print()
    for offset in _DEFAULT_OFFSETS:
        moved = base.copy()
        moved[1] = base[1] + np.radians(offset)
        report(moved, f"q2 + {offset:g}°")
    print()

    if args.case == "q2_zero":
        q0 = base
        pose = model.fk_flange(q0)
        if not fk.solve(pose, float(q0[6]), within_limits_only=True):
            print("The first row returns no in-limit solution for a pose the arm reaches:")
            print(f"  classify_failure -> {fk.classify_failure(pose, float(q0[6]))!r}")
            print("  Every branch fails, in two different ways:")
            for solution in fk.solve(pose, float(q0[6]), within_limits_only=False):
                degrees = np.degrees(solution.q)
                residual = float(np.abs(model.fk_flange(solution.q) - pose).max())
                if residual > 1e-9:
                    why = "wrong pose"
                else:
                    outside = [
                        str(index + 1)
                        for index in range(model.NUM_JOINTS)
                        if degrees[index] < fk.LOWER_LIMITS_DEG[index] - 1e-9
                        or degrees[index] > fk.UPPER_LIMITS_DEG[index] + 1e-9
                    ]
                    why = "outside joint " + ", ".join(outside)
                print(
                    f"    {solution.label:20s} q2={degrees[1]:9.3f}deg q4={degrees[3]:9.3f}deg "
                    f"residual={residual:.2e}  {why}"
                )
            print("  The branches the limits accept are the ones the collapse got wrong;")
            print("  the ones that are exact are outside the limits.  Nothing is both, so")
            print("  nothing is returned.  docs/limitations.md section 12; the rows below")
            print("  show the recovery, and that it happens within 1e-4 degrees.")
            print()

    from franka_ik.swift_viz import check_kinematics

    arm, urdf = swift_app.load_arm_for(args)
    worst = check_kinematics(arm) if (urdf is not None and arm is not None) else None
    env = launch_env(headless=args.headless, browser=args.browser)
    swift_app.print_model_summary(arm, urdf, worst=worst, model=args.model)
    print()

    # The scene: a faded reference at the base configuration and the live arm at the
    # slider's offset.  ``--delta 0`` (the default) starts them on top of each other,
    # which is the singular case itself.
    state: Dict[str, object] = {"offset": 0.0 if args.delta is None else float(args.delta)}
    reference = base.copy()
    moved = base.copy()
    moved[1] = base[1] + np.radians(float(state["offset"]))

    reference_arm = swift_app.build_arm_shapes(args, arm=arm, alpha=0.25)
    reference_arm.update(reference)
    live_arm = swift_app.build_arm_shapes(args, arm=arm)
    live_arm.update(moved)
    if reference_arm.kind == "skeleton":
        # A mesh's opacity is fixed at construction (hence ``alpha`` above); the
        # skeleton's is not, so it is written per shape here.
        for shape in reference_arm.shapes:
            if hasattr(shape, "color"):
                shape.color = [*_COLOUR_REFERENCE, 1.0]
            shape.opacity = 0.30
        for shape in live_arm.shapes:
            if hasattr(shape, "color"):
                shape.color = [*_COLOUR_LIVE, 1.0]

    ellipsoid = manipulability_ellipsoid(moved, scale=args.ellipsoid_scale)
    reference_frame = tool_axes(reference, length=0.12)
    live_frame = tool_axes(moved, length=0.12)
    # One group per arm: ~17 shapes each, and with a browser attached every
    # ``add_shape`` is a blocking round trip (docs/browser_debugging.md section 2.5).
    # Both arms stay movable -- a group reports its parts' current poses each frame.
    add_cloud(env, reference_arm.shapes, name="reference")
    add_cloud(env, live_arm.shapes, name="live")
    add_shapes(env, [ellipsoid, reference_frame, live_frame])
    apply_camera(env, args.camera)

    readout: Optional[object] = None
    slider: Optional[object] = None

    def readout_lines() -> List[str]:
        """The live arm's measurement, and the verdict the table implies.

        Five decimals on ``q2`` for the reason the table uses five: the window is
        ``1e-4`` degrees wide, and three decimals would hide its whole width.  ASCII
        only, because the text crosses a websocket into an HTML label.
        """
        in_limit, candidates, recovers, manip, sigma = solve_row(moved)
        offset = float(state["offset"])
        verdict = (
            "INSIDE the failure window: nothing the solver returns is both exact and "
            "in range"
            if in_limit == 0
            else f"solved ({in_limit} in-limit configuration(s))"
        )
        return [
            f"q2 = {np.degrees(moved[1]):.5f} deg &nbsp; (base {np.degrees(base[1]):.3f} + "
            f"offset {offset:.5f})",
            f"in-limit {in_limit} &nbsp; candidates {candidates} &nbsp; recovers the "
            f"generating configuration: {recovers}",
            f"manipulability = {manip:.3e} &nbsp; sigma_min = {sigma:.2e}",
            f"{verdict}",
            "the window is 1e-4 deg wide, the slider step is 1e-5: one arrow-key press "
            "is a tenth of it",
        ]

    def on_offset(value: float) -> None:
        """Move the live arm to base + offset and re-measure."""
        state["offset"] = float(value)
        moved[1] = base[1] + np.radians(float(value))
        live_arm.update(moved)
        update_manipulability_ellipsoid(ellipsoid, moved)
        live_frame.T = model.fk_tool(moved)
        if readout is not None:
            swift_app.set_readout(readout, readout_lines())

    if args.headless:
        print(
            f"headless: built {len(reference_arm.shapes) + len(live_arm.shapes)} "
            f"{live_arm.kind} shapes for two configurations, one ellipsoid and two tool "
            "frames, nothing drawn."
        )
    else:
        elements = swift_app.require_viz().swift.Elements
        readout = swift_app.add_readout(env, readout_lines())
        on_offset(float(state["offset"]))
        slider = swift_app.add_slider(
            env,
            low=-_SLIDER_HALF_WIDTH_DEG,
            high=_SLIDER_HALF_WIDTH_DEG,
            value=float(state["offset"]),
            label="joint 2 offset",
            step=1e-5,
            unit="deg",
            precision=5,
            name="offset",
            elements=elements,
        )

        def on_preset(index: int) -> None:
            """Jump the slider to one of the measured offsets."""
            if 0 <= index < len(_PRESET_OFFSETS):
                slider.value = _PRESET_OFFSETS[index]

        swift_app.add_radio(
            env,
            label="offset",
            options=[f"{value:g}" for value in _PRESET_OFFSETS],
            on_select=on_preset,
            checked=0,
            name="preset",
            elements=elements,
        )
        swift_app.add_camera_radio(env, initial=args.camera, elements=elements)
        print(
            f"interactive: joint-2 offset slider +/-{_SLIDER_HALF_WIDTH_DEG:g} deg with step "
            "1e-5, a preset radio, a camera radio,"
        )
        print("             and a readout of what the solver returns for the moved arm.")

    swift_app.interaction_loop(
        env,
        slider=slider,
        initial=float(state["offset"]),
        on_change=on_offset,
        steps=args.steps,
        tolerance=1e-9,
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
