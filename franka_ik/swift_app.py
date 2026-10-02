"""Shared plumbing for the Swift examples: arguments, configs, scenes.

The examples under ``examples/`` are meant to be read, so everything they have in
common lives here: the command-line flags, the sample poses, the readouts and the
two or three lines each example needs to put a skeleton on screen.  What is left
in an example file is then only the idea that example is about.

Nothing here imports Swift at module level; :mod:`franka_ik.swift_viz` does that
lazily inside its functions, which is what keeps ``import franka_ik`` free of the
visualisation stack.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import model
from .solver import solve
from .swift_viz import (
    ArmSkeleton,
    add_shapes,
    apply_camera,
    camera_presets,
    find_panda_urdf,
    launch_env,
    link_collision_shapes,
    link_mesh_shapes,
    require_viz,
    update_link_collision_shapes,
    update_link_mesh_shapes,
)

__all__ = [
    "POSES",
    "CAMERA_CHOICES",
    "pose_labels",
    "common_parser",
    "resolve_pose",
    "configuration_reader",
    "ArmShapes",
    "build_arm_shapes",
    "load_arm_for",
    "joint7_window",
    "selection_index",
    "checkbox_flags",
    "label_for",
    "add_readout",
    "set_readout",
    "add_radio",
    "add_button",
    "add_camera_radio",
    "add_slider",
    "read_slider",
    "interaction_loop",
    "hold",
    "scene",
    "print_model_summary",
]

#: The camera presets the examples put on a radio, in the order they are listed.
CAMERA_CHOICES: Tuple[str, ...] = ("iso", "front", "side", "top")


class ArmShapes:
    """Either a capsule skeleton or the model's own collision geometry.

    The two look different and are driven differently, but every example wants the
    same two things from them -- a list of shapes to add, and a way to move them to
    a configuration -- so both arrive behind this one object.  It exists because the
    difference is real: the skeleton is built from
    :func:`franka_ik.model.joint_frames` and is exact by construction, while the
    collision geometry comes from the loaded model and is only as good as that
    model's match to this arm.

    Args:
        shapes: The shapes to add to Swift.
        skeleton: The :class:`~franka_ik.swift_viz.ArmSkeleton`, or ``None`` when
            the collision geometry is being drawn.
        arm: The loaded model, needed to move collision shapes, or ``None`` for a
            skeleton.
        kind: ``"skeleton"`` or ``"collision"``, for the readout.
    """

    def __init__(self, shapes, *, skeleton=None, arm=None, urdf=None, kind: str = "skeleton"):
        self._shapes = list(shapes)
        self._skeleton = skeleton
        self._arm = arm
        self._urdf = urdf
        self.kind = kind

    @property
    def shapes(self):
        """The shapes to hand to ``Swift.add``."""
        return list(self._shapes)

    def update(self, q) -> None:
        """Move everything to the configuration ``q`` (radians)."""
        if self._skeleton is not None:
            self._skeleton.update(q)
        elif self._urdf is not None:
            update_link_mesh_shapes(self._shapes, self._urdf, q)
        else:
            update_link_collision_shapes(self._shapes, self._arm, q)


def build_arm_shapes(args: argparse.Namespace, arm=None, *, alpha: Optional[float] = None) -> ArmShapes:
    """Build the arm to draw, according to ``--model``.

    Args:
        args: Parsed arguments from :func:`common_parser`; ``--model``,
            ``--radius-scale`` and ``--joint-axes`` are used.
        arm: An already-loaded model, or ``None`` to load one when needed.  The
            collision geometry needs the model; the skeleton does not.
        alpha: Opacity to draw with, overriding ``--mesh-alpha``.  It exists for a
            faded *reference* arm next to the live one -- the skeleton's opacity can
            be set afterwards, but a mesh's is fixed when the shape is built, so a
            single override here is what keeps both models expressible.

    Returns:
        An :class:`ArmShapes`.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        ValueError: If ``--model`` is not a known choice.
    """
    if args.model == "mesh":
        urdf = args.urdf or find_panda_urdf()
        if urdf is None:
            raise FileNotFoundError(
                "no Panda URDF with visual meshes was found: pass --urdf PATH, or set "
                "FRANKA_IK_URDF, or restore third_party/franka_description.  "
                "--model collision and --model skeleton need no URDF."
            )
        shapes = link_mesh_shapes(urdf, arm, alpha=args.mesh_alpha if alpha is None else alpha)
        return ArmShapes(shapes, urdf=urdf, kind="mesh")
    if args.model == "collision":
        if arm is None:
            from .swift_viz import load_arm

            arm, _ = load_arm(urdf=args.urdf)
        shapes = link_collision_shapes(arm)
        return ArmShapes(shapes, arm=arm, kind="collision")
    if args.model != "skeleton":
        raise ValueError(
            f"unknown --model {args.model!r}; available: ['collision', 'mesh', 'skeleton']"
        )
    skeleton = ArmSkeleton(
        radius_scale=args.radius_scale,
        show_joint_axes=args.joint_axes,
        name=args.pose,
    )
    return ArmShapes(skeleton.shapes, skeleton=skeleton, kind="skeleton")

#: The configurations the examples can start from, in degrees.  Each is inside
#: the joint limits and was chosen by measurement for a different reason; the
#: numbers in the comments come from
#: ``python3 scripts/study_wrist_offset_ik.py`` plus a one-line check of
#: ``fk.solve(pose, q7, within_limits_only=True)``, and they are what makes each
#: pose worth looking at:
#:
#: ``ready``
#:     The factory "ready" pose, arm bent forward: well conditioned
#:     (manipulability 9.3e-2) and only one in-limit configuration at this joint
#:     7 -- the ordinary case.
#: ``eight_branches``
#:     All eight branches reach the pose inside the joint limits, four on each
#:     elbow root (manipulability 7.7e-4).  The generating configuration sits on
#:     the second root, which is the half the published code does not evaluate.
#: ``second_root``
#:     Seven in-limit solutions, of which only 3 are on the first elbow root
#:     (manipulability 2.5e-3) -- the case where the published subset loses most.
#: ``near_singular``
#:     Almost a wrist singularity (manipulability 2.7e-4, smallest singular value
#:     6.0e-4), where branches collapse onto each other: six in-limit solutions,
#:     and two of them differ by 65 degrees at joint 2 while reaching the same
#:     pose (``docs/limitations.md`` section 8).
POSES: Dict[str, List[float]] = {
    "ready": [0.0, -17.2, 0.0, -108.9, 0.0, 80.2, 34.4],
    "eight_branches": [41.633, -0.613, -103.197, -24.350, 126.188, 117.608, 68.012],
    "second_root": [-62.620, -2.833, 128.531, -16.213, -46.928, 122.307, -58.783],
    "near_singular": [31.119, -32.418, -35.766, -23.653, -90.038, 133.362, -137.275],
}


def pose_labels() -> List[str]:
    """The names of :data:`POSES`, in the order they are defined."""
    return list(POSES)


def common_parser(description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """An argument parser with the flags every Swift example accepts.

    Args:
        description: The example's own description.
        **kwargs: Passed through to :class:`argparse.ArgumentParser`.

    Returns:
        The parser, with ``--pose``, ``--q7``, ``--headless``, ``--browser``,
        ``--camera``, ``--radius-scale``, ``--joint-axes``, ``--steps``,
        ``--urdf`` and ``--log-level`` already added.
    """
    parser = argparse.ArgumentParser(description=description, **kwargs)
    parser.add_argument(
        "--pose",
        choices=pose_labels(),
        default="ready",
        help="starting configuration (see franka_ik.swift_app.POSES)",
    )
    parser.add_argument(
        "--q7",
        type=float,
        default=None,
        metavar="DEG",
        help="joint 7 to solve at, in degrees; default is the pose's own joint 7",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a browser: the numbers are printed, nothing is drawn",
    )
    parser.add_argument(
        "--browser",
        default=None,
        help="browser to open, or 'auto' to use the one detect_browser finds",
    )
    parser.add_argument(
        "--camera",
        choices=sorted(camera_presets()),
        default="iso",
        help="initial camera preset",
    )
    parser.add_argument(
        "--model",
        choices=("mesh", "collision", "skeleton"),
        default="mesh",
        help="how to draw the arm: 'mesh' is the Panda's own visual meshes from the "
        "description vendored in third_party (the real robot, expanded from xacro on "
        "first use); 'collision' is the bundled description's 30 collision capsules, "
        "needing no URDF at all; 'skeleton' is a stick figure built from the "
        "analytical model, which is the only one that can be recoloured per branch",
    )
    parser.add_argument(
        "--mesh-alpha",
        type=float,
        default=1.0,
        help="opacity of the --model mesh bodies",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="multiplier on the skeleton's derived link radii",
    )
    parser.add_argument(
        "--joint-axes",
        action="store_true",
        help="also draw the frame of every actuated joint",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="animation steps to run before exiting; default runs until interrupted",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="URDF to draw instead of the bundled Panda description",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        help="logging level, e.g. info or debug",
    )
    return parser


def resolve_pose(args: argparse.Namespace) -> Tuple[np.ndarray, float]:
    """The starting configuration and the joint 7 to solve at.

    Args:
        args: Parsed arguments from :func:`common_parser`.

    Returns:
        ``(q, q7)`` in radians.  ``q7`` is ``args.q7`` when it was given and the
        pose's own joint 7 otherwise.
    """
    q = np.radians(np.asarray(POSES[args.pose], dtype=float))
    q7 = float(q[6]) if args.q7 is None else float(np.radians(args.q7))
    return q, q7


def joint7_window(
    pose: np.ndarray, *, fallback: float = 0.0, samples: int = 361
) -> Tuple[float, float]:
    """The joint-7 interval that still reaches ``pose``, in degrees.

    Found by asking the solver rather than assumed: the window depends on the pose
    and is usually far narrower than the joint's own ``[-165, 165]`` degrees, so a
    slider built on the joint limits would spend most of its travel on poses nothing
    reaches.  This is the range the examples' joint-7 sliders span, and the same
    measurement the branch study uses to plot the self-motion.

    Args:
        pose: 4x4 target flange pose.
        fallback: Joint 7 in radians, reported as a degenerate window when nothing
            reaches the pose at all, so a caller still gets a usable slider.
        samples: Joint-7 values to try, evenly spaced across the joint's own range.

    Returns:
        ``(low, high)`` in degrees, with ``low == high`` when the pose is
        unreachable on every sampled joint 7.
    """
    limits = np.radians(model.LOWER_LIMITS_DEG[6]), np.radians(model.UPPER_LIMITS_DEG[6])
    found = [
        float(np.degrees(value))
        for value in np.linspace(limits[0], limits[1], samples)
        if solve(pose, float(value), within_limits_only=True)
    ]
    if not found:
        return float(np.degrees(fallback)), float(np.degrees(fallback))
    return min(found), max(found)


def configuration_reader(args: argparse.Namespace):
    """A callable that reads joint values from the terminal, or ``None``.

    Returns ``None`` when stdin is not a terminal, so a piped or headless run
    never blocks waiting for input it will not get.

    Args:
        args: Parsed arguments; only ``--headless`` is used.

    Returns:
        A function ``(prompt) -> Optional[np.ndarray]`` returning radians, or
        ``None``.
    """
    if args.headless or not sys.stdin.isatty():
        return None

    def read(prompt: str) -> Optional[np.ndarray]:
        try:
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not text:
            return None
        parts = [p for p in text.replace(",", " ").split() if p]
        try:
            values = np.radians([float(p) for p in parts])
        except ValueError:
            print(f"  could not read {text!r}; expected up to seven numbers")
            return None
        q = np.zeros(model.NUM_JOINTS)
        q[: min(len(values), model.NUM_JOINTS)] = values[: model.NUM_JOINTS]
        return q

    return read


def load_arm_for(args: argparse.Namespace):
    """Load the arm an example should draw, and the URDF it came from.

    With ``--model mesh`` that is the Panda URDF the meshes come from -- the
    vendored description by default -- *verified* against
    :func:`franka_ik.model.fk_tool` before anything is drawn.  Every other model
    loads the bundled description, which needs no URDF and no files.

    Args:
        args: Parsed arguments; ``--model`` and ``--urdf`` are used.

    Returns:
        ``(arm, urdf)`` where ``urdf`` is ``None`` when the bundled description was
        used.  ``arm`` is always verified when a URDF was involved, so no caller has
        to check again.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        ArmKinematicsError: If the URDF is a different arm.
        FileNotFoundError: If ``--model mesh`` and no URDF can be found.
    """
    from .swift_viz import find_panda_urdf, load_arm

    if args.model == "mesh":
        # The meshes are placed by ``model.forward_kinematics``, not by the parsed
        # URDF, so nothing here has to load a robot: the URDF contributes file names
        # and eleven fixed joint offsets, and the arm that drives them is this
        # repository's own model.  ``None`` for the arm is the honest answer -- the
        # model is not used, and pretending to verify one would be theatre.
        urdf = args.urdf or find_panda_urdf()
        if urdf is None:
            raise FileNotFoundError(
                "no Panda URDF with visual meshes was found: pass --urdf PATH, or set "
                "FRANKA_IK_URDF, or restore third_party/franka_description.  "
                "--model collision and --model skeleton need no URDF."
            )
        return None, urdf
    return load_arm(urdf=args.urdf)


def selection_index(value: object) -> Optional[int]:
    """Normalise a Swift UI callback argument into an index, or ``None``.

    Swift's ``Radio`` sends an ``int`` index once something is selected but ``[]``
    before that (its own JavaScript starts with ``data = []`` and ``update()`` calls
    the callback unconditionally), so a callback that trusts the annotation crashes
    on the first event.  The same applies to ``Checkbox`` (``list[bool]``),
    ``Slider`` (``float``) and ``Select`` (a ``str`` index).  Everything that arrives
    funnels through here.

    Args:
        value: Whatever the browser sent.

    Returns:
        An index, or ``None`` when nothing usable was sent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    # A sequence: ``[]`` means "nothing selected yet"; a list of booleans means
    # "the first one that is checked", which is what a radio group is.
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if not items:
        return None
    if all(isinstance(item, bool) for item in items):
        for index, item in enumerate(items):
            if item:
                return index
        return None
    return selection_index(items[0])


def checkbox_flags(value: object) -> List[bool]:
    """Normalise a ``Checkbox`` callback argument into a list of booleans."""
    if value is None:
        return []
    if isinstance(value, (bool, int, float)):
        return [bool(value)]
    try:
        return [bool(item) for item in value]  # type: ignore[arg-type]
    except TypeError:
        return []


def _as_html(lines: Sequence[str]) -> str:
    """Readout markup: Swift renders a ``Label``'s text as HTML."""
    return "<br>".join(lines)


def label_for(lines: Sequence[str]) -> object:
    """A readout label whose text can be replaced line by line.

    Swift's ``Label`` renders its string as HTML, so ``<br>`` breaks a line and
    ``<b>`` emphasises; a small helper keeps the examples from repeating the markup.

    Args:
        lines: The lines to show, already escaped by the caller.

    Returns:
        A ``Swift.Elements.Label``.
    """
    deps = require_viz()
    return deps.swift.Elements.Label(_as_html(lines))


def add_readout(
    env: object, lines: Sequence[str], *, name: str = "readout", elements: object = None
) -> object:
    """Add a text readout to the scene and return the label, so it can be updated.

    The readouts in these examples are the measurement, not decoration: each one
    prints the quantity the picture is supposed to demonstrate, which is what makes
    a screenshot checkable.  :func:`set_readout` replaces the text.

    Args:
        env: A launched Swift environment.
        lines: The lines to show; Swift renders them as HTML, so they are joined
            with ``<br>`` and may contain markup.
        name: The element's name in the browser, for debugging.
        elements: ``Swift.Elements``, or ``None`` to resolve it through
            :func:`~franka_ik.swift_viz.require_viz`.

    Returns:
        The ``Swift.Elements.Label``.
    """
    label = label_for(lines) if elements is None else elements.Label(_as_html(lines))
    env.add_ui(label, name=name)
    return label


def set_readout(label: object, lines: Sequence[str]) -> None:
    """Replace a readout's lines, the counterpart of :func:`add_readout`."""
    label.label = _as_html(lines)  # type: ignore[attr-defined]


def add_radio(
    env: object,
    *,
    label: str,
    options: Sequence[str],
    on_select: Callable[[int], None],
    checked: int = 0,
    name: Optional[str] = None,
    elements: object = None,
) -> object:
    """Add a radio group that calls ``on_select(index)`` when a choice is made.

    Swift sends ``[]`` from a radio before anything is selected -- and once more as
    the page attaches -- so the browser's argument is routed through
    :func:`selection_index`, which turns "nothing selected" into ``None``.  A
    callback that trusted its own annotation instead would be called with a list and
    crash on the first event.

    Args:
        env: A launched Swift environment.
        label: The group's label.
        options: The choices, in order.
        on_select: Called with the index of the selected choice.
        checked: Initially selected index.
        name: Optional element name in the browser.
        elements: ``Swift.Elements``, or ``None`` to resolve it through
            :func:`~franka_ik.swift_viz.require_viz`.

    Returns:
        The ``Swift.Elements.Radio``.
    """
    if elements is None:
        elements = require_viz().swift.Elements

    def selected(value: object) -> None:
        index = selection_index(value)
        if index is not None:
            on_select(index)

    radio = elements.Radio(selected, label=label, options=list(options), checked=checked)
    if name is None:
        env.add_ui(radio)
    else:
        env.add_ui(radio, name=name)
    return radio


def add_button(
    env: object, label: str, on_click: Callable[[], None], *, elements: object = None
) -> object:
    """Add a button.  Swift fires its callback once as the page attaches, with ``0``.

    So ``on_click`` has to be idempotent enough to survive being called once before
    anyone pressed anything -- a play/pause toggle simply starts out playing, which
    the examples' loops tolerate because they read the state, not the event.

    Args:
        env: A launched Swift environment.
        label: The button's text.
        on_click: Called with no arguments on each press.
        elements: ``Swift.Elements``, or ``None`` to resolve it lazily.

    Returns:
        The ``Swift.Elements.Button``.
    """
    if elements is None:
        elements = require_viz().swift.Elements

    def pressed(_value: object) -> None:
        on_click()

    button = elements.Button(pressed, label=label)
    env.add_ui(button)
    return button


def add_camera_radio(
    env: object, *, initial: str = "iso", elements: object = None
) -> object:
    """Add the camera-preset radio shared by the examples, with ``initial`` selected.

    Args:
        env: A launched Swift environment.
        initial: Preset selected at start-up; must be one of :data:`CAMERA_CHOICES`.
        elements: ``Swift.Elements``, or ``None`` to resolve it lazily.

    Returns:
        The ``Swift.Elements.Radio``.
    """
    if elements is None:
        elements = require_viz().swift.Elements
    available = camera_presets()
    choices = [name for name in CAMERA_CHOICES if name in available]
    checked = choices.index(initial) if initial in choices else 0

    def switch(index: int) -> None:
        if 0 <= index < len(choices):
            apply_camera(env, choices[index])

    return add_radio(
        env,
        label="camera",
        options=choices,
        on_select=switch,
        checked=checked,
        name="camera",
        elements=elements,
    )


def add_slider(
    env: object,
    *,
    low: float,
    high: float,
    value: float,
    label: str,
    step: float = 0.5,
    unit: str = "",
    precision: int = 1,
    name: Optional[str] = None,
    elements: object = None,
) -> object:
    """Add a slider whose live value :func:`interaction_loop` reads, and return it.

    Two things about Swift's slider have to be worked around, and both were measured
    here rather than guessed.  Swift's JavaScript assigns the new ``value`` *before*
    the new ``min``/``max`` (``swift/public/js/ui.js``, ``Slider.update``), while its
    markup starts the input at ``min=0, max=100``; so a slider whose range does not
    contain 0 has its initial value clamped to the nearer end, and the element is
    then reported once, on attach, as changed -- which is where the old
    "a callback carrying the wrong number" came from.  Measured: a joint-7 slider
    built with ``value=-58.78`` over ``[-77.92, -50.42]`` came back as ``-50.42``,
    its maximum.  Writing the value back after the element is in the scene sticks,
    because by then the real ``min``/``max`` are already in the browser.

    The second thing is that a callback argument is not the live value at all: the
    browser sends an element when it has changed, and the number it carries can be
    the clamped or rounded one.  So the callback here does nothing -- it exists
    because it is what makes the browser send the element -- and the value is read
    with :func:`read_slider`, which in a browser proved reliable in both directions:
    a write from Python reads back as written (``42.0``, then ``42``), and a drag
    arrives as the value dragged to (``-70``).

    Args:
        env: A launched Swift environment.
        low: Minimum value.
        high: Maximum value.
        value: Initial value; :func:`interaction_loop` should be started from the
            same number, and writes it back to the browser before its first frame.
        label: The slider's label.
        step: Smallest increment.  This also sets how fine
            :func:`interaction_loop`'s tolerance has to be.
        unit: Unit shown beside the value.
        precision: Decimal places shown.
        name: Optional element name in the browser.
        elements: ``Swift.Elements``, or ``None`` to resolve it through
            :func:`~franka_ik.swift_viz.require_viz`.

    Returns:
        The ``Swift.Elements.Slider``.
    """
    if elements is None:
        elements = require_viz().swift.Elements
    slider = elements.Slider(
        lambda _value: None,
        min=low,
        max=high,
        step=step,
        value=value,
        label=label,
        unit=unit,
        precision=precision,
    )
    if name is None:
        env.add_ui(slider)
    else:
        env.add_ui(slider, name=name)
    # The write-back that undoes the browser's clamping of the initial value.
    slider.value = float(value)
    return slider


def read_slider(slider: object, fallback: Optional[float]) -> float:
    """The slider's live value, or ``fallback`` before the browser has sent one.

    Reading it back is safe as well as simple: measured in a browser, a value written
    from Python reads back as written (``42.0`` immediately, ``42`` after further
    frames) and a browser drag arrives as the value dragged to (``-70``).  So the
    slider is one source of truth for both directions, which is what lets
    :func:`interaction_loop` compare against it every frame.
    """
    try:
        return float(slider.value)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return 0.0 if fallback is None else float(fallback)


def interaction_loop(
    env: object,
    *,
    slider: object = None,
    initial: Optional[float] = None,
    on_change: Optional[Callable[[float], None]] = None,
    on_frame: Optional[Callable[[float], Optional[float]]] = None,
    steps: Optional[int] = None,
    dt: float = 0.05,
    tolerance: float = 1e-9,
) -> None:
    """Render the scene while following a slider's live value, until the window closes.

    ``env.step(dt)`` is the only thing that renders -- see :func:`hold` -- so this is
    ``hold`` plus the one thing a slider needs: after each frame, the value the
    slider now holds is compared with the value the scene was last built for, and
    ``on_change`` re-solves and re-draws when they differ.  The comparison is against
    the *live* value, never against a callback argument (see :func:`add_slider`).

    Args:
        env: A launched Swift environment.
        slider: An ``Elements.Slider``, or ``None`` for a scene with no slider.
        initial: The value the scene was built for.  It is written back to the slider
            before the first frame -- see :func:`add_slider` for the clamp that makes
            that necessary -- and is the starting point when there is no slider at all.
        on_change: Called with the new value whenever it moves by more than
            ``tolerance``; this is what redraws the scene.
        on_frame: Called with the current value on every frame.  Returning a value
            writes it back to the slider -- which is how a play button sweeps -- and
            counts as a change, so ``on_change`` runs for it as well.
        steps: Stop after this many frames; ``None`` runs until interrupted.
        dt: Frame interval, in seconds.
        tolerance: How far the value has to move to count as a change.  It must be
            below the slider's own ``step``, and below the finest feature the slider
            is there to find: the joint-2 offset slider of
            ``examples/10_swift_singularities.py`` has ``step=1e-5`` and exists to
            bracket a failure window ``1e-4`` degrees wide.
    """
    current = None if initial is None else float(initial)
    count = 0
    if slider is not None and current is not None:
        # The scene was built for ``initial``, so make that the browser's value too.
        slider.value = current  # type: ignore[attr-defined]
    try:
        while steps is None or count < steps:
            value = read_slider(slider, current) if slider is not None else current
            if value is not None and (current is None or abs(value - current) > tolerance):
                current = value
                if on_change is not None:
                    on_change(current)
            if on_frame is not None and current is not None:
                written = on_frame(current)
                if written is not None and abs(float(written) - current) > tolerance:
                    current = float(written)
                    if slider is not None:
                        slider.value = current  # type: ignore[attr-defined]
                    if on_change is not None:
                        on_change(current)
            env.step(dt)
            count += 1
    except KeyboardInterrupt:
        pass


def hold(env: object, *, steps: Optional[int] = None, dt: float = 0.05) -> None:
    """Keep a scene on screen, rendering it, until the user closes the window.

    ``env.step(dt)`` is what pushes the scene to the browser: it advances the
    simulation clock, re-sends any shape whose pose changed, and *renders*.  A loop
    that only calls ``time.sleep`` therefore shows a frozen first frame and looks
    like a viewer that does not work -- which is exactly what it was.  ``step``
    blocks, so it also replaces the sleep.

    ``env.run`` is the same loop with disconnect handling, but it cannot be stopped
    after N steps, which the examples' ``--steps`` needs in order to be testable.
    A disconnected browser ends the loop here too, by raising, which the caller's
    ``finally`` turns into a clean close.

    Args:
        env: A launched Swift environment.
        steps: Stop after this many steps; ``None`` runs until interrupted.
        dt: Time step, in seconds; also the frame interval.
    """
    count = 0
    try:
        while steps is None or count < steps:
            env.step(dt)
            count += 1
    except KeyboardInterrupt:
        pass


def print_model_summary(
    arm: object,
    urdf: Optional[Path],
    *,
    worst: Optional[float] = None,
    model: str = "skeleton",
) -> None:
    """Print which model is being drawn, and how well it matches this one.

    Args:
        arm: The loaded model.
        urdf: The URDF it came from, or ``None`` for the bundled description.
        worst: Largest measured disagreement with ``fk_tool``, if it was checked.
        model: The ``--model`` choice, so the note explains what is on screen.
    """
    source = str(urdf) if urdf is not None else "bundled Panda description (rtb-data)"
    if arm is None:
        print("model   : franka_ik.model  (the analytical model places the meshes)")
    else:
        print(f"model   : {getattr(arm, 'name', '?')}  ({getattr(arm, 'n', '?')} joints)")
    print(f"source  : {source}")
    if worst is not None:
        print(f"fk check: agrees with franka_ik.model.fk_tool to {worst:.1e} (4x4, 5 poses)")
    if urdf is None:
        # Two different things can be on screen and they are worth telling apart:
        # "no URDF" does not mean "nothing to see", it means no *meshes*.
        drawn = (
            "the model's own collision capsules (30 primitives)"
            if model == "collision"
            else "the capsule skeleton built from the analytical model"
        )
        print(f"note    : no URDF found, so no meshes; drawing {drawn} instead.")


def scene(args: argparse.Namespace, arm=None) -> Tuple[object, ArmShapes]:
    """Launch a Swift environment with one arm in it.

    Args:
        args: Parsed arguments from :func:`common_parser`.
        arm: Optional already-loaded model, passed to :func:`build_arm_shapes`.

    Returns:
        ``(env, shapes)``, both ready to use; the caller owns ``env.close()``.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        RuntimeError: If Swift cannot start.
    """
    env = launch_env(headless=args.headless, browser=args.browser)
    arm_shapes = build_arm_shapes(args, arm=arm)
    add_shapes(env, arm_shapes.shapes)
    apply_camera(env, args.camera)
    return env, arm_shapes


def configure_logging(args: argparse.Namespace) -> None:
    """Apply ``--log-level`` to the root logger."""
    logging.basicConfig(level=str(args.log_level).upper(), format="%(levelname)s %(name)s: %(message)s")


def degrees(values: Sequence[float], precision: int = 2) -> str:
    """Format a joint vector as degrees, for a readout line."""
    return "[" + ", ".join(f"{np.degrees(v):.{precision}f}" for v in values) + "]"
