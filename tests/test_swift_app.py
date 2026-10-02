"""The Swift examples' shared plumbing: the loop, the slider, the UI elements.

Everything here runs **without** Swift, and without the ``viz`` extra: the subject is
:mod:`franka_ik.swift_app`'s argument handling and its interaction loop, none of
which needs a renderer.  The fakes below are the smallest objects the loop touches --
an environment that counts frames, a ``Swift.Elements`` that records what it was
built with -- so the tests pin behaviour the browser would otherwise be the only
witness to:

* the loop reads the slider's **live** value and never the callback argument, and it
  writes the scene's own value back before its first frame -- because Swift's slider
  JavaScript assigns ``value`` before ``min``/``max``, so a range that does not
  contain 0 has its initial value clamped to the nearer end.  Measured: a joint-7
  slider built with ``-58.78`` over ``[-77.92, -50.42]`` came back as ``-50.42``,
  its maximum, which is where the old "the callback carried the wrong number" came
  from;
* a value written from Python *is* visible to the next read, which is what lets a
  play button sweep and a preset radio jump the slider.  Measured in a browser:
  ``42.0`` read back as ``42``, and a drag to ``-70`` arrived as ``-70``;
* the tolerance has to be below the slider's own step, or a feature as narrow as the
  ``1e-4`` degree failure window of ``examples/10`` is invisible to the loop;
* a radio's first event is Swift's own empty selection, not a choice.

The geometry half of the Swift layer is in ``tests/test_swift_viz.py``, which does
need the extra and skips without it.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pytest

import franka_ik as fk
from franka_ik import swift_app, swift_viz


class _Element:
    """A stand-in for a Swift UI element: it records what it was built with.

    The first positional argument is the callback for every element except a
    ``Label``, whose first argument is its text -- which is exactly the distinction
    the real constructors make.
    """

    def __init__(self, *args: Any, kind: str = "?", **kwargs: Any) -> None:
        self.kind = kind
        self.kwargs = dict(kwargs)
        if kind == "Label":
            self.label = args[0] if args else ""
            self.callback = None
        else:
            self.callback = args[0] if args else None
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Swift's ``Slider`` keeps its value as a plain attribute, which the browser
        # overwrites with each change; ``read_slider`` reads exactly this.
        self.value = kwargs.get("value")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.kind} {self.kwargs}>"


class _Elements:
    """The constructors the examples use, as records."""

    _NAMES = ("Label", "Slider", "Button", "Radio", "Checkbox", "Select")

    def __init__(self) -> None:
        self.made: List[_Element] = []

    def __getattr__(self, name: str) -> Any:
        if name not in self._NAMES:
            raise AttributeError(name)

        def build(*args: Any, **kwargs: Any) -> _Element:
            element = _Element(*args, kind=name, **kwargs)
            self.made.append(element)
            return element

        return build


class _Env:
    """A stand-in for a launched Swift environment: it counts frames."""

    def __init__(self) -> None:
        self.ui: List[Any] = []
        self.frames = 0
        self.dt = 0.0
        self.camera: Optional[Any] = None

    def add_ui(self, element: Any, name: Optional[str] = None) -> None:
        self.ui.append((name, element))

    def set_camera_pose(self, position: Any, look_at: Any) -> None:
        self.camera = (list(position), list(look_at))

    def step(self, dt: float) -> None:
        self.frames += 1
        self.dt = dt


def test_add_slider_seeds_the_value_and_ignores_its_callback() -> None:
    """The callback does nothing, and the value the browser clamped is overwritten.

    Two measured traps in one place.  The callback is not the live value, so it is
    a no-op.  And Swift's slider JavaScript assigns ``value`` before ``min``/``max``
    while the markup starts at ``0..100``, so a range that does not contain 0 clamps
    the initial value to the nearer end -- a slider built with ``-58.78`` over
    ``[-77.92, -50.42]`` came back as ``-50.42``.  ``add_slider`` writes the value
    back once the element is in the scene, which is what makes the fake's ``.value``
    -- and the browser's -- the number the caller asked for.
    """
    env, elements = _Env(), _Elements()
    slider = swift_app.add_slider(
        env,
        low=-1.0,
        high=1.0,
        value=0.25,
        label="d",
        step=1e-5,
        unit="deg",
        precision=5,
        name="d",
        elements=elements,
    )
    assert slider.kwargs == {
        "min": -1.0,
        "max": 1.0,
        "step": 1e-5,
        "value": 0.25,
        "label": "d",
        "unit": "deg",
        "precision": 5,
    }
    assert env.ui[-1] == ("d", slider)
    assert swift_app.read_slider(slider, 0.0) == 0.25
    slider.callback(99.0)
    assert swift_app.read_slider(slider, 0.0) == 0.25, "the argument must not move it"
    slider.value = 0.5
    assert swift_app.read_slider(slider, 0.0) == 0.5


def test_read_slider_falls_back_when_nothing_has_been_sent() -> None:
    """A slider that is absent or empty yields the fallback, never an exception."""

    class _Blank:
        pass

    class _Empty:
        value = None

    assert swift_app.read_slider(_Blank(), 3.5) == 3.5
    assert swift_app.read_slider(_Blank(), None) == 0.0
    assert swift_app.read_slider(_Empty(), 1.0) == 1.0


def test_interaction_loop_follows_the_slider_it_is_given() -> None:
    """A move of the live value is what triggers a redraw, and only a move is."""
    env, elements = _Env(), _Elements()
    slider = swift_app.add_slider(env, low=0.0, high=10.0, value=1.0, label="x", elements=elements)

    def drive(_current: float) -> None:
        """Play the part of a browser: move the slider on two frames."""
        if env.frames == 0:
            slider.value = 4.0
        elif env.frames == 2:
            slider.value = 7.0

    changed: List[float] = []
    swift_app.interaction_loop(
        env,
        slider=slider,
        initial=1.0,
        on_change=changed.append,
        on_frame=drive,
        steps=6,
    )
    assert changed == [4.0, 7.0]
    assert env.frames == 6, "steps bounds the frames rendered"


def test_interaction_loop_undoes_a_browser_clamped_initial_value() -> None:
    """Frame 0 starts from the scene's value, not from whatever the browser clamped.

    One slider whose range is entirely negative is the measured case: the element is
    built with ``value=-58.78`` over ``[-77.92, -50.42]`` and the browser reports
    ``-50.42``, its maximum, because the JavaScript assigns ``value`` while the markup
    still says ``0..100``.  A loop that trusted its first read would jump to the end of
    the joint-7 window before the reader touched anything.
    """
    env, elements = _Env(), _Elements()
    slider = swift_app.add_slider(
        env, low=-77.92, high=-50.42, value=-58.78, label="joint 7", elements=elements
    )
    slider.value = -50.42  # what the browser's clamping leaves behind
    changed: List[float] = []
    swift_app.interaction_loop(
        env, slider=slider, initial=-58.78, on_change=changed.append, steps=3
    )
    assert changed == []
    assert slider.value == -58.78


def test_interaction_loop_writes_back_what_on_frame_returns() -> None:
    """A play hook advances by returning the next value, which is written back.

    The write-back is what makes the browser's own widget follow the sweep, and the
    fake mirrors the measured behaviour of the real one: a value written from Python
    reads back as written, so the next frame does not see a stale number and undo it.
    """
    env, elements = _Env(), _Elements()
    slider = swift_app.add_slider(env, low=0.0, high=100.0, value=0.0, label="x", elements=elements)

    def advance(current: float) -> float:
        return current + 1.0

    changed: List[float] = []
    swift_app.interaction_loop(
        env, slider=slider, initial=0.0, on_change=changed.append, on_frame=advance, steps=4
    )
    assert changed == [1.0, 2.0, 3.0, 4.0]
    assert slider.value == 4.0


def test_interaction_loop_reports_nothing_when_the_hook_declines() -> None:
    """``None`` from the hook means "do not move": a paused sweep leaves the scene."""

    def paused(_current: float) -> None:
        return None

    env, elements = _Env(), _Elements()
    slider = swift_app.add_slider(env, low=0.0, high=10.0, value=3.0, label="x", elements=elements)
    changed: List[float] = []
    swift_app.interaction_loop(
        env, slider=slider, initial=3.0, on_change=changed.append, on_frame=paused, steps=5
    )
    assert changed == []
    assert slider.value == 3.0


def test_interaction_loop_tolerance_has_to_beat_the_slider_step() -> None:
    """A tolerance above the step hides features as narrow as the failure window.

    ``examples/10_swift_singularities.py`` uses ``step=1e-5`` to bracket a failure
    window ``1e-4`` degrees wide.  With the loop's ``1e-9`` default the move is seen;
    a tolerance of ``1e-3`` -- comfortably above the window -- is blind to it.
    """
    for tolerance, expected in ((1e-9, 1), (1e-3, 0)):
        env, elements = _Env(), _Elements()
        slider = swift_app.add_slider(
            env, low=-0.01, high=0.01, value=0.0, label="d", step=1e-5, elements=elements
        )

        def drive(_current: float, env: _Env = env, slider: _Element = slider) -> None:
            if env.frames == 0:
                slider.value = 1e-4

        changed: List[float] = []
        swift_app.interaction_loop(
            env,
            slider=slider,
            initial=0.0,
            on_change=changed.append,
            on_frame=drive,
            steps=3,
            tolerance=tolerance,
        )
        assert len(changed) == expected, f"tolerance {tolerance:g}"


def test_add_radio_ignores_the_empty_selection_swift_sends_first() -> None:
    """A radio reports an index, and Swift's own empty event is not a choice.

    Measured: the browser calls a radio's callback with ``[]`` before anything is
    selected, and once more as the page attaches.  A callback that trusted its
    annotation would index a list with a list and crash on the first event.
    """
    env, elements = _Env(), _Elements()
    picked: List[int] = []
    radio = swift_app.add_radio(
        env,
        label="show",
        options=["roots", "all"],
        on_select=picked.append,
        checked=1,
        elements=elements,
    )
    assert radio.kwargs["options"] == ["roots", "all"]
    assert radio.kwargs["checked"] == 1
    radio.callback([])
    radio.callback(None)
    radio.callback(1)
    radio.callback([False, True])
    radio.callback("0")
    radio.callback(2.0)
    assert picked == [1, 1, 0, 2]


def test_add_camera_radio_switches_between_the_presets() -> None:
    """The camera radio covers every preset, starts where told, and ignores rubbish."""
    env, elements = _Env(), _Elements()
    swift_app.add_camera_radio(env, initial="side", elements=elements)
    radio = elements.made[-1]
    assert radio.kwargs["options"] == list(swift_app.CAMERA_CHOICES)
    assert radio.kwargs["checked"] == swift_app.CAMERA_CHOICES.index("side")
    assert env.ui[-1][0] == "camera"

    presets = swift_viz.camera_presets()
    radio.callback(3)
    assert env.camera == (presets[swift_app.CAMERA_CHOICES[3]][0], presets[swift_app.CAMERA_CHOICES[3]][1])
    radio.callback(99)
    assert env.camera == (presets[swift_app.CAMERA_CHOICES[3]][0], presets[swift_app.CAMERA_CHOICES[3]][1])


def test_add_button_tolerates_the_attach_callback() -> None:
    """Swift fires a button's callback once as the page attaches, with ``0``."""
    env, elements = _Env(), _Elements()
    calls: List[int] = []
    button = swift_app.add_button(env, "play", lambda: calls.append(1), elements=elements)
    button.callback(0)
    button.callback(0)
    assert calls == [1, 1], "the hook runs; what it must not do is crash"
    assert button.label == "play"
    assert env.ui[-1] == (None, button)


def test_add_readout_and_set_readout_share_the_html_markup() -> None:
    """Readout lines are joined with ``<br>``, because Swift renders them as HTML."""
    env, elements = _Env(), _Elements()
    label = swift_app.add_readout(env, ["a", "b"], elements=elements)
    assert label.kind == "Label"
    assert label.label == "a<br>b"
    assert env.ui[-1] == ("readout", label)
    swift_app.set_readout(label, ["c"])
    assert label.label == "c"


def test_selection_index_normalises_every_callback_swift_sends() -> None:
    """The one place the browser's argument shapes are collected."""
    assert swift_app.selection_index(None) is None
    assert swift_app.selection_index([]) is None
    assert swift_app.selection_index([False, False]) is None
    assert swift_app.selection_index(True) == 1
    assert swift_app.selection_index(3) == 3
    assert swift_app.selection_index(2.7) == 2
    assert swift_app.selection_index("1") == 1
    assert swift_app.selection_index("nonsense") is None
    assert swift_app.selection_index([False, True]) == 1
    assert swift_app.selection_index([2, 0]) == 2
    assert swift_app.selection_index(object()) is None


def test_checkbox_flags_normalises_every_callback_swift_sends() -> None:
    """A checkbox sends a list of booleans, and sometimes just a scalar."""
    assert swift_app.checkbox_flags(None) == []
    assert swift_app.checkbox_flags(True) == [True]
    assert swift_app.checkbox_flags(0) == [False]
    assert swift_app.checkbox_flags([True, False]) == [True, False]
    assert swift_app.checkbox_flags(object()) == []


def test_joint7_window_is_the_interval_the_solver_reaches_the_pose_in() -> None:
    """The window is measured, and a sample just outside it has no solution.

    The window is what the examples' joint-7 sliders span.  Built on the joint's own
    ``[-165, 165]`` degrees instead, most of the travel would be poses nothing
    reaches, so the interval is found by asking the solver -- and the check that it
    is the *interval* is that one sample step beyond either end fails.
    """
    step = 330.0 / 360.0  # the scan's own spacing, from np.linspace(-165, 165, 361)
    for name in ("ready", "eight_branches", "second_root", "near_singular"):
        q = np.radians(np.asarray(swift_app.POSES[name], dtype=float))
        pose = fk.fk_flange(q)
        low, high = swift_app.joint7_window(pose)
        assert low <= np.degrees(q[6]) <= high, name
        assert fk.solve(pose, float(np.radians(low)), within_limits_only=True), name
        assert fk.solve(pose, float(np.radians(high)), within_limits_only=True), name
        assert not fk.solve(pose, float(np.radians(low - step)), within_limits_only=True), name
        assert not fk.solve(pose, float(np.radians(high + step)), within_limits_only=True), name


def test_joint7_window_falls_back_to_a_degenerate_interval() -> None:
    """An unreachable pose still yields a usable slider, not an error."""
    pose = np.eye(4)
    pose[:3, 3] = [2.0, 0.0, 0.0]  # outside the reachable shell
    assert swift_app.joint7_window(pose, fallback=float(np.radians(7.0))) == (7.0, 7.0)
    assert swift_app.joint7_window(pose) == (0.0, 0.0)


def test_the_examples_say_what_is_missing_when_there_is_no_urdf(monkeypatch) -> None:
    """``--model mesh`` is the default, and a reader who deleted ``third_party/`` meets this.

    The check is the message: it has to name the flag and the two models that work
    without a URDF, rather than raise a bare ``FileNotFoundError``.  ``find_panda_urdf``
    is patched where :func:`franka_ik.swift_app.build_arm_shapes` looks it up, not where
    it is defined.
    """

    class _Args:
        model = "mesh"
        radius_scale = 1.0
        joint_axes = False
        pose = "ready"
        urdf = None
        mesh_alpha = 1.0

    monkeypatch.setattr(swift_app, "find_panda_urdf", lambda: None)
    with pytest.raises(FileNotFoundError) as raised:
        swift_app.build_arm_shapes(_Args())
    text = str(raised.value)
    assert "--urdf PATH" in text
    assert "FRANKA_IK_URDF" in text
    assert "third_party/franka_description" in text
    assert "--model collision" in text and "--model skeleton" in text


def test_camera_choices_match_the_presets() -> None:
    """Every name in :data:`CAMERA_CHOICES` is a preset, and none is missing."""
    assert list(swift_app.CAMERA_CHOICES) == ["iso", "front", "side", "top"]
    assert set(swift_app.CAMERA_CHOICES) == set(swift_viz.camera_presets())
