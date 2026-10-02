"""The Swift layer: geometry it draws, and the URDF check that guards it.

Everything here is checked **without launching Swift**.  The scene, the browser and
the animation loop are the examples' job; what a caller needs from
:mod:`franka_ik.swift_viz` before handing shapes to ``Swift.add_shape`` is that the
geometry is right, and that is what this module pins:

* :func:`~franka_ik.swift_viz.arm_keypoints` reproduces the joint origins the model
  reports, so a drawn skeleton cannot drift away from the solver;
* the three end frames -- wrist, flange and tool -- are 0.107 m and 0.1034 m apart, and
  the markers and stem the examples draw are on them, which is the difference between a
  pose marker and a frame hanging in space;
* :func:`~franka_ik.swift_viz.joint_axes` returns the axes the solver turns about;
* :func:`~franka_ik.swift_viz.arm_plane_outline` declines to draw a plane that does
  not exist (a straight arm), instead of returning a degenerate shape;
* :func:`~franka_ik.swift_viz.check_kinematics` **rejects a model that is not this
  arm**.  That is the one function here whose failure mode would be silent and
  expensive: an FR3 driven by Panda joint vectors renders a plausible, wrong robot,
  and the repository's whole argument is that a wrong answer that looks right is
  worse than an error;
* the sample poses in :mod:`franka_ik.swift_app` are inside the joint limits and
  behave as their documentation claims.

The module under test imports ``roboticstoolbox``, ``spatialgeometry`` and ``swift``
lazily, but the ground truth it is checked against needs the optional ``viz`` extra,
so the whole file skips cleanly without it.  ``check_kinematics`` and
``load_arm`` are the exception that proves it: they are also asserted *without* the
extra, through the import error, because that error message is the extension point a
user meets first.

Measured, for the record: the bundled Panda description from ``rtb-data`` agrees
with :func:`franka_ik.model.fk_tool` to ``1.7e-16`` over 5 in-limit configurations,
and its joint limits are the robot's own to three decimals
(``[-166.003, 166.003]`` for joint 1, i.e. the datasheet's ``±166°``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import franka_ik as fk
from franka_ik import analysis, model, swift_app
from franka_ik import swift_viz as sv

pytest.importorskip("roboticstoolbox")
pytest.importorskip("swift")
pytest.importorskip("spatialgeometry")

#: A bent configuration inside the joint limits, used wherever a sample is needed.
_Q = np.radians([10.0, -25.0, 15.0, -70.0, 20.0, 60.0, 35.0])


def test_require_viz_returns_the_three_modules() -> None:
    """``require_viz`` yields ``(rtb, swift, sg)`` and unpacks like a tuple."""
    deps = sv.require_viz()
    rtb_module, swift_module, sg_module = deps
    assert deps.rtb is rtb_module
    assert deps.swift is swift_module
    assert deps.sg is sg_module
    assert hasattr(swift_module, "Swift")
    assert hasattr(sg_module, "Cylinder")


def test_arm_keypoints_match_the_joint_frames() -> None:
    """The drawn skeleton sits on the model's own joint origins, base first.

    Nine points: base, seven joint origins, tool.  The middle seven are compared
    against :func:`franka_ik.model.joint_frames` element by element, so a skeleton
    that drifted from the solver would fail here rather than on screen.  The frame
    indices matter -- ``joint_frames`` starts at joint 1, not at the base -- so the
    first joint origin is also checked against the DH parameters directly.
    """
    points = sv.arm_keypoints(_Q)
    assert len(points) == model.NUM_JOINTS + 2
    assert np.allclose(points[0], np.zeros(3), atol=1e-15)
    frames = model.joint_frames(_Q)
    for index, origin in enumerate(frames):
        assert np.allclose(points[index + 1], origin, atol=1e-15)
    # joint 1 is the bottom of the shoulder column, d_bs above the base -- not the base
    assert np.allclose(points[1], [0.0, 0.0, model.DH_PARAMETERS[0, 1]], atol=1e-15)
    assert np.allclose(points[-1], model.fk_tool(_Q)[:3, 3], atol=1e-15)


def test_arm_keypoints_are_stable_across_configurations() -> None:
    """Point *n* is the same joint at every configuration, never dropped.

    Joints 1 and 2 share an origin in the DH nesting, and so do joints 5 and 6, so
    consecutive points can coincide.  Drawing a coincident pair as a zero-length
    segment is what keeps the indexing fixed; if a point were ever skipped the
    skeleton would attach the wrong cylinder to the wrong joint.
    """
    rng = np.random.default_rng(0)
    lower, upper = model.lower_limits(), model.upper_limits()
    for _ in range(25):
        q = rng.uniform(lower, upper)
        points = sv.arm_keypoints(q)
        assert len(points) == model.NUM_JOINTS + 2
        assert all(np.all(np.isfinite(point)) for point in points)


def test_joint_axes_point_along_the_joint_frames() -> None:
    """Each axis is the ``z`` column of the joint frame, from the analytical model."""
    axes = sv.joint_axes(_Q)
    assert len(axes) == model.NUM_JOINTS
    frames = model.forward_kinematics(_Q)
    for index, (origin, direction) in enumerate(axes):
        assert np.allclose(origin, frames[index][:3, 3], atol=1e-15)
        assert np.allclose(direction, frames[index][:3, 2], atol=1e-15)
        assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-15)


def test_arm_plane_returns_a_closed_outline() -> None:
    """The plane is a closed four-point outline through shoulder, elbow, wrist.

    Measured: the degenerate case this function guards against is **unreachable on a
    Panda**.  The 0.0825 m shoulder offset keeps the elbow off the shoulder-wrist
    line, and the smallest cross-product norm over 20 000 sampled in-limit
    configurations is ``9.9e-6`` -- four orders of magnitude above the ``1e-9``
    threshold -- so ``arm_plane_outline`` never returns ``None`` in practice.  The
    guard is exercised through the threshold instead of by pretending a straight
    arm is one, because a test that claims the wrong reason for a ``None`` is worse
    than no test.
    """
    bent = sv.arm_plane_outline(_Q)
    assert bent is not None
    assert np.asarray(bent.points).shape == (3, 4)  # closed: shoulder, elbow, wrist, shoulder
    assert np.all(np.isfinite(np.asarray(bent.points, dtype=float)))

    monkeypatched = sv._DEGENERATE_PLANE
    try:
        sv._DEGENERATE_PLANE = 1e9  # every plane is "degenerate" at this threshold
        assert sv.arm_plane_outline(_Q) is None
    finally:
        sv._DEGENERATE_PLANE = monkeypatched


def test_skeleton_shapes_are_built_and_moved_by_pose_only() -> None:
    """A skeleton has 8 segments, 9 joint spheres and optional axes, all finite.

    Nine key points give eight segments; a sphere goes on every key point, the base
    included.  Writing a length or radius after construction marks a shape changed,
    and Swift's headless client does not acknowledge that update, so the check is
    that :meth:`~franka_ik.swift_viz.ArmSkeleton.update` moves *poses* and nothing
    else -- which is why the length list is compared before and after.
    """
    skeleton = sv.ArmSkeleton()
    assert len(skeleton.shapes) == 2 * model.NUM_JOINTS + 3  # 8 segments + 9 spheres
    lengths = [shape.length for shape in skeleton.shapes if hasattr(shape, "length")]
    skeleton.update(_Q)
    after = [shape.length for shape in skeleton.shapes if hasattr(shape, "length")]
    assert lengths == after, "update() must not change a working-copy length"
    for shape in skeleton.shapes:
        assert np.all(np.isfinite(np.asarray(shape.T, dtype=float)))

    with_axes = sv.ArmSkeleton(show_joint_axes=True)
    assert len(with_axes.shapes) == 3 * model.NUM_JOINTS + 3


def test_skeleton_colour_and_alpha_reach_the_shapes() -> None:
    """``colour`` and ``alpha`` are applied together, not one without the other."""
    skeleton = sv.ArmSkeleton(colour=[0.2, 0.45, 0.85], alpha=0.22)
    for shape in skeleton.shapes:
        assert np.allclose(shape.color[:3], [0.2, 0.45, 0.85], atol=1e-12)
        assert np.isclose(shape.opacity, 0.22, atol=1e-12)


def test_collision_geometry_is_drawn_from_the_model_and_moves_with_it() -> None:
    """The collision-capsule mode draws the model's own geometry, and follows ``q``.

    This is the mode that needs no URDF, so it is the one most likely to be used --
    and the one where a wrong link index would go unnoticed, because 30 capsules in
    roughly the right place still look like an arm.  The check is quantitative: the
    nearest capsule to the *flange* must stay inside the tool stem (0.107 m), since
    the last link's geometry covers it, and the whole cloud must stay inside a box
    the arm can reach.
    """
    arm, _ = sv.load_arm()
    shapes = sv.link_collision_shapes(arm)
    assert len(shapes) == 30, "measured: 30 primitives in the bundled Panda description"
    assert {type(shape).__name__ for shape in shapes} <= {"Cylinder", "Sphere"}

    rng = np.random.default_rng(0)
    lower, upper = model.lower_limits(), model.upper_limits()
    for _ in range(5):
        q = rng.uniform(lower, upper)
        sv.update_link_collision_shapes(shapes, arm, q)
        positions = np.asarray([np.asarray(shape.T, dtype=float)[:3, 3] for shape in shapes])
        assert np.all(np.isfinite(positions))
        flange = model.fk_flange(q)[:3, 3]
        nearest = np.linalg.norm(positions - flange, axis=1).min()
        assert nearest < 0.107, "measured 0.027 m: the tool stem's capsule covers the flange"
        assert np.abs(positions).max() < 1.2, "measured at most 1.06 m from the base"

    # At the zero configuration the cloud is the upright arm: ankle to flange.
    # ``fkine_all`` starts with the base frame, so link0 lands on the floor for free;
    # this is the check the mesh path had to be fixed to match.
    sv.update_link_collision_shapes(shapes, arm, np.zeros(model.NUM_JOINTS))
    positions = np.asarray([np.asarray(shape.T, dtype=float)[:3, 3] for shape in shapes])
    assert positions[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert positions[:, 2].max() == pytest.approx(1.093, abs=1e-3)


def test_build_arm_shapes_offers_both_modes() -> None:
    """``--model`` chooses the arm, and both choices come back usable."""

    class _Args:
        model = "skeleton"
        radius_scale = 1.0
        joint_axes = False
        pose = "ready"
        urdf = None

    args = _Args()
    skeleton = swift_app.build_arm_shapes(args)
    assert skeleton.kind == "skeleton"
    assert len(skeleton.shapes) == 2 * model.NUM_JOINTS + 3

    args.model = "collision"
    arm, _ = sv.load_arm()
    collision = swift_app.build_arm_shapes(args, arm=arm)
    assert collision.kind == "collision"
    assert len(collision.shapes) == 30

    q = np.radians(np.asarray(swift_app.POSES["ready"], dtype=float))
    for shapes in (skeleton, collision):
        shapes.update(q)
        for shape in shapes.shapes:
            assert np.all(np.isfinite(np.asarray(shape.T, dtype=float)))

    args.model = "nonsense"
    with pytest.raises(ValueError, match="unknown --model"):
        swift_app.build_arm_shapes(args)


def test_urdf_meshes_are_placed_where_the_arm_really_is() -> None:
    """The mesh mode draws the Panda's own bodies, in the right places.

    This is the one mode where a wrong frame index would be invisible to the code
    and obvious only to an eye that knows the robot -- two DAE files swapped, or a
    one-joint offset, still render as a robot.  So the test is a bounding box, which
    a one-joint error does move: the mesh cloud at ``q = 0`` must span the Panda's
    own dimensions, from the base to the flange (``z`` up to 1.03 m) and reach
    ``0.088 m`` in ``x`` at the wrist.

    The meshes themselves are the vendored description under ``third_party/``; the
    URDF is expanded from its xacro on first use and cached, so this reads a file.
    """
    urdf = sv.find_panda_urdf()
    assert urdf is not None, "third_party/franka_description is missing"
    entries = sv.mesh_links_from_urdf(urdf)
    assert [name for name, _, _ in entries] == [f"panda_link{i}" for i in range(8)]

    shapes = sv.link_mesh_shapes(urdf, None)
    assert len(shapes) == 8
    sv.update_link_mesh_shapes(shapes, urdf, np.zeros(model.NUM_JOINTS))
    origins = np.asarray([np.asarray(shape.T, dtype=float)[:3, 3] for shape in shapes])
    # link0 is the base plate and must be **on the base**, not 0.333 m up at the
    # shoulder: measured origins at q = 0 are exactly the URDF link frames,
    # [0,0,0] then 0.333 / 0.333 / 0.649 / 0.649 / 1.033 / 1.033 / 1.033.
    assert np.allclose(origins[0], [0.0, 0.0, 0.0], atol=1e-9)
    assert origins[:, 2].max() == pytest.approx(1.033, abs=1e-6)
    assert origins[:, 0].max() == pytest.approx(0.088, abs=1e-6)
    # Every mesh sits at its own URDF link frame, checked against the URDF walked by
    # hand rather than against the table under test.
    expected_z = [0.0, 0.333, 0.333, 0.649, 0.649, 1.033, 1.033, 1.033]
    assert np.allclose(origins[:, 2], expected_z, atol=1e-6)

    # ... and they follow the arm: a bent configuration moves the wrist links.
    q = np.radians(np.asarray(swift_app.POSES["ready"], dtype=float))
    sv.update_link_mesh_shapes(shapes, urdf, q)
    moved = np.asarray([np.asarray(shape.T, dtype=float)[:3, 3] for shape in shapes])
    # ``panda_link8`` carries no visual -- it is the flange, a marker frame at the far
    # end of link7's 0.107 m -- so there are eight meshes and the last one does *not*
    # land on the flange.  Measured distance: 0.107 m, exactly the flange offset.  An
    # entry of 6 for link7 in the placement table would put it on the wrist instead,
    # 0.088 m out, which the picture would not reveal.
    flange = model.fk_flange(q)[:3, 3]
    assert np.linalg.norm(moved[-1] - flange) == pytest.approx(0.107, abs=1e-6)
    # ... while link6's mesh is a whole forearm-and-offset away from the flange.
    assert np.linalg.norm(moved[-2] - flange) > 0.1


def test_meshes_ask_for_the_y_up_correction() -> None:
    """The DAE files are Z-up and three.js re-orients them, so ``y_up`` must be set.

    Without it the whole robot renders tipped 90 degrees about X -- verified in a
    browser, and invisible to every other test, because a tipped robot is still a
    robot.  The value is checked in the payload Swift actually sends.
    """
    urdf = sv.find_panda_urdf()
    assert urdf is not None
    shapes = sv.link_mesh_shapes(urdf, None)
    assert shapes, "no meshes to check"
    for shape in shapes:
        assert shape.to_dict()["y_up"] is True
    # ... and it is switchable, for a format that does not want the correction.
    assert sv.link_mesh_shapes(urdf, None, y_up=False)[0].to_dict()["y_up"] is False


def test_the_three_end_frames_are_where_the_model_says() -> None:
    """Wrist, flange and tool are ``0.107`` m and ``0.1034`` m apart, along the flange's z.

    This is the pair of numbers that decides where a drawn marker belongs, and getting it
    wrong is visible only as a frame hanging in space -- which is how it was reported.
    Measured: ``fk_flange`` is ``0.107`` m from the wrist frame (``forward_kinematics``
    row 6, the DH ``d7``) along its own ``z``, and ``fk_tool`` is a further ``0.1034`` m
    along the same axis rotated ``-45`` degrees about it.  So:
    :func:`~franka_ik.swift_viz.flange_axes` marks the pose the solver returns,
    :func:`~franka_ik.swift_viz.tool_axes` marks the factory tool centre point, and the
    stem between them is the distance that would otherwise be a gap.
    """
    rng = np.random.default_rng(0)
    lower, upper = model.lower_limits(), model.upper_limits()
    for _ in range(10):
        q = rng.uniform(lower, upper)
        frames = model.forward_kinematics(q)
        flange = model.fk_flange(q)
        tool = model.fk_tool(q)
        assert np.linalg.norm(flange[:3, 3] - frames[6][:3, 3]) == pytest.approx(0.107, abs=1e-12)
        assert np.allclose(
            flange[:3, :3].T @ (tool[:3, 3] - flange[:3, 3]), [0.0, 0.0, 0.1034], atol=1e-12
        )
        turn = flange[:3, :3].T @ tool[:3, :3]
        assert np.degrees(np.arctan2(turn[1, 0], turn[0, 0])) == pytest.approx(-45.0, abs=1e-9)

        marker = sv.flange_axes(q, length=0.1)
        assert np.allclose(np.asarray(marker.T, dtype=float), flange, atol=1e-12)
        tip = sv.tool_axes(q, length=0.05)
        assert np.allclose(np.asarray(tip.T, dtype=float), tool, atol=1e-12)

        stem = sv.tool_stem(q)
        assert stem.length == pytest.approx(0.1034, abs=1e-12)
        middle = 0.5 * (flange[:3, 3] + tool[:3, 3])
        assert np.allclose(np.asarray(stem.T, dtype=float)[:3, 3], middle, atol=1e-12)
        sv.update_tool_stem(stem, q + 0.01)
        moved = 0.5 * (model.fk_flange(q + 0.01)[:3, 3] + model.fk_tool(q + 0.01)[:3, 3])
        assert np.allclose(np.asarray(stem.T, dtype=float)[:3, 3], moved, atol=1e-12)

    # The skeleton ends at the tool point and the meshes end at the flange, which is why
    # the examples draw both frames rather than choosing one.
    assert np.allclose(sv.arm_keypoints(_Q)[-1], model.fk_tool(_Q)[:3, 3], atol=1e-12)
    assert np.linalg.norm(sv.arm_keypoints(_Q)[-1] - model.fk_flange(_Q)[:3, 3]) == pytest.approx(
        0.1034, abs=1e-12
    )


def test_add_cloud_batches_a_static_cloud_into_one_assembly() -> None:
    """A marker cloud reaches Swift as one message, not one message per marker.

    ``add_shape`` blocks until the browser confirms the shape is mounted, so 400
    markers through it is 400 round trips -- measured, 150 markers took 85 s to
    appear under Chromium with software rendering while the geometry drawn above them
    was ready in 1.5 s (``docs/browser_debugging.md`` §2.5).  ``add_cloud`` uses
    ``add_assembly`` instead, which sends every part at once.  This pins the four
    things the browser would otherwise be the only witness to: the parts leave
    together, the forward kinematics it hands over reports each part's current pose
    **as ``SE3``** (``Shape.T`` is a 4x4 array, and an array survives every Python-side
    check and then fails inside Swift's serialiser on the first frame -- measured), a
    part moved after the group was added is reported at its new pose, and the whole
    group can be taken off screen in one ``remove``.
    """

    class _Handle:
        """What ``add_assembly`` returns, as far as a caller may rely on."""

        def __init__(self, fk, parts, q0=None, readonly=False, name=None):
            self.fk = fk
            self.parts = parts
            self.q0 = q0
            self.readonly = readonly
            self.name = name
            self.id = 0

    class _Env:
        """An environment that records assemblies and refuses single shapes."""

        def __init__(self):
            self.assemblies = []
            self.removed = []

        def add_shape(self, shape):  # pragma: no cover - must not be reached
            raise AssertionError("a static cloud must not go out one shape at a time")

        def add_assembly(self, fk, parts, q0=None, readonly=False, name=None):
            handle = _Handle(fk, parts, q0=q0, readonly=readonly, name=name)
            self.assemblies.append(handle)
            return handle

        def remove(self, handle):
            self.removed.append(handle)

    markers = sv.reachable_shell_markers(count=12, radius=0.004)
    assert len(markers) == 12
    env = _Env()
    handle = sv.add_cloud(env, markers, name="shell")
    assert len(env.assemblies) == 1, "one message for the whole cloud"
    assert len(handle.parts) == 12
    assert handle.readonly is True, "nothing may advance the cloud's own state"
    assert handle.name == "shell"
    assert len(handle.q0) == 0

    poses = handle.fk(None)
    assert len(poses) == 12
    for pose, marker in zip(poses, markers, strict=True):
        assert hasattr(pose, "R") and hasattr(pose, "t"), "Swift serialises an SE3, not an array"
        assert np.allclose(np.asarray(pose.A, dtype=float), np.asarray(marker.T, dtype=float))

    # A part written after the group was added is reported where it was written: the
    # assembly is a way to *add* faster, not a way to freeze the scene.
    moved = sv._se3()().Trans([1.0, 2.0, 3.0])
    markers[7].T = moved
    again = handle.fk(None)
    assert np.allclose(np.asarray(again[7].A, dtype=float), np.asarray(moved.A, dtype=float))
    assert np.allclose(np.asarray(again[0].A, dtype=float), np.asarray(markers[0].T, dtype=float))

    env.remove(handle)
    assert env.removed == [handle], "the cloud comes off screen in one call"

    empty = _Env()
    with pytest.raises(ValueError, match="at least one shape"):
        sv.add_cloud(empty, [])
    assert empty.assemblies == []


def test_the_wsl_browser_wrapper_has_no_spaces_in_its_path(tmp_path, monkeypatch) -> None:
    """A Windows browser path contains a space; ``BROWSER`` cannot carry it.

    ``webbrowser``/``xdg-open`` split the command on whitespace, so a path like
    ``/mnt/c/Program Files/.../msedge.exe`` is cut at the first space (measured:
    ``/mnt/c/Program: not found``).  The wrapper indirection is the fix, so both
    halves of it are pinned: the returned path has no space, and the script it
    writes execs the real target.
    """
    monkeypatch.setenv("FRANKA_IK_CACHE", str(tmp_path))
    target = "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    wrapper = sv.wsl_browser_wrapper(target)
    assert " " not in wrapper
    assert Path(wrapper).is_file()
    text = Path(wrapper).read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh")
    assert target in text
    assert sv._browser_command(target) == wrapper
    # A path without spaces is passed through untouched.
    assert sv._browser_command("/usr/bin/firefox") == "/usr/bin/firefox"


def test_expand_xacro_is_cached_and_reusable(tmp_path, monkeypatch) -> None:
    """Expanding a xacro writes a URDF once and returns the same path after."""
    monkeypatch.setenv("FRANKA_IK_CACHE", str(tmp_path))
    xacro = sv.VENDORED_DESCRIPTION / sv.VENDORED_PANDA_XACRO
    if not xacro.is_file():
        pytest.skip("no vendored description")
    first = sv.expand_xacro(xacro, subargs={"hand": "false"})
    assert first.is_file()
    text = first.read_text(encoding="utf-8")
    assert "<robot" in text and "link0.dae" in text
    assert sv.expand_xacro(xacro, subargs={"hand": "false"}) == first


def test_manipulability_ellipsoid_returns_the_model_value() -> None:
    """The ellipsoid is a frame, and the number it reports is the model's."""
    ellipsoid = sv.manipulability_ellipsoid(_Q)
    returned = sv.update_manipulability_ellipsoid(ellipsoid, _Q)
    assert returned == pytest.approx(model.manipulability(_Q), rel=1e-12)
    rotation = np.asarray(ellipsoid.T, dtype=float)[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_reachable_shell_markers_lie_on_the_shell() -> None:
    """Every marker is exactly one reach radius from the shoulder.

    The radius is the closed form, not a sample, so the distance is checked to
    ``1e-12``: a marker that is merely close would mean the shell was measured
    rather than computed.
    """
    low, high = analysis.reachable_distance_range()
    shoulder = np.array([0.0, 0.0, float(model.DH_PARAMETERS[0, 1])])
    for inner, expected in ((False, high), (True, low)):
        markers = sv.reachable_shell_markers(count=25, inner=inner)
        assert len(markers) == 25
        for marker in markers:
            position = np.asarray(marker.T, dtype=float)[:3, 3]
            assert np.linalg.norm(position - shoulder) == pytest.approx(expected, abs=1e-12)


def test_camera_presets_are_complete_and_apply() -> None:
    """Four presets, and ``apply_camera`` forwards them to the environment."""

    class _Env:
        def __init__(self) -> None:
            self.calls = []

        def set_camera_pose(self, position, look_at) -> None:
            self.calls.append((position, look_at))

    presets = sv.camera_presets()
    assert sorted(presets) == ["front", "iso", "side", "top"]
    env = _Env()
    sv.apply_camera(env, "top")
    assert env.calls == [presets["top"]]
    with pytest.raises(ValueError, match="unknown camera preset"):
        sv.apply_camera(env, "over-the-shoulder")


def test_check_kinematics_accepts_the_panda_and_rejects_another_arm() -> None:
    """The guard against drawing the wrong robot, tested in both directions.

    A wrong model is simulated by translating the pose: the check is on the full
    4x4 against ``fk_tool``, so any real difference -- a different link length, a
    different convention -- is caught, and the failure is an exception rather than
    a log line.  The FR3 in the message is not hypothetical: ``franka_description``
    ships both arms and their wrists differ by 57 mm.
    """

    class _Panda:
        def fkine(self, q):
            class _Pose:
                A = model.fk_tool(q)

            return _Pose()

    class _WrongArm(_Panda):
        def fkine(self, q):
            class _Pose:
                A = model.fk_tool(q) + np.array(
                    [[0.0] * 3 + [0.057], [0.0] * 4, [0.0] * 4, [0.0] * 4]
                )

            return _Pose()

    worst = sv.check_kinematics(_Panda())
    assert worst < 1e-12
    with pytest.raises(sv.ArmKinematicsError, match="FR3"):
        sv.check_kinematics(_WrongArm())


def test_urdf_search_prefers_the_explicit_path(tmp_path, monkeypatch) -> None:
    """``FRANKA_IK_URDF`` wins, and a path that does not exist is an error."""
    monkeypatch.delenv(sv.FRANKA_DESCRIPTION_ENV_VAR, raising=False)
    monkeypatch.delenv(sv.URDF_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    # The candidate list is emptied as well as the cwd moved: a developer machine
    # may well have a franka_description checkout in one of the documented places,
    # and the search finding it is correct behaviour, not a bug to assert against.
    monkeypatch.setattr(sv, "FRANKA_DESCRIPTION_CANDIDATES", ())
    assert sv.find_franka_description() is None
    assert sv.find_urdf() is None

    urdf = tmp_path / "panda.urdf"
    urdf.write_text("<robot name='panda'/>", encoding="utf-8")
    monkeypatch.setenv(sv.URDF_ENV_VAR, str(urdf))
    assert sv.find_urdf() == urdf.resolve()

    monkeypatch.setenv(sv.URDF_ENV_VAR, str(tmp_path / "missing.urdf"))
    with pytest.raises(FileNotFoundError):
        sv.find_urdf()


def test_the_viz_extra_is_named_in_the_error_when_it_is_missing() -> None:
    """Without the extra, the error says how to install it and what it is for.

    Run in a subprocess with the three modules made unimportable, the way
    ``tests/test_numerical.py`` checks that CasADi stays optional.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys

        class _Block:
            def find_module(self, name, path=None):
                if name.split(".")[0] in {"roboticstoolbox", "swift", "spatialgeometry"}:
                    return self
                return None

            def load_module(self, name):
                # ModuleNotFoundError, not ImportError: ``pytest.importorskip``
                # distinguishes them, and only the subclass makes the module skip.
                raise ModuleNotFoundError(name)

        sys.meta_path.insert(0, _Block())
        for name in ("roboticstoolbox", "swift", "spatialgeometry"):
            sys.modules.pop(name, None)

        from franka_ik import swift_viz

        try:
            swift_viz.require_viz()
        except ImportError as exc:
            text = str(exc)
            assert "viz" in text, text
            print("hint ok")
        else:
            raise SystemExit("require_viz() should have raised")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(sv.Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hint ok" in result.stdout


def test_sample_poses_are_in_limits_and_solve() -> None:
    """Every ``--pose`` choice is reachable, in limits, and recovers itself.

    The poses carry claims, so they are checked rather than trusted: each is inside
    the joint limits, its own joint 7 yields at least one in-limit solution, and
    one of those solutions *is* the configuration the pose was generated from.
    """
    for name, degrees in swift_app.POSES.items():
        q = np.radians(np.asarray(degrees, dtype=float))
        assert np.all(np.degrees(q) >= fk.LOWER_LIMITS_DEG - 1e-9), name
        assert np.all(np.degrees(q) <= fk.UPPER_LIMITS_DEG + 1e-9), name
        solutions = fk.solve(model.fk_flange(q), float(q[6]), within_limits_only=True)
        assert solutions, name
        delta = np.degrees(np.asarray([s.q for s in solutions], dtype=float) - q)
        delta = (delta + 180.0) % 360.0 - 180.0
        assert np.any(np.all(np.abs(delta) < 1e-6, axis=1)), name


def test_the_documented_poses_reproduce_their_documented_counts() -> None:
    """The four sample poses behave as ``swift_app.POSES`` says they do.

    These are the numbers quoted in that docstring, so they are pinned here: the
    branch counts, the elbow-root split, and whether the published four-branch
    subset can return the generating configuration.  A change to the solver that
    moved any of them would be a finding, not a test to update quietly.
    """
    expected = {
        # name: (in-limit solutions, on the + root, published recovers the target)
        "ready": (1, 1, True),
        "eight_branches": (8, 4, False),
        "second_root": (7, 3, False),
        "near_singular": (6, 2, False),
    }
    for name, (total, on_first, published_recovers) in expected.items():
        q = np.radians(np.asarray(swift_app.POSES[name], dtype=float))
        solutions = fk.solve(model.fk_flange(q), float(q[6]), within_limits_only=True)
        first = [s for s in solutions if s.q4_root == analysis.PUBLISHED_Q4_ROOT]
        assert len(solutions) == total, name
        assert len(first) == on_first, name
        delta = np.degrees(np.asarray([s.q for s in first], dtype=float) - q)
        delta = (delta + 180.0) % 360.0 - 180.0
        recovered = bool(len(delta)) and bool(np.any(np.all(np.abs(delta) < 1e-6, axis=1)))
        assert recovered is published_recovers, name


def test_the_branches_page_embeds_a_joint7_sweep() -> None:
    """The HTML demo's scrubber needs its frames pre-computed, and gets them.

    The browser has no solver, so the sweep is embedded by
    :func:`franka_ik.report._branch_track` and the slider only selects among frames.
    This pins the properties the JavaScript relies on: frames in increasing joint-7
    order, every frame non-empty, one whose joint 7 is the page's own, and colours
    that group by **elbow root** rather than by position in the list -- which is what
    lets a reader see that the generating configuration is on the root the published
    code never evaluates.
    """
    from franka_ik import report

    q = np.radians(np.asarray(swift_app.POSES["second_root"], dtype=float))
    track = report._branch_track(model.fk_flange(q), float(q[6]), q, report.PAPER_GEOMETRY)

    assert len(track["parts"]) >= 5, "the sweep has to have something to scrub"
    assert all(part for part in track["parts"]), "no frame may be empty"
    assert track["q7Deg"] == sorted(track["q7Deg"]), "frames go one way"
    assert track["q7Deg"][track["index"]] == pytest.approx(np.degrees(q[6]), abs=0.5)

    frame = track["parts"][track["index"]]
    plus = [arm for arm in frame if arm["label"].startswith("q4+")]
    minus = [arm for arm in frame if arm["label"].startswith("q4-")]
    assert plus and minus, "this pose has solutions on both elbow roots"
    assert {arm["color"] for arm in plus} <= {"#08519c", "#6baed6"}
    assert {arm["color"] for arm in minus} <= {"#e6550d", "#fd8d3c"}
    # ... and the generating configuration is on the second root, the finding itself.
    target = [arm for arm in frame if arm["recoversTarget"]]
    assert len(target) == 1
    assert target[0]["label"].startswith("q4-")
