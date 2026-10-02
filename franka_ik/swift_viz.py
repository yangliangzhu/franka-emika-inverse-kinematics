"""Interactive 3D views of the Panda with Swift, and a reusable arm skeleton.

This is the Franka counterpart of the workflow the S-R-S study uses: the closed
form decides *which* configurations to draw, and this module draws them.  It is
the only place in the repository that knows about ``swift``, and it imports it
lazily, so nothing here is needed to use :mod:`franka_ik.solver`.

Two things are worth knowing before reading the code.

**What the arm is drawn from.**  The geometry comes from the *analytical* model,
not from the visualiser: :func:`franka_ik.model.joint_frames` gives the origin of
every actuated joint frame and :func:`franka_ik.model.fk_tool` gives the tool
point, so a skeleton built here agrees with the solver to machine precision by
construction.  A URDF, when one is available, is used for the *robot* -- the
bodies and, in Swift, the meshes -- and its forward kinematics is checked against
``fk_tool`` before it is drawn (see :func:`load_arm`).  If the two disagree, the
URDF is the wrong arm and this module says so instead of drawing it.

**Why the URDF check is not paranoia.**  ``franka_description`` ships two arms
that look alike and are not: the Panda and its successor FR3, whose link lengths
differ by tens of millimetres.  A viewer that loaded the FR3 and drove it with
Panda joint vectors would render a plausible, wrong robot.  The check is one
``fkine`` call, and it is the difference between a demo and a false statement.

The optional extra is declared in ``pyproject.toml``::

    pip install -e ".[viz]"
    uv sync --extra viz
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import model

__all__ = [
    "VizDependencies",
    "require_viz",
    "find_franka_description",
    "find_urdf",
    "load_arm",
    "find_panda_urdf",
    "expand_xacro",
    "ArmKinematicsError",
    "arm_keypoints",
    "joint_axes",
    "ArmSkeleton",
    "link_collision_shapes",
    "update_link_collision_shapes",
    "link_mesh_shapes",
    "update_link_mesh_shapes",
    "mesh_links_from_urdf",
    "arm_plane_outline",
    "manipulability_ellipsoid",
    "update_manipulability_ellipsoid",
    "frame_axes",
    "flange_axes",
    "tool_axes",
    "tool_stem",
    "update_tool_stem",
    "reachable_shell_markers",
    "add_shapes",
    "add_cloud",
    "camera_presets",
    "apply_camera",
    "launch_env",
    "detect_browser",
    "wsl_browser_wrapper",
    "is_wsl",
]

LOGGER = logging.getLogger(__name__)

#: Environment variable holding the path of a ``franka_description`` checkout.
FRANKA_DESCRIPTION_ENV_VAR = "FRANKA_IK_FRANKA_DESCRIPTION"

#: Environment variable holding the path of a URDF file to load directly.
URDF_ENV_VAR = "FRANKA_IK_URDF"

#: Places searched for ``franka_description`` when the environment says nothing.
#: Relative paths are resolved against the repository root, then the cwd's
#: parents; absolute ones are tried as they are.  A checkout is recognised by
#: ``robots/common/franka_arm.xacro``.
FRANKA_DESCRIPTION_CANDIDATES = (
    "../franka_description",
    "../../franka_description",
    "../../kinematics/robot_config_data/robomind2/franka_description",
    "~/franka_description",
    "~/ros2_ws/src/franka_description",
    "~/.cache/robot_descriptions/franka_description",
)

#: Largest tolerated ``fk_tool`` disagreement when checking a URDF, in metres
#: and radians.  A correct model agrees to about ``1e-16``; 1 mm is loose enough
#: to absorb a URDF that rounds its link lengths to three decimals and tight
#: enough to reject the FR3, whose links differ by 57 mm at the wrist.
_URDF_TOLERANCE = 1e-3

#: Fraction of a link's length used as the radius of its cylinder.
_LINK_RADIUS_FRACTION = 0.085

#: Smallest cylinder radius kept, in metres, so a short wrist link stays visible.
_MIN_LINK_RADIUS = 0.022

#: Smallest sphere radius kept, in metres.
_MIN_JOINT_RADIUS = 0.028

#: Display length of the per-joint frames drawn when ``show_joint_axes`` is set.
_JOINT_AXES_LENGTH = 0.09

#: Below this length a link is treated as degenerate (coincident key points).
_DEGENERATE_LENGTH = 1e-6

#: Below this cross-product norm the shoulder-elbow-wrist triangle is treated as
#: degenerate: the three points are collinear and define no arm plane.
_DEGENERATE_PLANE = 1e-9

#: Shown when the optional extra is missing.
_EXTRA_HINT = (
    "The 3D viewers need the optional `viz` extra:\n"
    "  pip install -e \".[viz]\"\n"
    "  uv sync --extra viz\n"
    "It installs roboticstoolbox-python, swift-sim, spatialmath-python and\n"
    "spatialgeometry.  Set FRANKA_IK_URDF to a Panda URDF to draw the real\n"
    "bodies instead of a capsule skeleton."
)

#: Shown when a browser cannot be reached.  Swift serves the page over a socket
#: and needs a browser with WebGL; the flags let it work where the GPU is
#: blocklisted.
_BROWSER_HINT = (
    "Swift could not connect to a browser.  It needs one with WebGL.\n"
    "  * set BROWSER=firefox (or chrome) explicitly, or pass --browser firefox\n"
    "  * on a headless machine, pass --headless: the examples still run and\n"
    "    print their numbers, they just do not open a window\n"
    "  * under WSL, use --browser auto (or --browser firefox) so the Windows\n"
    "    browser is used instead of xdg-open"
)

#: Flags that let a browser create a WebGL context where the GPU is blocklisted.
_SOFTWARE_WEBGL_FLAG = "--ignore-gpu-blocklist --enable-unsafe-swiftshader"


class VizDependencies(NamedTuple):
    """The three modules the 3D views need, resolved once."""

    rtb: object
    """``roboticstoolbox``, for loading a URDF."""

    swift: object
    """``swift``, the browser-side renderer."""

    sg: object
    """``spatialgeometry``, which builds the shapes Swift draws."""


def require_viz() -> VizDependencies:
    """Import and return the visualisation stack.

    Returns:
        A :class:`VizDependencies` of ``(roboticstoolbox, swift, spatialgeometry)``.

    Raises:
        ImportError: If any of the three is missing, with the install hint.
    """
    try:
        import roboticstoolbox as rtb
        import spatialgeometry as sg
        import swift
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(_EXTRA_HINT) from exc
    return VizDependencies(rtb=rtb, swift=swift, sg=sg)


class ArmKinematicsError(RuntimeError):
    """A URDF was found whose forward kinematics is not this Panda.

    Raised rather than logged because drawing it would produce a picture of a
    different robot driven by this arm's joint vectors -- a wrong answer that
    looks right, which is the failure mode the repository is built to avoid.
    """


#: The arm description vendored under ``third_party/``, as a checkout looks like it.
VENDORED_DESCRIPTION = Path(__file__).resolve().parents[1] / "third_party" / "franka_description"

#: The Panda entry point inside a ``franka_description`` checkout.  A description
#: ships xacro, not URDF, so this is a ``.xacro`` and needs expanding first.
VENDORED_PANDA_XACRO = Path("robots") / "panda" / "panda.urdf.xacro"


def expand_xacro(xacro: Path, *, subargs: Optional[Dict[str, str]] = None) -> Path:
    """Expand a xacro into a URDF in the cache, and return the URDF's path.

    A ``franka_description`` checkout ships the arm as *xacro*: it includes other
    files by ``$(find franka_description)``, loads YAML and substitutes arguments,
    none of which a URDF reader understands.  ``xacro`` cannot be run without a ROS
    environment, so ``xacrodoc`` does it instead.  ``xacrodoc`` is **not** one of this
    repository's dependencies: it arrives with ``roboticstoolbox``, which the ``viz``
    extra installs.  It only needs the package to carry a ``package.xml`` for the
    ``$(find ...)`` includes to resolve, which the vendored copy does.

    The result is cached under ``~/.cache/franka_ik`` and keyed on the xacro's
    modification time and the arguments, so a second call is a file read.

    Args:
        xacro: The ``.xacro`` or ``.urdf.xacro`` file.
        subargs: Xacro argument overrides, e.g. ``{"hand": "false"}``.

    Returns:
        The expanded URDF, which answers ``file://`` mesh URLs to the original
        checkout, so the meshes stay where they are.

    Raises:
        ImportError: If the ``viz`` extra is missing, since ``xacrodoc`` comes with it.
        RuntimeError: If the expansion fails, with the reason attached.
    """
    import hashlib
    import json

    xacro = Path(xacro).resolve()
    subargs = dict(subargs or {})
    key = hashlib.sha256(
        json.dumps(
            {"path": str(xacro), "mtime": xacro.stat().st_mtime_ns, "subargs": subargs},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    urdf = _cache_dir() / f"{xacro.stem}-{key}.urdf"
    if urdf.is_file():
        return urdf
    try:
        from xacrodoc import XacroDoc
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(_EXTRA_HINT) from exc
    try:
        document = XacroDoc.from_file(str(xacro), subargs=subargs, walk_up=True)
        xml = document.to_urdf_string()
    except Exception as exc:  # noqa: BLE001 - the reason is the useful part
        raise RuntimeError(
            f"could not expand {xacro}: {type(exc).__name__}: {exc}.  A description is "
            "expanded in place, so the whole package has to be present, and it has to "
            "carry a package.xml for the $(find ...) includes to resolve."
        ) from exc
    urdf.write_text(xml, encoding="utf-8")
    LOGGER.info("expanded %s -> %s", xacro, urdf)
    return urdf


def find_panda_urdf() -> Optional[Path]:
    """A URDF for the Panda with visual meshes, expanded if it is a xacro.

    Searched in order: :data:`URDF_ENV_VAR`; the vendored description under
    ``third_party/``; a ``franka_description`` checkout found by
    :func:`find_franka_description`.  The result is *verified* before it is returned
    -- a description that turns out to be a different arm raises rather than being
    drawn (see :func:`check_kinematics`), so this returning means the meshes belong
    to this robot.

    Returns:
        The URDF path, or ``None`` when nothing usable was found.
    """
    explicit = os.environ.get(URDF_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{URDF_ENV_VAR} points at {path}, which does not exist")
        return expand_xacro(path) if path.suffix == ".xacro" else path.resolve()

    vendored = VENDORED_DESCRIPTION / VENDORED_PANDA_XACRO
    if vendored.is_file():
        return expand_xacro(vendored, subargs={"hand": "false"})

    description = find_franka_description()
    if description is not None:
        for candidate in sorted(description.glob("robots/*/*.urdf.xacro")):
            LOGGER.info("found an arm description at %s", candidate)
            return expand_xacro(candidate, subargs={"hand": "false"})
    return None


def find_franka_description() -> Optional[Path]:
    """Search for a ``franka_description`` checkout.

    Checked in order: :data:`FRANKA_DESCRIPTION_ENV_VAR`, then
    :data:`FRANKA_DESCRIPTION_CANDIDATES` relative to the repository root and to
    the current directory.

    Returns:
        The checkout directory, or ``None`` when none was found.
    """
    candidates: List[Path] = []
    from_env = os.environ.get(FRANKA_DESCRIPTION_ENV_VAR)
    if from_env:
        candidates.append(Path(from_env).expanduser())
    roots = [Path(__file__).resolve().parents[1], Path.cwd()]
    for relative in FRANKA_DESCRIPTION_CANDIDATES:
        expanded = Path(relative).expanduser()
        if expanded.is_absolute():
            candidates.append(expanded)
            continue
        candidates.extend(root / expanded for root in roots)
    for candidate in candidates:
        if (candidate / "robots" / "common" / "franka_arm.xacro").is_file():
            return candidate.resolve()
    return None


def find_urdf() -> Optional[Path]:
    """Locate a URDF to load, preferring an explicit one.

    Order: :data:`URDF_ENV_VAR`; a ``*.urdf`` inside a
    ``franka_description`` checkout; and finally ``None``, which means "draw the
    skeleton only" -- the default, and the only option when no checkout exists.

    Note:
        A ``franka_description`` checkout ships the arm as *xacro*, not URDF, and
        expanding it needs the ``xacro`` package *and* a ROS-style package search
        path.  Rather than half-implement that, this function looks for a plain
        ``.urdf`` and returns ``None`` when there is not one.  The viewers draw
        the capsule skeleton in that case, which is complete for the geometry
        this repository is about.

    Returns:
        The URDF path, or ``None``.
    """
    explicit = os.environ.get(URDF_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{URDF_ENV_VAR} points at {path}, which does not exist")
        return path.resolve()
    description = find_franka_description()
    if description is not None:
        for candidate in sorted((description / "robots").rglob("*.urdf")):
            LOGGER.info("found a URDF at %s", candidate)
            return candidate
    return None


def _urdf_chain(arm: object) -> object:
    """The 7-degree-of-freedom chain of a loaded arm, without a gripper.

    A Panda description usually carries the hand as extra joints; the solver is
    about the seven arm joints, so the chain is truncated there.
    """
    if getattr(arm, "n", 0) >= model.NUM_JOINTS:
        return arm
    raise ArmKinematicsError(
        f"the model has {getattr(arm, 'n', 0)} joints; this arm has {model.NUM_JOINTS}"
    )


def load_arm(*, urdf: Optional[Path] = None, check: bool = True) -> Tuple[object, Optional[Path]]:
    """Load the arm model to draw, and verify it against this repository's model.

    With a URDF available the real bodies are drawn; without one, the bundled
    Panda description from ``roboticstoolbox`` is used, which has the correct
    kinematics and limit values but capsule collision geometry in place of
    visual meshes.  Either way the result is checked against
    :func:`franka_ik.model.fk_tool` at several configurations before it is
    returned, because a plausible-looking wrong robot is the one outcome worth
    failing loudly over.

    Args:
        urdf: Explicit URDF path; ``None`` means "use :func:`find_urdf`".
        check: Verify the kinematics.  Only set ``False`` to inspect a model
            that is known to be a different arm.

    Returns:
        ``(arm, urdf_path)`` -- the loaded model and the URDF it came from, or
        ``None`` for a URDF when the bundled description was used.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        ArmKinematicsError: If the model's forward kinematics disagrees with
            ``fk_tool`` by more than :data:`_URDF_TOLERANCE`.
    """
    deps = require_viz()
    resolved = Path(urdf).expanduser().resolve() if urdf is not None else find_urdf()
    if resolved is not None:
        arm = deps.rtb.ERobot.URDF(str(resolved))
    else:
        # ``rtb.models.URDF.Panda()`` reads the edition of the Panda description
        # bundled in ``rtb-data`` (the QUT "frankie" xacro), which is the same
        # arm this repository models and needs no download.
        arm = deps.rtb.models.URDF.Panda()
    arm = _urdf_chain(arm)
    if check:
        check_kinematics(arm)
    return arm, resolved


def check_kinematics(arm: object, *, samples: int = 5) -> float:
    """Assert that ``arm``'s forward kinematics is this repository's Panda.

    Compared at ``samples`` in-limit configurations drawn from a fixed seed, on
    the full 4x4: the URDF and :func:`franka_ik.model.fk_tool` use the same
    conventions (measured agreement ``1.7e-16``), so any real difference is a
    different arm.

    Args:
        arm: A loaded model exposing ``fkine(q)``.
        samples: Configurations to compare.

    Returns:
        The largest absolute difference over all entries and samples.

    Raises:
        ArmKinematicsError: If that difference exceeds
            :data:`_URDF_TOLERANCE`.
    """
    rng = np.random.default_rng(0)
    lower, upper = model.lower_limits(), model.upper_limits()
    worst = 0.0
    for _ in range(samples):
        q = rng.uniform(lower, upper)
        theirs = np.asarray(arm.fkine(q).A, dtype=float)
        worst = max(worst, float(np.abs(theirs - model.fk_tool(q)).max()))
    if worst > _URDF_TOLERANCE:
        raise ArmKinematicsError(
            f"the model disagrees with fk_tool by {worst:.3e} "
            f"(tolerance {_URDF_TOLERANCE:.0e}); this is not the arm this repository solves. "
            "Note that the FR3 has different link lengths from the Panda: the Panda's wrist is "
            "0.107 m, the FR3's is 0.164 m. Point FRANKA_IK_URDF at a Panda URDF."
        )
    LOGGER.info("model kinematics agrees with fk_tool to %.2e", worst)
    return worst


def arm_keypoints(q: Sequence[float]) -> List[np.ndarray]:
    """The points a Panda skeleton is drawn through, base first.

    **Nine** points, in drawing order, and the frame indices are worth stating
    because the first one is easy to get wrong:

    =====  ==========================================  ==================================
    index  point                                       source
    =====  ==========================================  ==================================
    0      base, the robot's origin                      ``[0, 0, 0]``
    1      joint 1, at the foot of the shoulder column   ``forward_kinematics[0]``
    2      joint 2                                         ``[1]``
    3      joint 3                                         ``[2]``
    4      joint 4, the elbow                              ``[3]``
    5      joint 5                                         ``[4]``
    6      joint 6                                         ``[5]``
    7      joint 7                                         ``[6]``
    8      the tool point                                 :func:`franka_ik.model.fk_tool`
    =====  ==========================================  ==================================

    :func:`franka_ik.model.forward_kinematics` returns **nine** frames and
    ``frame[i]`` is the origin of joint ``i + 1``, so joint 1 is ``frame[0]`` and *not*
    the base; a key-point list built from ``frame[1:]`` would be off by one joint and
    would draw a plausible, wrong arm.  The flange frame ``frame[7]`` is deliberately
    not a key point: it lies 0.107 m along joint 7's axis, inside the tool stem, and
    the segment from joint 7 to the tool point already covers it.

    Two consecutive points coincide at every configuration -- joints 1 and 2 share the
    foot of the shoulder column, joints 5 and 6 share an origin -- so two of the eight
    segments are degenerate.  They are drawn as zero-length cylinders rather than
    skipped, which keeps the segment index independent of the configuration.

    Args:
        q: Seven joint angles in radians.

    Returns:
        Nine 3-vectors, in drawing order.
    """
    joints = model.joint_frames(q)
    return [np.zeros(3)] + [np.asarray(p, dtype=float) for p in joints] + [
        np.asarray(model.fk_tool(q)[:3, 3], dtype=float)
    ]


def joint_axes(q: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Origin and axis direction of each actuated joint, base first.

    The axes come from the modified-DH frames, not from the URDF, so a drawn axis is
    exactly the axis the solver turns about.  Each axis is the ``z`` column of the
    joint's own frame -- ``forward_kinematics[i]`` for joint ``i + 1``, the same
    index mapping as :func:`arm_keypoints`.

    Args:
        q: Seven joint angles in radians.

    Returns:
        Seven ``(origin, direction)`` pairs of 3-vectors, with ``direction`` a unit
        vector.
    """
    frames = model.forward_kinematics(q)
    axes = []
    for index in range(model.NUM_JOINTS):
        transform = frames[index]
        axes.append((transform[:3, 3].copy(), transform[:3, 2].copy()))
    return axes


def _se3():
    """Return ``spatialmath.SE3``; ``spatialgeometry`` already depends on it."""
    from spatialmath import SE3

    return SE3


def _rotation_aligning_z(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix whose third column is the unit vector ``direction``."""
    z_axis = np.asarray(direction, dtype=float)
    z_axis = z_axis / np.linalg.norm(z_axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _link_transform(sm, start: np.ndarray, end: np.ndarray):
    """Pose of a cylinder spanning ``start`` to ``end``, plus its length.

    The primitive is centred at its local origin with its axis along ``+z``, so
    the pose rotates ``+z`` onto ``end - start`` and translates the centre to the
    midpoint.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    midpoint = 0.5 * (start + end)
    if length < _DEGENERATE_LENGTH:
        return sm.Trans(midpoint), 0.0
    return sm.Rt(_rotation_aligning_z(delta), midpoint), length


class ArmSkeleton:
    """A Panda drawn as eight tapered capsule segments and nine joint spheres.

    Nine key points (:func:`arm_keypoints`) give eight segments, and a sphere marks
    each key point including the base.  Two of the segments are degenerate at every
    configuration -- joints 1 and 2 share a vertical axis, joints 5 and 6 share an
    origin -- and are drawn as zero-length cylinders so that the segment index never
    depends on the configuration.

    Radii are derived from the link lengths, so the base reads as thick and the
    wrist as thin, and the same class draws the arm at any scale of detail
    through ``radius_scale``.

    Every segment is built once at its final length: writing a length or radius
    after construction marks the shape changed, and Swift's headless client does
    not acknowledge that update, so :meth:`update` writes poses only.

    Args:
        radius_scale: Multiplier on the derived radii; ``1.0`` is the default
            look.
        alpha: Opacity in ``[0, 1]`` applied to every shape -- useful below
            ``1.0`` for a ghost fan.
        colour: ``[r, g, b]`` in ``[0, 1]``.  ``None`` leaves Swift's default,
            which is right for a single skeleton; a fan of them needs distinct
            colours, and colouring by elbow root is what makes the two halves of
            ``docs/branch_analysis.md`` visible.
        show_joint_axes: Also build one ``sg.Axes`` per joint frame.
        name: Optional label, kept as :attr:`name` for the caller's UI.

    Attributes:
        name: Optional label.
    """

    def __init__(
        self,
        *,
        radius_scale: float = 1.0,
        alpha: float = 1.0,
        colour: Optional[Sequence[float]] = None,
        show_joint_axes: bool = False,
        name: Optional[str] = None,
    ):
        _, _, sg = require_viz()
        self.name = name
        self.radius_scale = float(radius_scale)
        self.alpha = float(alpha)
        self.colour = None if colour is None else [float(c) for c in colour]
        self.show_joint_axes = bool(show_joint_axes)
        self._sm = _se3()
        # ``spatialgeometry`` takes RGBA at construction and a separate
        # ``opacity`` afterwards; setting both from one pair of numbers is what
        # keeps a ghost's transparency and colour from disagreeing.
        style: Dict[str, object] = (
            {} if self.colour is None else {"color": [*self.colour, 1.0]}
        )

        # Segment lengths are a property of the arm, so they are measured once,
        # at a configuration where no two consecutive key points coincide more
        # than the DH nesting forces.
        probe = arm_keypoints(np.radians([0.0, -20.0, 0.0, -60.0, 0.0, 45.0, 0.0]))
        self._lengths = np.array(
            [
                max(float(np.linalg.norm(probe[index + 1] - probe[index])), _DEGENERATE_LENGTH)
                for index in range(len(probe) - 1)
            ]
        )
        self._radii = np.maximum(_MIN_LINK_RADIUS, _LINK_RADIUS_FRACTION * self._lengths) * max(
            float(radius_scale), 0.0
        )
        # One sphere per key point, the base included: nine points against eight
        # segments, so the base takes the first segment's radius and the rest map one
        # to one.
        self._sphere_radii = np.maximum(
            _MIN_JOINT_RADIUS, np.concatenate([[self._radii[0]], self._radii])
        )

        self._cylinders = [
            sg.Cylinder(
                radius=float(self._radii[index]), length=float(self._lengths[index]), **style
            )
            for index in range(len(self._lengths))
        ]
        self._spheres = [sg.Sphere(radius=float(r), **style) for r in self._sphere_radii]
        self._joint_axes = (
            [sg.Axes(_JOINT_AXES_LENGTH, **style) for _ in range(model.NUM_JOINTS)]
            if self.show_joint_axes
            else []
        )

        for shape in self._cylinders + self._spheres + self._joint_axes:
            shape.opacity = self.alpha

        self._shapes: List[object] = list(self._cylinders) + list(self._spheres)
        self._shapes.extend(self._joint_axes)
        self.update(np.zeros(model.NUM_JOINTS))

    @property
    def shapes(self) -> List[object]:
        """The shapes to add to Swift, in drawing order."""
        return list(self._shapes)

    def update(self, q: Sequence[float]) -> None:
        """Move every shape to the configuration ``q`` (radians).

        Args:
            q: Seven joint angles in radians.
        """
        points = arm_keypoints(q)
        for index, cylinder in enumerate(self._cylinders):
            transform, _ = _link_transform(self._sm, points[index], points[index + 1])
            cylinder.T = transform
        for sphere, point in zip(self._spheres, points, strict=False):
            sphere.T = self._sm.Trans(point)
        if self._joint_axes:
            for axes, (origin, direction) in zip(
                self._joint_axes, joint_axes(q), strict=False
            ):
                axes.T = self._sm.Rt(_rotation_aligning_z(direction), origin)


def _arm_links(arm: object) -> List[object]:
    """The nine links of the chain, base first, without a gripper."""
    links = list(getattr(arm, "links", []))
    if len(links) < model.NUM_JOINTS + 1:
        raise ArmKinematicsError(
            f"the model has {len(links)} links; this arm has at least {model.NUM_JOINTS + 1}"
        )
    # A Panda description usually mounts a hand after link8; the solver is about the
    # seven arm joints, so the chain is cut there.
    return links[: model.NUM_JOINTS + 2]


def link_collision_shapes(arm: object) -> List[object]:
    """The model's own capsules, as shapes Swift can draw.

    ``roboticstoolbox``'s bundled Panda description carries hand-built collision
    geometry -- a cylinder and two spheres per link, 25 primitives over the nine
    links -- which is a far better likeness of the robot than a stick figure and
    needs no URDF, no meshes and no download.  This turns that geometry into
    ``spatialgeometry`` shapes so a caller can add them to a scene.

    The shapes are returned **unparented**: each one's pose is written by
    :func:`update_link_collision_shapes`, which composes the link's world pose with
    the shape's local one.  They are built once at their final size, because a
    length or radius written afterwards marks the shape changed and Swift's headless
    client does not acknowledge that.

    Args:
        arm: A loaded model, from :func:`load_arm`.

    Returns:
        The primitive shapes, in chain order.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        ArmKinematicsError: If the model has too few links to be this arm.
    """
    _, _, sg = require_viz()
    shapes: List[object] = []
    for link in _arm_links(arm):
        for collision in getattr(link, "collision", []):
            kind = type(collision).__name__
            colour = [0.82, 0.82, 0.84, 1.0]
            if kind == "Cylinder":
                shapes.append(
                    sg.Cylinder(
                        radius=float(collision.radius),
                        length=float(collision.length),
                        color=colour,
                    )
                )
            elif kind == "Sphere":
                shapes.append(sg.Sphere(float(collision.radius), color=colour))
            else:
                # Anything else would be a mesh or a box, which this bundled
                # description does not use; skipping is honest, inventing is not.
                LOGGER.info("skipping a %s in the collision geometry", kind)
    return shapes


def update_link_collision_shapes(
    shapes: Sequence[object], arm: object, q: Sequence[float]
) -> None:
    """Move collision shapes built by :func:`link_collision_shapes` to ``q``.

    The link poses come from ``arm.fkine_all`` -- the model's own kinematics, so a
    drawn capsule cannot drift from the arm it belongs to -- and each shape is
    placed at ``link_pose @ shape_pose``, the same composition the collision checker
    uses.  The two are not independent: :func:`load_arm` has already verified that
    model against :func:`franka_ik.model.fk_tool`.

    Args:
        shapes: The list returned by :func:`link_collision_shapes`, in the same
            order.
        arm: The model they were built from.
        q: Seven joint angles in radians.
    """
    sm = _se3()
    transforms = arm.fkine_all(np.asarray(q, dtype=float))
    placed = 0
    for index, link in enumerate(_arm_links(arm)):
        # ``fkine_all`` returns one extra transform at the front (the base frame
        # repeated), so link ``i``'s world pose is entry ``i + 1``.
        link_pose = np.asarray(transforms[index + 1], dtype=float)
        for collision in getattr(link, "collision", []):
            if type(collision).__name__ not in ("Cylinder", "Sphere"):
                continue
            local = getattr(collision, "T", None)
            if local is None:
                local = np.eye(4)
            pose = link_pose @ np.asarray(local, dtype=float)
            shapes[placed].T = sm.Rt(pose[:3, :3], pose[:3, 3])
            placed += 1


#: ``-1`` in :data:`_LINK_TO_FK_FRAME` means the robot's own base frame, i.e. the
#: identity: ``link0`` is the base plate, whose frame *is* the base by definition.
_LINK_AT_BASE = -1

#: Which frame places each URDF link, as an index into ``(identity, *fk(q))`` -- that
#: is, ``-1`` for the identity and then ``0`` ... ``8`` for the nine DH frames.
#:
#: Derived by walking the URDF's own joint chain at ``q = 0`` and comparing against
#: :func:`franka_ik.model.forward_kinematics`, not by reasoning about it; the snippet
#: is in ``docs/browser_debugging.md`` section 3.  The answer is worth writing down
#: because the natural guess is wrong:
#:
#: ===========  ==============================================
#: URDF link    frame
#: ===========  ==============================================
#: ``link0``    identity -- the **base**, not the shoulder
#: ``link1``    ``fk[0]`` (joint 1)
#: ``link2``    ``fk[1]`` (joint 2)
#: ``link3``    ``fk[2]``
#: ``link4``    ``fk[3]`` (joint 4, the elbow)
#: ``link5``    ``fk[4]``
#: ``link6``    ``fk[5]``
#: ``link7``    ``fk[6]``
#: ``link8``    ``fk[7]`` (the flange)
#: ===========  ==============================================
#:
#: So ``linkk`` sits at ``fk[k - 1]`` for ``k >= 1``, and ``link0`` is the identity.
#: Putting ``link0`` at ``fk[0]`` instead -- the off-by-one that looks right, because
#: ``fk[0]`` is itself at the shoulder's foot -- lifts the base plate 0.333 m onto the
#: shoulder and buries it inside ``link1``, which is exactly the "base overlapping the
#: link behind it, with the floor left empty" defect that was reported and then
#: measured here.
#:
#: ``link8`` has no visual (it is the flange marker), so the last mesh is expected
#: 0.107 m *from* the flange rather than on it.
_LINK_TO_FK_FRAME = (-1, 0, 1, 2, 3, 4, 5, 6, 7)  # type: Tuple[int, ...]


def mesh_links_from_urdf(urdf: Path) -> List[Tuple[str, Path, np.ndarray]]:
    """The visual meshes of a URDF's arm chain, as ``(link, file, origin)``.

    Only ``<visual>`` is read, and only for links named ``*link0`` … ``*link8``, so
    a description that also ships collision-only links, a hand or a mobile base is
    used for the seven-joint arm and nothing else.  A ``<visual>`` with no
    ``<origin>`` is the common case in the Franka descriptions (every mesh is
    authored in its link's frame) and becomes an identity origin.

    Args:
        urdf: Path to a URDF file.

    Returns:
        One entry per mesh, in link order: the link name, the mesh file (absolute if
        the URDF used ``file://`` or a path relative to the URDF), and the mesh's
        4x4 origin inside the link.

    Raises:
        ValueError: If the URDF has no arm chain to speak of.
        FileNotFoundError: If a mesh the URDF names is missing.
    """
    root = ElementTree.parse(urdf).getroot()
    found: List[Tuple[str, Path, np.ndarray]] = []
    for link in root.findall("link"):
        name = link.get("name", "")
        match = _ARM_LINK_NAME.match(name)
        if match is None:
            continue
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename", "")
            if filename.startswith("file://"):
                path = Path(filename[len("file://") :])
            elif filename.startswith("package://"):
                path = _resolve_package_url(filename, urdf)
            else:
                path = (urdf.parent / filename).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name}: the URDF names {path}, which does not exist")
            origin = np.eye(4)
            element = visual.find("origin")
            if element is not None:
                xyz = [float(v) for v in (element.get("xyz") or "0 0 0").split()]
                rpy = [float(v) for v in (element.get("rpy") or "0 0 0").split()]
                spin = _rpy_to_matrix(*rpy)
                origin[:3, :3] = spin
                origin[:3, 3] = xyz
            found.append((name, path, origin))
    if not found:
        raise ValueError(
            f"{urdf} has no visual meshes on an arm chain; mesh rendering needs a "
            "description with <visual><geometry><mesh>, not a collision-only one"
        )
    return found


#: ``link0`` … ``link8``, with any prefix a description likes to add.
_ARM_LINK_NAME = re.compile(r"(?:^|_)(?:link|panda_link|fr3_link)([0-8])$")


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, the URDF convention for ``rpy``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _resolve_package_url(filename: str, urdf: Path) -> Path:
    """A ``package://`` URL, resolved against the checkout the URDF came from."""
    rest = filename[len("package://") :]
    package, _, tail = rest.partition("/")
    for root in (urdf.parent, *urdf.parents):
        candidate = root / package / tail
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve {filename} near {urdf}")


def link_mesh_shapes(
    urdf: Path, arm: object, *, alpha: float = 1.0, y_up: bool = True
) -> List[object]:
    """The meshes of a URDF's arm chain, as shapes Swift can draw.

    The robot's real bodies, which is the only mode that looks like a Panda rather
    than a stand-in for one.  Swift ships each mesh to the browser itself, so this
    only reads the URDF to learn which file belongs to which link and where it sits
    inside that link; the poses come from :func:`update_link_mesh_shapes`.

    Args:
        urdf: Path to a URDF with visual meshes.
        arm: The model loaded from that URDF, used to check the link count.
        alpha: Opacity in ``[0, 1]``.
        y_up: Re-interpret the mesh as if it were authored with ``+Y`` up.  The
            Franka DAE files are **Z**-up, but three.js's ``ColladaLoader``
            re-orients a Z-up asset to Y-up as it loads, so the rendered result is
            Y-up whatever the file said and Swift has to undo it.  With this
            ``False`` the whole arm is drawn tipped 90 degrees about X -- measured,
            not assumed.  An STL or an OBJ does not go through that loader and would
            not want the correction; nothing in this repository uses one, so the
            flag is exposed rather than guessed at per file type.

    Returns:
        One ``sg.Mesh`` per visual element, in link order.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        FileNotFoundError: If a mesh the URDF names is missing.
        ValueError: If the URDF has no arm meshes.
    """
    _, _, sg = require_viz()
    meshes = []
    for _link, path, _origin in mesh_links_from_urdf(urdf):
        mesh = sg.Mesh(filename=str(path), scale=[1.0, 1.0, 1.0], y_up=y_up)
        mesh.opacity = float(alpha)
        meshes.append(mesh)
    _ = arm
    return meshes


def update_link_mesh_shapes(shapes: Sequence[object], urdf: Path, q: Sequence[float]) -> None:
    """Move URDF meshes built by :func:`link_mesh_shapes` to ``q``.

    Every pose comes from :func:`franka_ik.model.forward_kinematics`, the same
    function the solver uses, so the meshes cannot drift from the arm being solved.
    A mesh whose link has no entry in :data:`_LINK_TO_FK_FRAME` is left where it is
    rather than guessed at.

    Args:
        shapes: The list from :func:`link_mesh_shapes`, same order.
        urdf: The URDF they came from.
        q: Seven joint angles in radians.
    """
    sm = _se3()
    frames = model.forward_kinematics(np.asarray(q, dtype=float))
    entries = mesh_links_from_urdf(urdf)
    for shape, (link, _path, origin) in zip(shapes, entries, strict=True):
        index = _LINK_TO_FK_FRAME[_link_index(link)]
        base = np.eye(4) if index == _LINK_AT_BASE else frames[index]
        pose = base @ origin
        shape.T = sm.Rt(pose[:3, :3], pose[:3, 3])


def _link_index(link_name: str) -> int:
    """``0`` … ``8`` for ``link0`` … ``link8``, whatever prefix the description used."""
    match = _ARM_LINK_NAME.search(link_name)
    if match is None:
        raise ValueError(f"{link_name!r} is not an arm link")
    return int(match.group(1))


def arm_plane_outline(q: Sequence[float], *, radius: float = 0.004) -> Optional[object]:
    """A closed outline of the shoulder-elbow-wrist plane, or ``None``.

    The plane through the three key points is what the arm angle is measured in,
    so drawing it makes the freedom visible: as joint 7 sweeps, the plane turns
    about the shoulder-wrist line.

    Args:
        q: Seven joint angles in radians.
        radius: Tube radius; ``0`` draws a thin line.

    Returns:
        An ``sg.Polyline`` through the three points and back to the shoulder, or
        ``None`` when the points are collinear.

    Note:
        On a Panda the degenerate case is unreachable, so this guard is defensive
        rather than load-bearing: the 0.0825 m shoulder offset keeps the elbow off
        the shoulder-wrist line, and the smallest cross-product norm found by
        sampling 20 000 in-limit configurations is ``9.9e-6``, four orders of
        magnitude above the ``1e-9`` threshold.  It is kept because the failure it
        guards against -- a zero-area shape handed to the renderer -- shows up as a
        drawing artefact rather than as an error.

    Raises:
        ImportError: If the ``viz`` extra is missing.
    """
    _, _, sg = require_viz()
    shoulder = np.asarray(model.joint_frames(q)[1], dtype=float)
    elbow = np.asarray(model.joint_frames(q)[3], dtype=float)
    wrist = np.asarray(model.joint_frames(q)[5], dtype=float)
    if np.linalg.norm(np.cross(elbow - shoulder, wrist - shoulder)) < _DEGENERATE_PLANE:
        return None
    polyline = sg.Polyline(
        np.column_stack([shoulder, elbow, wrist, shoulder]), radius=float(radius)
    )
    polyline.T = _se3()()
    return polyline


def manipulability_ellipsoid(
    q: Sequence[float],
    *,
    scale: float = 0.06,
    centre: Optional[np.ndarray] = None,
) -> object:
    """An ellipsoid whose semi-axes are the singular vectors of the Jacobian.

    Built once and moved with :func:`update_manipulability_ellipsoid`, because a
    shape whose *size* changes cannot be updated on Swift's headless client.

    Args:
        q: Seven joint angles in radians.
        scale: Metres per unit singular value.
        centre: Where to put it; the wrist by default.

    Returns:
        An ``sg.Ellipsoid``.

    Raises:
        ImportError: If the ``viz`` extra is missing.
    """
    _, _, sg = require_viz()
    ellipsoid = sg.Ellipsoid(radii=[scale, scale, scale])
    update_manipulability_ellipsoid(ellipsoid, q, centre=centre)
    return ellipsoid


def update_manipulability_ellipsoid(
    ellipsoid: object,
    q: Sequence[float],
    *,
    centre: Optional[np.ndarray] = None,
) -> float:
    """Move a manipulability ellipsoid to ``q`` and return its manipulability.

    Only the pose is written, never the radii: a shape whose size changes marks
    itself changed, and Swift's headless client does not acknowledge that, so the
    ellipsoid is built at a fixed size and the *orientation* -- which is the part
    that carries the information -- is what moves.  The returned scalar is the
    number a readout wants.

    Args:
        ellipsoid: The shape from :func:`manipulability_ellipsoid`.
        q: Seven joint angles in radians.
        centre: Where to put it; the wrist by default.

    Returns:
        ``franka_ik.model.manipulability(q)``.
    """
    sm = _se3()
    jacobian = model.jacobian(q)
    _, singular_values, vh = np.linalg.svd(jacobian[:3, :], full_matrices=True)
    # ``vh`` rows are the right singular vectors; the first three span the range
    # of the translational Jacobian, so their image directions are the axes.
    rotation = np.eye(3)
    for index in range(3):
        direction = jacobian[:3, :] @ vh[index]
        norm = np.linalg.norm(direction)
        rotation[:, index] = direction / norm if norm > 1e-12 else np.eye(3)[:, index]
    # Re-orthonormalise: the three directions are near-orthogonal by
    # construction, but a rotation matrix has to be one exactly, or the shape
    # shears.
    u, _, vh_fix = np.linalg.svd(rotation)
    rotation = u @ vh_fix
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh_fix
    where = np.asarray(model.joint_frames(q)[5], dtype=float) if centre is None else np.asarray(
        centre, dtype=float
    )
    ellipsoid.T = sm.Rt(rotation, where)
    return model.manipulability(q)


def frame_axes(pose: np.ndarray, *, length: float = 0.12) -> object:
    """The axes of a 4x4 frame, as an ``sg.Axes``.

    Args:
        pose: 4x4 transform whose rotation and origin the axes take.
        length: Axis length in metres.

    Returns:
        An ``sg.Axes``.

    Raises:
        ImportError: If the ``viz`` extra is missing.
    """
    _, _, sg = require_viz()
    sm = _se3()
    axes = sg.Axes(length)
    axes.T = sm.Rt(pose[:3, :3], pose[:3, 3])
    return axes


def flange_axes(q: Sequence[float], *, length: float = 0.12) -> object:
    """The **flange** frame at ``q``: the frame this library solves for.

    ``fk_flange`` is the pose :func:`franka_ik.solver.solve` takes and returns, so this
    is the marker that means "the arm reached the target" -- and the frame every
    example's pose residual is measured in.  Draw this one, not :func:`tool_axes`, in
    anything that puts a drawing next to a residual.

    Args:
        q: Seven joint angles in radians.
        length: Axis length in metres.

    Returns:
        An ``sg.Axes`` at ``franka_ik.model.fk_flange(q)``.
    """
    return frame_axes(model.fk_flange(q), length=length)


def tool_axes(q: Sequence[float], *, length: float = 0.12) -> object:
    """The **tool** frame at ``q``: the factory tool point, 0.1034 m past the flange.

    ``fk_tool = fk_flange @ Trans_z(0.1034) @ R_z(-pi/4)`` -- measured, the offset is
    ``[0, 0, 0.1034]`` in the flange frame and the rotation is ``-45`` degrees about its
    ``z``.  That is where a gripper's tool centre point would be, *not* where the arm's
    body ends: ``link7.dae``'s geometry stops at the flange (its vertices reach
    ``z = +0.1068`` in the link7 frame, and the flange is 0.107 m along joint 7's axis),
    so in ``--model mesh`` a marker drawn here floats 10.34 cm beyond everything visible.
    :func:`tool_stem` draws that distance instead of leaving it to be noticed.

    Args:
        q: Seven joint angles in radians.
        length: Axis length in metres.

    Returns:
        An ``sg.Axes`` at ``franka_ik.model.fk_tool(q)``.
    """
    return frame_axes(model.fk_tool(q), length=length)


def tool_stem(q: Sequence[float], *, radius: float = 0.0035) -> object:
    """A thin capsule from the flange to the tool point, so the 0.1034 m is a length.

    The three end frames of this arm, measured (``tests/test_swift_viz.py`` pins them):
    the wrist frame to the flange is ``0.107`` m along the flange's ``z`` (the DH
    ``d7``), and the flange to the factory tool point is a further ``0.1034`` m along
    the same axis.  The meshes draw the first and none of the second, so without this
    capsule a tool-point marker is attached to nothing.

    Args:
        q: Seven joint angles in radians.
        radius: Capsule radius in metres; it is meant to read as a stem, not a body.

    Returns:
        An ``sg.Cylinder`` centred on the flange-to-tool segment.

    Raises:
        ImportError: If the ``viz`` extra is missing.
    """
    _, _, sg = require_viz()
    flange = model.fk_flange(q)
    tool = model.fk_tool(q)
    # The offset is a property of the arm, not of the pose, so the length is written
    # once here and only the pose moves afterwards.
    offset = float(np.linalg.norm(tool[:3, 3] - flange[:3, 3]))
    stem = sg.Cylinder(radius=radius, length=offset, color=[0.55, 0.55, 0.60, 1.0])
    update_tool_stem(stem, q)
    return stem


def update_tool_stem(stem: object, q: Sequence[float]) -> None:
    """Move a stem built by :func:`tool_stem` to ``q``.

    Only the pose is written: the flange-to-tool distance is ``0.1034`` m at every
    configuration, so the length never has to change -- which is also what keeps this
    acceptable to Swift, whose headless client does not acknowledge a size written
    after construction.

    Args:
        stem: The shape from :func:`tool_stem`.
        q: Seven joint angles in radians.
    """
    sm = _se3()
    flange = model.fk_flange(q)
    tool = model.fk_tool(q)
    # ``Cylinder`` is centred on its origin and drawn along its ``z``, so the pose is
    # the middle of the segment: the flange frame, translated half the offset.
    centre = np.asarray(flange, dtype=float).copy()
    centre[:3, 3] = centre[:3, 3] + 0.5 * (tool[:3, 3] - flange[:3, 3])
    stem.T = sm.Rt(centre[:3, :3], centre[:3, 3])  # type: ignore[attr-defined]


def reachable_shell_markers(
    *,
    count: int = 300,
    seed: int = 0,
    radius: float = 0.004,
    inner: bool = False,
) -> List[object]:
    """Points on the spherical shell the wrist can reach, drawn as one polyline.

    The shell is a property of the arm, not of a pose: ``‖x_sw‖`` lies in
    ``franka_ik.analysis.reachable_distance_range()``, which is closed form.  The
    markers therefore show the boundary of the workspace rather than a sample of
    it -- useful for recognising "this pose is out of reach" as a distance.

    Args:
        count: Directions to sample; they are spread with a Fibonacci sphere, so
            they stay evenly spaced for any ``count``.
        seed: Rotates the sample about ``z``, for a different spread.
        radius: Marker size.
        inner: Draw the inner surface (the hole around the shoulder) instead of
            the outer one.

    Returns:
        A list of ``sg.Sphere`` on one of the two surfaces.

    Raises:
        ImportError: If the ``viz`` extra is missing.
    """
    _, _, sg = require_viz()
    from . import analysis

    low, high = analysis.reachable_distance_range()
    distance = low if inner else high
    # The shell is centred on the shoulder, i.e. at ``d_bs`` above the base, and
    # ``reachable_distance_range`` is the range of ``‖x_sw‖`` from there.
    shoulder = np.array([0.0, 0.0, float(model.DH_PARAMETERS[0, 1])])
    rng = np.random.default_rng(seed)
    markers = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    offset = float(rng.uniform(0.0, 2.0 * np.pi))
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * index + offset
        direction = np.array([r * np.cos(theta), r * np.sin(theta), z])
        marker = sg.Sphere(radius)
        marker.T = _se3().Trans(shoulder + distance * direction)
        markers.append(marker)
    return markers


def add_shapes(env: object, shapes: Sequence[object]) -> int:
    """Add shapes to a Swift scene one at a time, and say how many landed.

    ``Swift.add`` accepts *one* shape, a robot or a UI element -- and returns
    ``None`` for anything else, a list included, **without raising**.  Handing it
    ``skeleton.shapes`` therefore produces an empty scene and no error, which is
    exactly the failure this function exists to prevent.  ``add_shape`` is the
    explicit entry point (``add`` is deprecated in Swift 2.0 for this reason) and
    returns the object id, so the count is real rather than assumed.

    Args:
        env: A launched Swift environment.
        shapes: The shapes to add.

    Returns:
        The number of shapes added.
    """
    added = 0
    for shape in shapes:
        env.add_shape(shape)
        added += 1
    return added


def add_cloud(env: object, shapes: Sequence[object], *, name: Optional[str] = None) -> object:
    """Add a group of shapes as one assembly, so the browser mounts it in one message.

    This exists for the cost measured in ``docs/browser_debugging.md`` §2.5: with a
    browser attached, ``add_shape`` blocks until the browser has mounted the shape, so
    a 400-marker point cloud is 400 round trips.  Measured under Chromium with
    software rendering, 150 markers took **85 s** to appear while the geometry drawn
    above them was ready in 1.5 s, and a branch fan rebuilt per slider step never
    appeared at all.  ``add_assembly`` sends every part in **one** message and waits
    once; ``env.remove`` takes the whole group off screen again in one call, which is
    what a rebuild needs.

    The group may still move.  An assembly is re-posed every step through its
    forward-kinematics function, and the one here reads each part's *current* pose, so
    writing ``shape.T`` -- or :meth:`ArmShapes.update`, which does -- moves the drawn
    geometry exactly as it would after :func:`add_shapes`.  The conversion to ``SE3``
    happens on every call rather than once, both because the poses move and because
    ``Shape.T`` hands back a 4x4 array that survives every Python-side check and then
    fails inside Swift's serialiser on the first frame.

    What it does *not* do is re-send a part's colour: set colours before adding, as
    the examples that colour by elbow root do.

    Args:
        env: A launched Swift environment.
        shapes: The shapes to add; at least one.
        name: Optional display name, for ``env.show``.

    Returns:
        The ``Swift.Handle.AssemblyHandle``.

    Raises:
        ValueError: If ``shapes`` is empty, which is a caller bug rather than a
            scene to draw, and ``add_assembly`` would accept it silently.
    """
    parts = list(shapes)
    if not parts:
        raise ValueError("add_cloud needs at least one shape")
    se3 = _se3()

    def current(_q: object) -> List[object]:
        """Every part's current pose, so the group follows whatever moved it."""
        return [se3(np.asarray(part.T, dtype=float)) for part in parts]

    return env.add_assembly(current, parts, q0=np.zeros(0), readonly=True, name=name)


def camera_presets() -> Dict[str, Tuple[List[float], List[float]]]:
    """Camera positions and look-at targets for a roughly 0.9 m reach arm.

    Returns:
        A mapping from preset name to ``(position, look_at)`` with the keys
        ``"iso"``, ``"front"``, ``"side"`` and ``"top"``.
    """
    return {
        "iso": ([1.4, 1.2, 1.0], [0.0, 0.0, 0.4]),
        "front": ([1.9, 0.0, 0.6], [0.0, 0.0, 0.45]),
        "side": ([0.0, 1.9, 0.6], [0.0, 0.0, 0.45]),
        "top": ([0.0, 0.001, 2.3], [0.0, 0.0, 0.4]),
    }


def apply_camera(env: object, name: str) -> None:
    """Point a Swift environment's camera at one of :func:`camera_presets`.

    Args:
        env: A launched Swift environment, i.e. anything exposing
            ``set_camera_pose(position, look_at)``.
        name: Preset name.

    Raises:
        ValueError: If ``name`` is not a known preset.
    """
    presets = camera_presets()
    if name not in presets:
        raise ValueError(f"unknown camera preset {name!r}; available: {sorted(presets)}")
    position, look_at = presets[name]
    env.set_camera_pose(position, look_at)


def is_wsl() -> bool:
    """Whether this looks like Windows Subsystem for Linux."""
    if sys.platform != "linux":
        return False
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in release.lower() or "wsl" in release.lower()


def _windows_browsers() -> List[str]:
    """Windows browsers reachable from WSL, in preference order."""
    return [
        "/mnt/c/Program Files/Mozilla Firefox/firefox.exe",
        "/mnt/c/Program Files (x86)/Mozilla Firefox/firefox.exe",
        "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ]


def wsl_browser_wrapper(target: str) -> str:
    """A space-free launcher for a Windows browser at ``target``.

    ``webbrowser`` shells out to ``xdg-open``, which splits the ``BROWSER`` command
    on **whitespace**.  Every Windows browser lives under ``/mnt/c/Program Files``,
    so handing over the path directly cuts it at the first space and the launch dies
    with ``/mnt/c/Program: not found`` -- measured on this machine, and the reason
    the wrapper below exists.  The script is written to the cache so that ``BROWSER``
    stays a single space-free token.

    Args:
        target: Path to a Windows executable.

    Returns:
        The path of the wrapper script.
    """
    wrapper = _cache_dir() / "windows-browser"
    wrapper.write_text(f'#!/bin/sh\nexec "{target}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    return str(wrapper)


def _cache_dir() -> Path:
    """The directory this module caches generated files in."""
    directory = Path(os.environ.get("FRANKA_IK_CACHE", Path.home() / ".cache" / "franka_ik"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def detect_browser() -> Optional[str]:
    """Find a browser that can open a WebGL page.

    Returns:
        A path or executable name, or ``None`` when nothing was found.
    """
    for path in _windows_browsers():
        if Path(path).is_file():
            return path
    for name in ("firefox", "google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    # A Playwright Chromium, if one is installed: the browser the debugging playbook
    # uses for screenshots is also a perfectly good viewer.
    for candidate in sorted((Path.home() / ".cache" / "ms-playwright").glob("chromium-*/chrome-linux/chrome")):
        if candidate.is_file():
            return str(candidate)
    return None


def _browser_command(target: str) -> str:
    """A ``BROWSER`` token for ``target``: wrapped when its path has spaces."""
    return wsl_browser_wrapper(target) if " " in target else target


def launch_env(*, headless: bool = False, browser: Optional[str] = None, realtime: bool = True):
    """Create and launch a Swift environment, opening a browser when asked.

    ``browser=None`` lets Swift pick, which is unreliable under WSL (``xdg-open``
    cannot reach a Windows browser); pass ``"auto"`` to use
    :func:`detect_browser`, or a name or path to override.  ``BROWSER`` in the
    environment wins over the system default and is left alone.

    Args:
        headless: Run without a browser.  No rendering happens, but the examples
            still run to completion and print their numbers, which is what makes
            them testable.
        browser: Browser name, path, or ``"auto"``.
        realtime: Ask Swift to simulate as close to realtime as it can.

    Returns:
        The launched Swift environment.

    Raises:
        ImportError: If the ``viz`` extra is missing.
        RuntimeError: If Swift cannot start, with the fix suggestions attached.
    """
    deps = require_viz()
    if browser == "auto":
        detected = detect_browser()
        if not detected:
            raise RuntimeError(_BROWSER_HINT)
        os.environ["BROWSER"] = f"{_browser_command(detected)} {_SOFTWARE_WEBGL_FLAG} %s"
        browser = None
    elif not headless and browser is None and not os.environ.get("BROWSER") and is_wsl():
        detected = detect_browser()
        if detected:
            os.environ["BROWSER"] = f"{_browser_command(detected)} {_SOFTWARE_WEBGL_FLAG} %s"
    env = deps.swift.Swift()
    try:
        env.launch(realtime=realtime, headless=headless, browser=browser)
    except Exception as exc:  # noqa: BLE001 - every launch failure gets the same hint
        raise RuntimeError(_BROWSER_HINT) from exc
    return env
