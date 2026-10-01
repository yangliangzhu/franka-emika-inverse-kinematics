"""STEP 0 -- the modified-DH model everything else is written against.

Prints the modified (Craig) DH table, the difference between the flange and tool
frames, the real joint limits and a manipulability sweep over each joint; the
figure draws the arm at four configurations and that sweep.

Points to look for: row 8 (``d = 0.107 m``) closes the flange frame the IK solves
for while row 9 is only reached by the tool frame; ``fk_tool`` is ``fk_flange``
translated by ``0.1034 m`` along the flange ``z`` and rotated by the factory
``RotZ(-45 deg)``; joints 4 and 6 are one-sided; manipulability is invariant to
joint 1 and peaks on the elbow.  The 3D panel goes through
:func:`franka_ik.viz.import_mplot3d`, which is needed on any machine where a
distribution ``mpl_toolkits`` shadows the pip one.

Run from the repository root::

    MPLBACKEND=Agg python3 examples/01_forward_kinematics.py --save-dir /tmp/fk
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import (  # noqa: E402
    DH_PARAMETERS,
    NUM_JOINTS,
    fk_flange,
    fk_tool,
    forward_kinematics,
    lower_limits,
    manipulability,
    upper_limits,
)

# ``joint_frames`` and ``TOOL_ROTATION`` live in ``franka_ik.model`` but are not
# re-exported by the package ``__init__``, so they come from the module itself.
from franka_ik.model import TOOL_ROTATION  # noqa: E402

LOGGER = logging.getLogger("examples.01_forward_kinematics")

#: A configuration inside every joint range with the elbow and wrist bent, so the
#: Jacobian is not singular.
HOME_DEG = np.array([0.0, -30.0, 0.0, -60.0, 0.0, 90.0, 0.0])


def import_pyplot(headless: bool) -> Any:
    """Import ``matplotlib.pyplot``, selecting a backend before it is imported.

    Args:
        headless: Use ``Agg`` so that saving works without a display.

    Returns:
        The ``matplotlib.pyplot`` module.
    """
    if headless:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finish(figure: Any, save_dir: Optional[Path], stem: str) -> Optional[Path]:
    """Save the figure when ``--save-dir`` was given, otherwise show it.

    Args:
        figure: The figure to render.
        save_dir: Destination directory, or ``None`` for an interactive window.
        stem: Figure file name without extension.

    Returns:
        The written path, or ``None`` when the figure was shown instead.
    """
    plt = import_pyplot(save_dir is not None)
    if save_dir is None:
        plt.show()
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{stem}.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    LOGGER.info("wrote %s", path)
    return path


def scan_manipulability(base: np.ndarray, joint: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep one joint over its whole range, holding the others.

    Args:
        base: Reference configuration in radians.
        joint: Zero-based index of the joint to sweep.
        samples: Number of samples.

    Returns:
        ``(angles, measures)``, both of length ``samples``.
    """
    span = np.linspace(lower_limits()[joint], upper_limits()[joint], samples)
    measures = np.empty(samples)
    for index, angle in enumerate(span):
        configuration = base.copy()
        configuration[joint] = angle
        measures[index] = manipulability(configuration)
    return (span, measures)


def build_figure(plt: Any, seed: int, base: np.ndarray, span: np.ndarray, measures: np.ndarray) -> Any:
    """Draw the arm at four configurations and the manipulability sweep.

    Args:
        plt: The imported ``matplotlib.pyplot`` module.
        seed: RNG seed for the three extra configurations.
        base: Reference configuration in radians, drawn with a thick line.
        span: Angles of the detailed sweep, in radians.
        measures: Manipulability values of that sweep.

    Returns:
        The matplotlib figure.
    """
    # Imported here rather than at module level: ``franka_ik.viz`` needs
    # matplotlib, and the backend has to be chosen before pyplot is loaded.
    from franka_ik import viz

    viz.import_mplot3d()  # a distribution mpl_toolkits can shadow the pip one
    figure = plt.figure(figsize=(11.0, 4.6))
    arm = figure.add_subplot(1, 2, 1, projection="3d")
    curves = figure.add_subplot(1, 2, 2)
    rng = np.random.default_rng(seed)
    for index in range(4):
        configuration = base if index == 0 else rng.uniform(lower_limits(), upper_limits())
        viz.plot_arm_3d(
            arm,
            configuration,
            color=viz.BRANCH_COLORS[2 * index],  # every other entry: the dark set
            label="home" if index == 0 else f"random {index}",
            linewidth=2.5 if index == 0 else 1.5,
        )
    arm.set_xlabel("x [m]")
    arm.set_ylabel("y [m]")
    arm.set_zlabel("z [m]")
    arm.set_title("joint frames (7 joints + flange)")
    arm.set_xlim(-0.8, 0.8)
    arm.set_ylim(-0.8, 0.8)
    arm.set_zlim(0.0, 1.4)
    arm.set_box_aspect((1.0, 1.0, 1.4))
    arm.legend(fontsize=8)

    base_span, base_measures = scan_manipulability(base, 0, span.size)
    curves.plot(np.degrees(span), measures, label="joint 4 (elbow)")
    curves.plot(np.degrees(base_span), base_measures, label="joint 1 (base rotation)")
    curves.set_xlabel("joint angle [deg]")
    curves.set_ylabel("manipulability")
    curves.set_title("manipulability versus one joint")
    curves.grid(alpha=0.3)
    curves.legend(fontsize=8)
    figure.tight_layout()
    return figure


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the command line (see ``--help``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=None, help="write the figure here")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the drawn configurations")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO or DEBUG")
    parser.add_argument("--joint", type=int, default=4, help="joint (1-based) scanned in detail")
    parser.add_argument("--samples", type=int, default=181, help="samples per sweep")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Print the model and draw the figure.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    if not 1 <= args.joint <= NUM_JOINTS:
        raise SystemExit(f"--joint must be in 1..{NUM_JOINTS}")
    base = np.radians(HOME_DEG)
    flange, tool = fk_flange(base), fk_tool(base)

    print("STEP 0 -- the modified-DH model of the Franka Emika Panda")
    print(f"home configuration: {HOME_DEG} deg\n")
    print("Modified DH table (RotX(alpha) TransX(a) TransZ(d) RotZ(theta)):")
    print(f"  {'row':>3} {'a [m]':>9} {'d [m]':>9} {'alpha [deg]':>12} {'theta off [deg]':>16}")
    for row in range(DH_PARAMETERS.shape[0]):
        a, d, alpha, offset = DH_PARAMETERS[row]
        role = f"joint {row + 1}" if row < NUM_JOINTS else "rigid link"
        print(
            f"  {row + 1:>3} {a:>9.4f} {d:>9.4f} {np.degrees(alpha):>12.1f} "
            f"{np.degrees(offset):>16.1f}   {role}"
        )

    relative = flange[:3, :3].T @ tool[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    print("\nFlange frame versus tool frame:")
    print(f"  p_flange                 = {np.round(flange[:3, 3], 6)} m")
    print(
        f"  |p_tool - p_flange|      = {np.linalg.norm(tool[:3, 3] - flange[:3, 3]):.6f} m"
        "   (DH row 9, d = 0.1034 m)"
    )
    print(
        f"  angle(R_flange.T R_tool) = {angle:.6f} deg"
        f"   (factory RotZ({np.degrees(TOOL_ROTATION):.0f} deg))"
    )

    lower, upper = np.degrees(lower_limits()), np.degrees(upper_limits())
    print("\nJoint limits as the real controller reports them:")
    print(f"  {'joint':>5} {'lower [deg]':>12} {'upper [deg]':>12} {'width [deg]':>12}")
    for index in range(NUM_JOINTS):
        one_sided = "  one-sided" if lower[index] >= 0.0 or upper[index] <= 0.0 else ""
        print(
            f"  {index + 1:>5} {lower[index]:>12.1f} {upper[index]:>12.1f} "
            f"{upper[index] - lower[index]:>12.1f}{one_sided}"
        )

    print("\nYoshikawa manipulability sqrt(det(J J^T)) swept over each joint:")
    print(f"  {'joint':>5} {'min':>12} {'max':>12} {'spread':>12} {'argmax [deg]':>14}")
    detailed = None
    for index in range(NUM_JOINTS):
        span, measures = scan_manipulability(base, index, args.samples)
        print(
            f"  {index + 1:>5} {measures.min():>12.6f} {measures.max():>12.6f} "
            f"{measures.max() - measures.min():>12.6f} "
            f"{np.degrees(span[int(measures.argmax())]):>14.2f}"
        )
        if index == args.joint - 1:
            detailed = (span, measures)
    assert detailed is not None
    span, measures = detailed
    print(
        f"  joint {args.joint} in detail: minimum {measures.min():.6f} at "
        f"{np.degrees(span[int(measures.argmin())]):.2f} deg, maximum {measures.max():.6f} "
        f"at {np.degrees(span[int(measures.argmax())]):.2f} deg"
    )
    print(f"\nforward_kinematics returns {forward_kinematics(base).shape} (9 frames of 4x4)")
    print(f"  frame 8 (flange) origin = {np.round(flange[:3, 3], 6)} m")

    plt = import_pyplot(args.save_dir is not None)
    finish(build_figure(plt, args.seed, base, span, measures), args.save_dir, "01_forward_kinematics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
