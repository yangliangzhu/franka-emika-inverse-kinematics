"""Matplotlib figures for the Franka analytical IK.

The figures are built around the one question this repository is about: *which
configurations reach this pose, and does the solver find all of them?*  So the
central drawing is not a single arm but several arms superimposed, one per
branch, sharing a tool frame and differing in where the elbow and wrist sit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np

from . import model
from .analysis import CoverageReport, SolutionCountReport
from .geometry import PAPER_GEOMETRY, EquivalentGeometry, link_offset, q4_coefficients
from .solver import solve

__all__ = [
    "import_mplot3d",
    "use_headless_backend",
    "save",
    "show",
    "plot_arm_3d",
    "plot_branches_3d",
    "plot_elbow_quadratic",
    "plot_coverage_comparison",
    "plot_solution_histogram",
    "plot_joint_trajectories",
    "BRANCH_COLORS",
]

#: One colour per branch, in the order :func:`franka_ik.solver.branch_solutions`
#: returns them, so a reader can match a curve to an arm by colour.
BRANCH_COLORS: Tuple[str, ...] = (
    "#08519c",
    "#6baed6",
    "#238b45",
    "#74c476",
    "#cb181d",
    "#fb6a4a",
    "#6a51a3",
    "#9e9ac8",
)


def use_headless_backend() -> None:
    """Select the non-interactive ``Agg`` backend."""
    matplotlib.use("Agg", force=True)


def import_mplot3d():
    """Import ``mpl_toolkits.mplot3d``, tolerating a split matplotlib install.

    ``projection="3d"`` only works once that module has been imported.  When two
    matplotlib installations coexist -- a distribution package under
    ``/usr/lib/python3/dist-packages`` and a pip package under ``~/.local``, the
    usual situation on a ROS machine -- the older ``mpl_toolkits`` shadows the
    newer one and the import fails with ``cannot import name 'docstring'``.  This
    helper first tries the ordinary import and, if it fails, loads
    ``mpl_toolkits.mplot3d`` from the directory belonging to the *imported*
    matplotlib, which is the version that matches.

    Returns:
        The ``mpl_toolkits.mplot3d`` module.

    Raises:
        ImportError: If neither the normal nor the fallback import works.
    """
    try:
        import mpl_toolkits.mplot3d as mplot3d

        return mplot3d
    except ImportError as first_error:
        import importlib.util
        import sys

        candidate = (
            Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits" / "mplot3d"
        )
        if not (candidate / "__init__.py").is_file():
            raise ImportError(
                "matplotlib's 3D projection is unavailable and no matching "
                f"mpl_toolkits.mplot3d was found next to {matplotlib.__file__}"
            ) from first_error

        import mpl_toolkits  # noqa: F401  make sure the parent package exists

        spec = importlib.util.spec_from_file_location(
            "mpl_toolkits.mplot3d",
            candidate / "__init__.py",
            submodule_search_locations=[str(candidate)],
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"cannot load mpl_toolkits.mplot3d from {candidate}") from first_error
        module = importlib.util.module_from_spec(spec)
        sys.modules["mpl_toolkits.mplot3d"] = module
        spec.loader.exec_module(module)
        mplot3d = module

    # matplotlib.projections decides once, at its own import time, whether the 3d
    # projection exists.  If it was imported before us, register it now.
    import sys as _sys

    projections = _sys.modules.get("matplotlib.projections")
    if projections is not None:
        try:
            registered = projections.projection_registry.get_projection_names()
        except Exception:  # pragma: no cover - defensive
            registered = ["3d"]
        if "3d" not in registered:
            projections.register_projection(mplot3d.Axes3D)
    return mplot3d


def save(figure, path: Union[str, Path], dpi: int = 140, close: bool = True) -> Path:
    """Save a figure, creating parent directories.

    Args:
        figure: The matplotlib figure.
        path: Destination path.
        dpi: Resolution for raster formats.
        close: Close the figure afterwards.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return path


def show() -> None:
    """Show pending figures (no-op on the ``Agg`` backend)."""
    import matplotlib.pyplot as plt

    plt.show()


# --------------------------------------------------------------------------- #
# 3D drawings
# --------------------------------------------------------------------------- #
def _arm_points(q: Sequence[float]) -> np.ndarray:
    """Joint origins of a configuration, base first, plus the flange."""
    frames = model.forward_kinematics(q)
    points = [np.zeros(3)]
    points.extend(frames[index][:3, 3] for index in range(model.NUM_JOINTS))
    return np.array(points)


def plot_arm_3d(
    ax,
    q: Sequence[float],
    color: str = "#08519c",
    label: Optional[str] = None,
    alpha: float = 1.0,
    linewidth: float = 2.0,
):
    """Draw one arm configuration into an existing 3D axis.

    Args:
        ax: A 3D axis.
        q: Seven joint angles in radians.
        color: Line colour.
        label: Legend label.
        alpha: Opacity, so that overlapping configurations stay readable.
        linewidth: Width of the arm line.

    Returns:
        The axis.
    """
    points = _arm_points(q)
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        "-o",
        color=color,
        lw=linewidth,
        ms=3.5,
        alpha=alpha,
        label=label,
    )
    return ax


def _set_scene_bounds(ax, points: np.ndarray) -> None:
    """Give the 3D scene a cubic box, so the arm is not distorted."""
    low = points.min(axis=0)
    high = points.max(axis=0)
    centre = 0.5 * (low + high)
    radius = 0.55 * float(np.max(high - low)) + 0.12
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(max(centre[2] - radius, -0.1), centre[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])


def plot_branches_3d(
    pose: np.ndarray,
    q7: float,
    target: Optional[Sequence[float]] = None,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    figure_size: Tuple[float, float] = (7.5, 7.0),
):
    """Draw every branch that reaches the pose, on top of each other.

    This is the picture of the headline finding: the tool frame is shared, the
    elbow and wrist differ, and the configuration the pose was generated from is
    highlighted when it is known.

    Args:
        pose: 4x4 homogeneous flange pose.
        q7: Joint 7 value in radians.
        target: Configuration the pose came from, highlighted in red when given.
        geometry: Geometric parameters.
        figure_size: Figure size in inches.

    Returns:
        ``(figure, ax, solutions)``.
    """
    import_mplot3d()
    import matplotlib.pyplot as plt

    solutions = solve(pose, q7, geometry=geometry, within_limits_only=True)
    figure = plt.figure(figsize=figure_size)
    ax = figure.add_subplot(111, projection="3d")

    all_points = [_arm_points(solution.q) for solution in solutions]
    for index, solution in enumerate(solutions):
        recovered = False
        if target is not None:
            delta = solution.q - np.asarray(target, dtype=float)
            recovered = bool(
                np.all(np.abs(np.arctan2(np.sin(delta), np.cos(delta))) < 1e-6)
            )
        plot_arm_3d(
            ax,
            solution.q,
            color=BRANCH_COLORS[index % len(BRANCH_COLORS)],
            label=f"{solution.label}" + ("  <- target" if recovered else ""),
            alpha=0.9 if recovered else 0.55,
            linewidth=3.0 if recovered else 1.8,
        )

    if target is not None:
        target_points = _arm_points(target)
        ax.plot(
            target_points[:, 0],
            target_points[:, 1],
            target_points[:, 2],
            "--",
            color="#cb181d",
            lw=1.2,
            alpha=0.8,
            label="generating configuration",
        )
        all_points.append(target_points)

    if all_points:
        _set_scene_bounds(ax, np.vstack(all_points))

    tip = pose[:3, 3]
    ax.plot([tip[0]], [tip[1]], [tip[2]], marker="*", ms=15, color="#000000", label="target pose")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(
        f"every branch that reaches the pose at joint 7 = {math.degrees(q7):.1f} deg\n"
        f"{len(solutions)} distinct configuration(s), all with the same flange pose"
    )
    ax.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure, ax, solutions


def plot_elbow_quadratic(
    distance: float,
    geometry: EquivalentGeometry = PAPER_GEOMETRY,
    figure_size: Tuple[float, float] = (11.0, 4.2),
):
    """The quadratic of STEP2, with both roots marked.

    Args:
        distance: ``||x_sw||`` in metres.
        geometry: Geometric parameters.
        figure_size: Figure size in inches.

    Returns:
        ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    from .geometry import q4_discriminant, q4_roots

    a2, a1, a0 = q4_coefficients(distance**2, geometry)
    roots = q4_roots(distance**2, geometry)

    figure, axes = plt.subplots(1, 2, figsize=figure_size)

    limit = 6.0
    scale = max(abs(a2), abs(a1), abs(a0), 1e-6)
    xs = np.linspace(-limit, limit, 600)
    values = (a2 * xs**2 - a1 * xs + a0) / scale
    axes[0].axhline(0.0, color="#888888", lw=1.0)
    axes[0].plot(xs, values, color="#08519c", lw=1.8, label=r"$a_2x^2 - a_1x + a_0$")
    for index, theta4 in enumerate(roots):
        root = math.tan(theta4 / 2.0)
        axes[0].plot([root], [0.0], marker="o", ms=9,
                     color=["#cb181d", "#238b45"][index % 2],
                     label=rf"$\theta_4 = {math.degrees(theta4):.2f}$ deg")
    if not roots:
        axes[0].text(0.0, 0.0, "no real root\n(out of reach)", ha="center", color="#cb181d")
    axes[0].set_xlabel(r"$x = \tan(\theta_4/2)$")
    axes[0].set_ylabel("normalised value")
    axes[0].set_title(
        rf"elbow quadratic at $\|x_{{sw}}\| = {distance:.3f}$ m"
        f"\ndiscriminant {q4_discriminant(distance**2, geometry):+.3e}"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # The equivalent links grow with theta4, which is what makes this a family
    # of S-R-S arms rather than one.
    angles = np.linspace(-math.pi + 1e-3, math.pi - 1e-3, 721)
    deltas = np.array([link_offset(value, geometry) for value in angles])
    axes[1].plot(np.degrees(angles), 1000.0 * deltas, color="#6a51a3", lw=1.8)
    axes[1].axhline(0.0, color="#888888", lw=1.0)
    for index, theta4 in enumerate(roots):
        axes[1].plot(
            [math.degrees(theta4)],
            [1000.0 * link_offset(theta4, geometry)],
            marker="o",
            ms=9,
            color=["#cb181d", "#238b45"][index % 2],
        )
    axes[1].set_xlabel(r"$\theta_4$ (deg)")
    axes[1].set_ylabel(r"$\delta$ (mm)")
    axes[1].set_title(
        r"link correction $\delta = -\tan(\theta_4/2)\cdot$bias"
        "\nthe reduction is a one-parameter family of S-R-S arms"
    )
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure, axes


# --------------------------------------------------------------------------- #
# 2D drawings
# --------------------------------------------------------------------------- #
def plot_coverage_comparison(
    report: CoverageReport, figure_size: Tuple[float, float] = (7.0, 4.6)
):
    """Bar chart of the four-branch subset against the full solver.

    Args:
        report: Result of :func:`franka_ik.analysis.coverage_study`.
        figure_size: Figure size in inches.

    Returns:
        ``(figure, ax)``.
    """
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=figure_size)
    names = ["published\nfour branches", "full\neight branches"]
    values = [100.0 * report.published_rate, 100.0 * report.full_rate]
    bars = ax.bar(names, values, color=["#fdae6b", "#08519c"], width=0.55)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.1f} %",
            (bar.get_x() + bar.get_width() / 2.0, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontweight="bold",
        )
    ax.set_ylim(0.0, 108.0)
    ax.set_ylabel("target configuration recovered (%)")
    ax.set_title(
        f"does the solver find the configuration the pose came from?\n"
        f"{report.samples} random configurations"
    )
    ax.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure, ax


def plot_solution_histogram(
    report: SolutionCountReport, figure_size: Tuple[float, float] = (7.0, 4.6)
):
    """How many configurations actually reach a given pose.

    Args:
        report: Result of :func:`franka_ik.analysis.solution_count_study`.
        figure_size: Figure size in inches.

    Returns:
        ``(figure, ax)``.
    """
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=figure_size)
    counts = sorted(report.histogram)
    values = [report.histogram[count] for count in counts]
    ax.bar([str(count) for count in counts], values, color="#238b45", width=0.7)
    for index, value in enumerate(values):
        ax.annotate(
            str(value),
            (index, value),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("distinct in-limit solutions for the same pose and joint 7")
    ax.set_ylabel("number of poses")
    ax.set_title(
        f"the answer is not four: {report.samples} random poses, mean {report.mean:.2f}"
    )
    ax.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure, ax


def plot_joint_trajectories(
    times: np.ndarray,
    naive: np.ndarray,
    continuous: np.ndarray,
    figure_size: Tuple[float, float] = (12.0, 6.0),
):
    """Joint angles of a naive branch choice against a continuity-aware one.

    Args:
        times: Sample times, seconds.
        naive: Joint angles when the first solution is taken at every sample,
            shape ``(n, 7)``, radians.
        continuous: The same with :func:`franka_ik.solver.solve_closest`,
            shape ``(n, 7)``.
        figure_size: Figure size in inches.

    Returns:
        ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True)
    for index in range(model.NUM_JOINTS):
        axes[0].plot(times, np.degrees(naive[:, index]), lw=1.1, label=f"q{index + 1}")
        axes[1].plot(times, np.degrees(continuous[:, index]), lw=1.1, label=f"q{index + 1}")
    axes[0].set_title("first solution returned by the solver (branches jump)")
    axes[1].set_title("solution closest to the previous command (continuous)")
    for axis in axes:
        axis.set_ylabel("joint (deg)")
        axis.grid(alpha=0.3)
        axis.legend(ncol=4, fontsize=8)
    axes[1].set_xlabel("time (s)")
    figure.tight_layout()
    return figure, axes
