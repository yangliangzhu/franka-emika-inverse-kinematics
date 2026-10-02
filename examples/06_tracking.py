"""Tracking a Cartesian path: why the branch you pick matters.

The eight branches are *discrete*.  A controller that simply takes the first
solution the solver returns gets a correct answer at every step and still jumps
between configurations: at some point the first branch changes and the arm
teleports by the angle between the two.  :func:`franka_ik.solver.solve_closest`
avoids that by returning the solution nearest the previous command.

The example follows a short path with joint 7 prescribed at every step -- which is
what parameterising the redundancy by joint 7 buys the caller -- twice: once with
``solve_closest`` and once with ``solve(...)[0]``.  Both are correct at every step
(the residual stays around 1e-16) and they differ only in how far the joints move,
which is exactly what a controller cares about.

The path is built by interpolating in joint space between two configurations
inside the limits and pushing the result through
:func:`franka_ik.model.fk_flange`, so every pose along it is reachable by
construction and the comparison is not an artefact of an unreachable target.

Run from the repository root::

    MPLBACKEND=Agg python3 examples/06_tracking.py --save-dir /tmp/tracking
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import (  # noqa: E402
    fk_flange,
    lower_limits,
    solve,
    solve_closest,
    upper_limits,
)

LOGGER = logging.getLogger("examples.06_tracking")

#: Start and end of the path in degrees.  Both are inside the real limits, and
#: joint 2 stays negative throughout so the path never crosses the ``q2 = 0`` arm
#: configuration, where the branch structure degenerates.
START_DEG = np.array([0.0, -55.0, 0.0, -80.0, 0.0, 90.0, 0.0])
END_DEG = np.array([45.0, -20.0, -35.0, -120.0, 30.0, 140.0, 60.0])


@dataclass
class TrackingRun:
    """What one control rule did along the path.

    Attributes:
        name: Human-readable name of the rule.
        q: One configuration per step, ``None`` where there was no solution.
        labels: Branch label per step, ``None`` where there was no solution.
        residuals: Pose residual of the commanded configuration, per step.
    """

    name: str
    q: List[Optional[np.ndarray]] = field(default_factory=list)
    labels: List[Optional[str]] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)

    def gaps(self) -> int:
        """Number of steps with no solution at all."""
        return sum(1 for item in self.q if item is None)

    def joint_steps(self) -> np.ndarray:
        """Largest joint-wise change between consecutive commands, in degrees."""
        result = []
        for previous, current in zip(self.q[:-1], self.q[1:], strict=True):
            if previous is None or current is None:
                result.append(math.nan)
                continue
            delta = current - previous
            result.append(
                float(np.degrees(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))))
            )
        return np.asarray(result)

    def switches(self) -> int:
        """How often the branch label changed along the path."""
        return sum(
            1 for a, b in zip(self.labels[:-1], self.labels[1:], strict=True) if a and b and a != b
        )


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


def make_path(steps: int) -> List[np.ndarray]:
    """Interpolate between the two end configurations.

    Args:
        steps: Number of samples along the path.

    Returns:
        The configurations, start first.

    Raises:
        ValueError: If an end configuration is outside the joint limits.
    """
    start, end = np.radians(START_DEG), np.radians(END_DEG)
    lower, upper = lower_limits(), upper_limits()
    for name, configuration in (("start", start), ("end", end)):
        if not bool(np.all((configuration >= lower) & (configuration <= upper))):
            raise ValueError(f"the {name} configuration is outside the joint limits")
    return [(1.0 - t) * start + t * end for t in np.linspace(0.0, 1.0, steps)]


def track(configurations: List[np.ndarray], use_closest: bool) -> TrackingRun:
    """Follow the path with one of the two control rules.

    Args:
        configurations: The interpolated configurations.
        use_closest: Use ``solve_closest`` against the previous command, rather
            than the first solution the solver returns.

    Returns:
        The run.
    """
    run = TrackingRun(
        name="solve_closest (previous command as reference)"
        if use_closest
        else "solve(...)[0] (first solution)"
    )
    previous = configurations[0].copy()
    for configuration in configurations:
        pose = fk_flange(configuration)
        q7 = float(configuration[6])
        if use_closest:
            solution = solve_closest(pose, q7, previous, within_limits_only=True)
        else:
            found = solve(pose, q7, within_limits_only=True)
            solution = found[0] if found else None
        if solution is None:
            LOGGER.warning("no solution at q7 = %.3f deg", np.degrees(q7))
            run.q.append(None)
            run.labels.append(None)
            run.residuals.append(math.inf)
            continue
        run.q.append(solution.q.copy())
        run.labels.append(solution.label)
        run.residuals.append(float(np.max(np.abs(fk_flange(solution.q) - pose))))
        previous = solution.q.copy()
    return run


def print_run(run: TrackingRun, steps: int) -> None:
    """Print what one run did.

    Args:
        run: The run.
        steps: Number of steps along the path.
    """
    changes = run.joint_steps()
    finite = changes[np.isfinite(changes)]
    print(f"  {run.name}")
    print(f"    steps                    : {steps} ({len(run.q) - run.gaps()} solved)")
    print(f"    steps without a solution : {run.gaps()}")
    print(f"    branch label changes     : {run.switches()}  (starts at {run.labels[0]})")
    print(f"    largest joint step       : {finite.max():.3f} deg")
    print(f"    total joint travel       : {finite.sum():.3f} deg")
    print(f"    worst pose residual      : {max(run.residuals):.3e}")


def build_figure(plt: Any, nearest: TrackingRun, first: TrackingRun) -> Any:
    """Draw the joint trajectories of both runs and their step sizes.

    Args:
        plt: The imported ``matplotlib.pyplot`` module.
        nearest: The run that used ``solve_closest``.
        first: The run that took the first solution.

    Returns:
        The matplotlib figure.
    """
    figure, panels = plt.subplots(2, 2, figsize=(12.0, 7.0))
    for panel, run in zip(panels[0], (nearest, first), strict=True):
        trajectory = np.degrees(np.asarray([item for item in run.q if item is not None]))
        for joint in range(trajectory.shape[1]):
            panel.plot(trajectory[:, joint], label=f"q{joint + 1}")
        for step in range(1, len(run.labels)):
            if run.labels[step] != run.labels[step - 1]:
                panel.axvline(step, color="red", linestyle=":", alpha=0.6)
        panel.set_xlabel("step")
        panel.set_ylabel("joint angle [deg]")
        panel.set_title(run.name, fontsize=9)
        panel.grid(alpha=0.3)
    panels[0][0].legend(fontsize=7, ncol=2)

    for run, colour in ((nearest, "tab:blue"), (first, "tab:red")):
        changes = run.joint_steps()
        panels[1][0].semilogy(range(1, len(changes) + 1), changes, color=colour, label=run.name)
        panels[1][1].semilogy(
            range(len(run.residuals)), run.residuals, color=colour, label=run.name
        )
    panels[1][0].set_xlabel("step")
    panels[1][0].set_ylabel("largest joint change [deg]")
    panels[1][0].set_title("continuity: the branch jump shows up as a spike", fontsize=9)
    panels[1][0].legend(fontsize=7)
    panels[1][1].set_xlabel("step")
    panels[1][1].set_ylabel("pose residual")
    panels[1][1].set_title("both rules are correct at every step", fontsize=9)
    panels[1][1].legend(fontsize=7)
    for panel in panels[1]:
        panel.grid(alpha=0.3, which="both")
    figure.tight_layout()
    return figure


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (see ``--help``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=None, help="write the figure here")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (the path is deterministic)")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO or DEBUG")
    parser.add_argument("--steps", type=int, default=61, help="samples along the path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Follow the path with both rules, print them and draw the figure.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    LOGGER.debug("seed %d is recorded for symmetry; the path is deterministic", args.seed)

    configurations = make_path(args.steps)
    positions = np.asarray([fk_flange(item)[:3, 3] for item in configurations])
    print("Tracking a Cartesian path with joint 7 prescribed")
    print(f"  start (deg)     : {START_DEG}")
    print(f"  end   (deg)     : {END_DEG}")
    print(f"  steps           : {args.steps}")
    print(
        f"  path length     : {float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))):.4f} m"
    )
    print(
        f"  joint 7 goes    : {np.degrees(configurations[0][6]):.3f} deg -> "
        f"{np.degrees(configurations[-1][6]):.3f} deg\n"
    )

    nearest, first = track(configurations, True), track(configurations, False)
    print("Two control rules, same path:")
    print_run(nearest, args.steps)
    print_run(first, args.steps)
    print("  both rules return a correct configuration at every step; only the joint motion")
    print("  differs, which is exactly what a controller cares about")

    plt = import_pyplot(args.save_dir is not None)
    finish(build_figure(plt, nearest, first), args.save_dir, "06_tracking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
