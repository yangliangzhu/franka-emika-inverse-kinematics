"""Joint limits: the wrap that the published ``limit_joints`` got wrong.

Two different things are called "wrapping" on the branch this repository grew
from, and :func:`franka_ik.solver.wrap_to_limits` keeps them apart:

* mapping an angle into the range the real robot reports, by adding the right
  multiple of ``2*pi``.  Joints 4 and 6 are **one-sided** (``[-175, -5]`` deg and
  ``[0, 214]`` deg), so this is not the usual symmetric wrap and for some angles no
  representative fits at all;
* deciding whether a solution is usable.  :func:`franka_ik.solver.solve` returns
  every candidate with a ``within_limits`` flag instead of silently projecting it.

The example walks through angles that land inside, angles one turn away from
inside and angles that cannot be represented at all, then shows the ``ValueError``
raised for a non-finite input.  That error is deliberate: the published
``limit_joints`` of ``original/ik_ca.py`` uses ``while True`` loops, and since
neither ``-181 <= nan`` nor ``-181 > nan`` holds, it subtracts 360 from ``nan``
forever.  With ``--check-original`` (the default) that hang is reproduced in a
*subprocess with a timeout*, never in this process.

Run from the repository root::

    MPLBACKEND=Agg python3 examples/04_joint_limits.py --save-dir /tmp/limits
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import lower_limits, upper_limits, wrap_to_limits  # noqa: E402

LOGGER = logging.getLogger("examples.04_joint_limits")

#: A configuration inside every joint range, used as the base vector.
HOME_DEG = np.array([0.0, -30.0, 0.0, -60.0, 0.0, 90.0, 0.0])

#: ``(joint index, angle in degrees, why it is interesting)``.
CASES: Tuple[Tuple[int, float, str], ...] = (
    (0, 200.0, "two-sided: one full turn brings it back inside"),
    (0, 350.0, "two-sided: -360 deg lands exactly on -10 deg"),
    (3, -90.0, "joint 4 is already inside its one-sided range"),
    (3, 90.0, "joint 4: +360 is 450 > -5 and -360 is -270 < -175, nothing fits"),
    (3, 170.0, "joint 4: the nearest representative, -190 deg, is still outside"),
    (5, 30.0, "joint 6 is already inside its one-sided range"),
    (5, -30.0, "joint 6: +360 would be 330 > 214, so the value stays negative"),
    (5, -100.0, "joint 6: the +360 representative, 260 deg, is still > 214"),
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


def run_cases() -> List[Dict[str, Any]]:
    """Wrap a table of hand-picked angles and print what came back.

    Returns:
        One record per case, with the numbers the figure needs.
    """
    base = np.radians(HOME_DEG)
    lower, upper = np.degrees(lower_limits()), np.degrees(upper_limits())
    print("\nwrap_to_limits on a single joint at a time:")
    print(f"  {'joint':>5} {'range [deg]':>18} {'input':>10} {'wrapped':>12} "
          f"{'inside':>7}  what happens")
    records: List[Dict[str, Any]] = []
    for index, angle, note in CASES:
        configuration = base.copy()
        configuration[index] = np.radians(angle)
        wrapped, inside = wrap_to_limits(configuration)
        if not np.allclose(np.delete(wrapped, index), np.delete(configuration, index)):
            raise AssertionError("wrap_to_limits modified an unrelated joint")
        print(f"  {index + 1:>5} {f'[{lower[index]:.1f}, {upper[index]:.1f}]':>18} {angle:>10.1f} "
              f"{np.degrees(wrapped[index]):>12.1f} {str(bool(inside[index])):>7}  {note}")
        records.append({"joint": index, "lower": float(lower[index]), "upper": float(upper[index]),
                        "input": angle, "wrapped": float(np.degrees(wrapped[index])),
                        "inside": bool(inside[index])})
    print("  the other six joints come back unchanged in every case")
    return records


def print_value_errors() -> None:
    """Show that a non-finite input is an error instead of an infinite loop."""
    print("\nwrap_to_limits rejects input it cannot map:")
    bad_finite = np.radians(HOME_DEG)
    bad_finite[3] = np.nan
    cases = (("nan in joint 4", bad_finite), ("six joints instead of seven", np.zeros(6)))
    for label, value in cases:
        try:
            wrap_to_limits(value)
        except ValueError as error:
            print(f"  {label:>28} -> ValueError: {error}")
        else:  # pragma: no cover - the library is expected to raise
            raise AssertionError(f"{label} should have raised ValueError")


def check_published_hang(timeout: float) -> Optional[bool]:
    """Run the published ``limit_joints`` on a ``nan`` in a subprocess.

    The published function never returns, so a timeout around a child process is
    the only safe way to observe it; it is never called in this process.

    Args:
        timeout: Seconds to wait before declaring the call hung.

    Returns:
        ``True`` when the call hung, ``False`` when it returned, ``None`` when the
        check could not be run at all.
    """
    original = REPO_ROOT / "original"
    if not (original / "ik_ca.py").exists():
        return None
    script = (
        f"import sys; sys.path.insert(0, {str(original)!r});"
        "import numpy as np, ik_ca; print('imported', flush=True);"
        "ik_ca.limit_joints(np.array([0.,0.,0.,float('nan'),0.,0.,0.]))"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return True
    except OSError as error:  # pragma: no cover - no interpreter available
        LOGGER.warning("could not start the child process: %s", error)
        return None
    return False


def build_figure(plt: Any, records: List[Dict[str, Any]]) -> Any:
    """Draw the one-sided ranges with the input and wrapped angles marked.

    Args:
        plt: The imported ``matplotlib.pyplot`` module.
        records: The records returned by :func:`run_cases`.

    Returns:
        The matplotlib figure.
    """
    joints = sorted({record["joint"] for record in records})
    figure, panels = plt.subplots(len(joints), 1, figsize=(10.0, 2.2 * len(joints)))
    for panel, joint in zip(np.atleast_1d(panels), joints):
        rows = [record for record in records if record["joint"] == joint]
        panel.axvspan(rows[0]["lower"], rows[0]["upper"], color="tab:green", alpha=0.18)
        panel.axhline(0.0, color="black", linewidth=1.0)
        for row in rows:
            panel.plot(row["input"], 0.35, "x", color="tab:red", markersize=9)
            panel.annotate(f"in {row['input']:.0f}", (row["input"], 0.35), fontsize=7,
                           textcoords="offset points", xytext=(0, 7), ha="center", color="tab:red")
            colour = "tab:blue" if row["inside"] else "tab:orange"
            panel.plot(row["wrapped"], -0.35, "o", color=colour, markersize=8)
            panel.annotate(f"out {row['wrapped']:.0f}", (row["wrapped"], -0.35), fontsize=7,
                           textcoords="offset points", xytext=(0, -14), ha="center", color=colour)
            panel.annotate("", xy=(row["wrapped"], -0.3), xytext=(row["input"], 0.3),
                           arrowprops={"arrowstyle": "->", "color": "grey", "alpha": 0.7})
        panel.set_title(f"joint {joint + 1}: allowed [{rows[0]['lower']:.0f}, "
                        f"{rows[0]['upper']:.0f}] deg", fontsize=9)
        panel.set_yticks([])
        panel.set_xlim(-400.0, 400.0)
        panel.set_ylim(-0.9, 0.9)
        panel.grid(axis="x", alpha=0.3)
    np.atleast_1d(panels)[-1].set_xlabel("angle [deg]")
    figure.tight_layout()
    return figure


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (see ``--help``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=None, help="write the figure here")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (kept for symmetry)")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO or DEBUG")
    parser.add_argument("--check-original", action=argparse.BooleanOptionalAction, default=True,
                        help="reproduce the published limit_joints() hang in a subprocess")
    parser.add_argument("--original-timeout", type=float, default=5.0,
                        help="seconds to wait before declaring the published call hung")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Print the limits and the wrapping behaviour, then draw the figure.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    LOGGER.debug("seed %d (this example is deterministic)", args.seed)

    lower, upper = lower_limits(), upper_limits()
    print("Joint limits and the wrap that the published implementation got wrong")
    print(f"  {'joint':>5} {'lower [deg]':>12} {'upper [deg]':>12}  note")
    for index in range(7):
        one_sided = "one-sided" if lower[index] >= 0.0 or upper[index] <= 0.0 else ""
        print(f"  {index + 1:>5} {np.degrees(lower[index]):>12.1f} "
              f"{np.degrees(upper[index]):>12.1f}  {one_sided}")

    records = run_cases()
    print_value_errors()

    if args.check_original:
        print("\nThe published limit_joints() of original/ik_ca.py, on a nan:")
        outcome = check_published_hang(args.original_timeout)
        if outcome is None:
            print("  skipped: original/ik_ca.py or the interpreter is not available")
        elif outcome:
            print(f"  hung: the child process was killed after {args.original_timeout:.1f} s")
            print("  cause: `while True` with `-181 <= nan` and `-181 > nan` both false, so it")
            print("         subtracts 360 from nan forever")
        else:
            print(f"  returned within {args.original_timeout:.1f} s (hang not reproduced)")
    else:
        print("\n--no-check-original given: the published limit_joints() was not exercised")

    plt = import_pyplot(args.save_dir is not None)
    finish(build_figure(plt, records), args.save_dir, "04_joint_limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
