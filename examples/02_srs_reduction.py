"""STEP 1 and STEP 2 -- reduce the Panda to an equivalent S-R-S arm.

**STEP 1 (the wrist offset).**  DH row 7 carries ``offset = 0.088 m`` sideways, so
the flange frame sits at a fixed extra angle ``beta = atan2(offset, d_wt)`` from a
true S-R-S flange and the last link is effectively ``hypot(offset, d_wt)``.  The
example shows that rotating the target by ``Rz(-q7) Ry(-beta) Rz(q7)`` makes the
elbow law hold exactly, while on the raw frame it does not.

**STEP 2 (the shoulder bias).**  ``bias = 0.0825 m`` shifts the elbow, so joint 4
is a root of the quadratic ``a2 x^2 - a1 x + a0 = 0`` in ``x = tan(theta4 / 2)``
instead of obeying the plain S-R-S law.  Both roots are printed and drawn, because
the published solver kept only the ``+`` one.  Setting ``bias = 0`` collapses the
quadratic back onto eq. (12) of Shimizu et al.,
``cos(theta4) = (d^2 - d_se^2 - d_ew^2) / (2 d_se d_ew)``.

Run from the repository root::

    MPLBACKEND=Agg python3 examples/02_srs_reduction.py --save-dir /tmp/srs
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_ik import (  # noqa: E402
    PAPER_GEOMETRY,
    EquivalentGeometry,
    fk_flange,
    lower_limits,
    q4_coefficients,
    q4_discriminant,
    q4_roots,
    upper_limits,
)
from franka_ik.geometry import (  # noqa: E402
    effective_wrist_length,
    equivalent_link_vectors,
    link_offset,
    shoulder_to_wrist,
    wrist_correction,
    wrist_offset_angle,
)

LOGGER = logging.getLogger("examples.02_srs_reduction")

#: Wrist distances used for the coefficient table and the quadratic plot.
DISTANCES = (0.30, 0.40, 0.50, 0.60)


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


def without_bias(geometry: EquivalentGeometry) -> EquivalentGeometry:
    """The same arm with ``bias = 0``, i.e. a plain S-R-S arm.

    Args:
        geometry: Geometric parameters.

    Returns:
        The parameters with the shoulder offset removed.
    """
    return EquivalentGeometry(
        d_bs=geometry.d_bs,
        d_se=geometry.d_se,
        d_ew=geometry.d_ew,
        d_wt=geometry.d_wt,
        offset=geometry.offset,
        bias=0.0,
    )


def report_step1(configuration: np.ndarray, geometry: EquivalentGeometry) -> None:
    """Show the wrist correction making the flange behave like an S-R-S flange.

    Args:
        configuration: Configuration to analyse, in radians.
        geometry: Geometric parameters.
    """
    pose = fk_flange(configuration)
    q7, true_q4 = float(configuration[6]), float(configuration[3])
    correction = wrist_correction(q7, geometry)
    print("STEP 1 -- the wrist correction makes the flange frame S-R-S like")
    print(f"  configuration   = {np.round(np.degrees(configuration), 3)} deg")
    print(f"  true joint 4    = {math.degrees(true_q4):.6f} deg")
    print(f"  beta            = {math.degrees(wrist_offset_angle(geometry)):.6f} deg, "
          f"effective wrist length = {effective_wrist_length(geometry):.9f} m "
          f"(d_wt = {geometry.d_wt:.4f} m)")
    print("  wrist_correction(q7) = Rz(-q7) Ry(-beta) Rz(q7), first row "
          f"{np.round(correction[0], 6)}, det = {np.linalg.det(correction):.9f}")
    for name, rotation in (("corrected", pose[:3, :3] @ correction), ("raw      ", pose[:3, :3])):
        p_sw = shoulder_to_wrist(pose[:3, 3], rotation, geometry)
        roots = q4_roots(float(p_sw @ p_sw), geometry)
        nearest = min(roots, key=lambda angle: abs(angle - true_q4))
        print(
            f"  {name} frame: ||x_sw|| = {np.linalg.norm(p_sw):.9f} m, roots = "
            f"[{', '.join(f'{math.degrees(a):.4f}' for a in roots)}] deg, nearest to the "
            f"true joint 4 off by {abs(math.degrees(nearest - true_q4)):.3e} deg"
        )


def report_step2(configuration: np.ndarray, geometry: EquivalentGeometry) -> Dict[float, np.ndarray]:
    """Print the elbow quadratic, its discriminant and both of its roots.

    Args:
        configuration: Configuration to analyse, in radians.
        geometry: Geometric parameters.

    Returns:
        The roots of the quadratic, keyed by wrist distance.
    """
    pose = fk_flange(configuration)
    corrected = pose[:3, :3] @ wrist_correction(float(configuration[6]), geometry)
    p_sw = shoulder_to_wrist(pose[:3, 3], corrected, geometry)
    print("\nSTEP 2 -- the elbow quadratic a2 x^2 - a1 x + a0 = 0 with x = tan(theta4 / 2)")
    print(f"  {'d [m]':>7} {'a2':>11} {'a1':>10} {'a0':>10} {'discriminant':>14} "
          f"{'x roots':>20} {'theta4 roots [deg]':>24}")
    roots_by_distance: Dict[float, np.ndarray] = {}
    for distance in DISTANCES:
        squared = distance * distance
        a2, a1, a0 = q4_coefficients(squared, geometry)
        roots = np.asarray(q4_roots(squared, geometry))
        roots_by_distance[distance] = roots
        print(
            f"  {distance:>7.2f} {a2:>11.6f} {a1:>10.6f} {a0:>10.6f} "
            f"{q4_discriminant(squared, geometry):>14.6f} "
            f"{', '.join(f'{math.tan(a / 2.0):.4f}' for a in roots):>20} "
            f"{', '.join(f'{math.degrees(a):.4f}' for a in roots):>24}"
        )
    print(f"\n  at this configuration ||x_sw|| = {np.linalg.norm(p_sw):.9f} m, "
          f"roots: {len(q4_roots(float(p_sw @ p_sw), geometry))}")
    for angle in q4_roots(float(p_sw @ p_sw), geometry):
        l_se, l_ew = equivalent_link_vectors(angle, geometry)
        print(f"    theta4 = {math.degrees(angle):>9.4f} deg -> delta = "
              f"{link_offset(angle, geometry):+.6f} m, l_se = [0, {l_se[1]:.6f}, 0], "
              f"l_ew = [0, 0, {l_ew[2]:.6f}]")
    return roots_by_distance


def report_collapse(geometry: EquivalentGeometry) -> None:
    """Check that ``bias = 0`` collapses the quadratic onto the S-R-S elbow law.

    Args:
        geometry: Geometric parameters.
    """
    distances = np.linspace(0.10, 0.68, 40)
    print("\nbias = 0: the quadratic collapses onto eq. (12) of Shimizu et al.")
    for name, item in (("bias = 0", without_bias(geometry)), (f"bias = {geometry.bias}", geometry)):
        worst, a1_values = 0.0, set()
        for distance in distances:
            squared = float(distance) ** 2
            a2, a1, a0 = q4_coefficients(squared, item)
            a1_values.add(round(a1, 15))
            law = (squared - item.d_se**2 - item.d_ew**2) / (2.0 * item.d_se * item.d_ew)
            for angle in q4_roots(squared, item):
                worst = max(worst, abs(math.cos(angle) - law))
        print(f"  {name:>14}: a1 = {sorted(a1_values)}, "
              f"max |cos(theta4)_quadratic - cos(theta4)_SRS| = {worst:.3e}")
    print("  a1 = 4 b (d_se + d_ew) vanishes with the bias, leaving a2 x^2 + a0 = 0")


def build_figure(
    plt: Any, geometry: EquivalentGeometry, roots_by_distance: Dict[float, np.ndarray]
) -> Any:
    """Draw the elbow quadratic with both roots, and the bias = 0 collapse.

    Args:
        plt: The imported ``matplotlib.pyplot`` module.
        geometry: Geometric parameters.
        roots_by_distance: Roots of the quadratic, keyed by wrist distance.

    Returns:
        The matplotlib figure.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    grid = np.linspace(-6.0, 2.5, 600)
    for index, (distance, roots) in enumerate(sorted(roots_by_distance.items())):
        a2, a1, a0 = q4_coefficients(distance * distance, geometry)
        (line,) = left.plot(grid, a2 * grid**2 - a1 * grid + a0, label=f"d = {distance:.2f} m")
        x_roots = [math.tan(angle / 2.0) for angle in roots]
        left.plot(x_roots, np.zeros(len(x_roots)), "o", color=line.get_color(), markersize=7)
        # The positive roots of the four curves pile up near x ~ 0.5, so the
        # annotations are staggered vertically instead of overlapping.
        for position, (x_root, angle) in enumerate(zip(x_roots, roots)):
            offset = (0, 10 + 14 * index) if position == 0 else (0, -13 - 14 * index)
            left.annotate(f"{math.degrees(angle):.0f} deg", (x_root, 0.0), textcoords="offset points",
                          xytext=offset, ha="center", fontsize=7, color=line.get_color())
    left.axhline(0.0, color="black", linewidth=0.8)
    left.set_xlim(-6.0, 2.5)
    left.set_ylim(-1.5, 1.5)
    left.set_xlabel("x = tan(theta4 / 2)")
    left.set_ylabel("f(x) = a2 x^2 - a1 x + a0")
    left.set_title("elbow quadratic, both roots marked")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    distances = np.linspace(0.10, 0.68, 120)
    law = [(d**2 - geometry.d_se**2 - geometry.d_ew**2) / (2.0 * geometry.d_se * geometry.d_ew)
           for d in distances]
    right.plot(distances, law, linewidth=3.0, alpha=0.4, label="S-R-S law, eq. (12)")
    right.plot(distances, [math.cos(q4_roots(float(d) ** 2, without_bias(geometry))[0]) for d in distances],
               "--", label="quadratic with bias = 0")
    right.plot(distances, [math.cos(q4_roots(float(d) ** 2, geometry)[0]) for d in distances],
               "-.", label=f"quadratic with bias = {geometry.bias} m")
    right.set_xlabel("||x_sw|| [m]")
    right.set_ylabel("cos(theta4)")
    right.set_title("bias = 0 collapses onto the S-R-S law")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    figure.tight_layout()
    return figure


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (see ``--help``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, default=None, help="write the figure here")
    parser.add_argument("--seed", type=int, default=3, help="RNG seed (3 gives a moderate joint 4)")
    parser.add_argument("--log-level", default="INFO", help="logging level, e.g. INFO or DEBUG")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run both steps of the reduction and draw the figure.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    geometry = PAPER_GEOMETRY

    print("STEP 1 and STEP 2 -- reduction of the Panda to an equivalent S-R-S arm")
    print(f"  d_bs = {geometry.d_bs:.4f}, d_se = {geometry.d_se:.4f}, d_ew = {geometry.d_ew:.4f}, "
          f"d_wt = {geometry.d_wt:.4f} m")
    print(f"  offset = {geometry.offset:.4f} m (DH row 7), bias = {geometry.bias:.4f} m "
          f"(DH rows 4 and 5)")

    configuration = np.random.default_rng(args.seed).uniform(lower_limits(), upper_limits())
    report_step1(configuration, geometry)
    roots_by_distance = report_step2(configuration, geometry)
    report_collapse(geometry)

    plt = import_pyplot(args.save_dir is not None)
    finish(build_figure(plt, geometry, roots_by_distance), args.save_dir, "02_srs_reduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
