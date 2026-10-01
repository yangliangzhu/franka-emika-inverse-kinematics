"""Shared fixtures for the ``franka_ik`` test suite.

This file holds everything the five test modules need to stay deterministic:

* a seeded RNG, so no test ever depends on the global NumPy state;
* a fixed sample of configurations drawn uniformly inside the **real** Panda
  joint limits -- joint 4 is one-sided at ``[-175, -5]`` degrees and joint 6 at
  ``[0, 214]``, so "inside the limits" is not the same as "inside +-165 degrees",
  and any test that samples uniformly on ``[-165, 165]`` accidentally produces
  unreachable configurations for joint 4;
* the flange poses of those configurations, and
* the published CasADi reference model and the four published entry points kept
  verbatim in ``original/`` (``conftest.py`` at the repository root puts that
  directory on ``sys.path``; the insert is repeated here so the suite also runs
  when pytest is pointed straight at ``tests/``).

How the suite is split:

======================  ========================================================
``test_model.py``       modified-DH model, forward kinematics, Jacobian, limits
``test_geometry.py``    the S-R-S reduction and the elbow quadratic
``test_solver.py``      the eight-branch solver and joint-limit wrapping
``test_branches.py``    branch structure, the published reference and its defects
``test_analysis.py``    the solution-set studies and failure classification
======================  ========================================================
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _path in (_ROOT, _ROOT / "original"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from franka_ik import coverage_study, fk_flange, lower_limits, upper_limits  # noqa: E402
from franka_ik.geometry import PAPER_GEOMETRY, EquivalentGeometry  # noqa: E402

#: Seed of every randomness in the suite.  Fixed, because a test that only
#: sometimes fails is worse than no test at all.
SEED = 20240901

#: Number of configurations in the shared in-limit sample.  The branch tests use
#: every one of them against all four published entry points (4 x 200 = 800
#: comparisons, the sample the measurements in the brief were taken on); the
#: other modules slice off what they need.  A pure-Python eight-branch solve
#: costs ~0.5 ms, and a CasADi entry-point call ~1 ms, so the sample is cheap.
IN_LIMIT_CONFIGS = 200

#: How many of those configurations get a flange pose attached.  Each pose costs
#: eight branch evaluations, so this is the expensive part of the sample.
POSE_CASES = 60

#: Samples for the shared :func:`franka_ik.analysis.coverage_study` report.
#: ``coverage_study`` is deterministic in its seed, so the report is computed
#: once per session and shared by ``test_branches.py`` and ``test_analysis.py``.
COVERAGE_SAMPLES = 200


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Joint-wise difference of two configurations, wrapped to ``[-pi, pi]``.

    Two joint vectors describe the same configuration when they agree modulo
    ``2*pi``; a plain subtraction does not see that, and the published code
    happily returns e.g. ``q1 + pi`` where the library returns ``q1 - pi``.
    """
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh generator seeded with :data:`SEED`.

    Function-scoped and seeded, so tests neither share state nor inherit the
    caller's global random state.
    """
    return np.random.default_rng(SEED)


@pytest.fixture
def angdiff() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The modulo-``2*pi`` joint difference, exposed as a callable fixture."""
    return angular_difference


@pytest.fixture(scope="session")
def in_limit_configurations() -> np.ndarray:
    """``(IN_LIMIT_CONFIGS, 7)`` configurations uniform inside the real limits.

    Session-scoped: the sample is fixed data, and re-drawing it per test would
    only add noise.  Sampling each joint independently and uniformly over its
    own range is what the repository's own studies do; the shoulder bias and the
    wrist offset mean the resulting workspace sample is not uniform in space,
    which is fine for the reachability claims tested here (the studies report
    per-sample rates, not volumes).
    """
    generator = np.random.default_rng(SEED)
    return generator.uniform(lower_limits(), upper_limits(), size=(IN_LIMIT_CONFIGS, 7))


@pytest.fixture(scope="session")
def pose_cases(
    in_limit_configurations: np.ndarray,
) -> Tuple[Tuple[np.ndarray, float, np.ndarray], ...]:
    """``(q, q7, pose)`` triples for the first :data:`POSE_CASES` configurations.

    The pose is the flange pose (the frame the analytical IK solves for, not the
    tool frame), and ``q7`` is the joint-7 value the pose was built with, which
    is exactly the redundancy parameter of this method.  Immutable, so no test
    can corrupt the shared sample.
    """
    cases = []
    for q in in_limit_configurations[:POSE_CASES]:
        cases.append((q, float(q[6]), fk_flange(q)))
    return tuple(cases)


@pytest.fixture(scope="session")
def published_panda():
    """The published CasADi model ``original/panda.py``: ``Panda()``.

    Built once per session: constructing it compiles several CasADi functions.
    """
    pytest.importorskip("casadi", reason="the published reference in original/ is a CasADi model")
    from panda import Panda

    return Panda()


@pytest.fixture(scope="session")
def published_entry_points() -> Dict[str, Callable[[np.ndarray, float], np.ndarray]]:
    """The four published IK entry points, keyed by their dotted names.

    Each takes ``(target_4x4_pose, zeta)`` and returns seven joint angles in
    radians, **unwrapped**: the published functions do not respect the real joint
    ranges (their ``limit_joints`` helper uses windows that are both wrong and,
    for joints 4 and 6, wider than a full turn).
    """
    pytest.importorskip("casadi", reason="the published reference in original/ is a CasADi module")
    import ik_ca
    import ik_ca2

    return {
        "ik_ca.ik_ca": ik_ca.ik_ca,
        "ik_ca.ik_ca_neg": ik_ca.ik_ca_neg,
        "ik_ca2.ik_ca": ik_ca2.ik_ca,
        "ik_ca2.ik_ca_neg": ik_ca2.ik_ca_neg,
    }


@pytest.fixture(scope="session")
def published_tool_pose(published_panda):
    """``Panda().fk`` wrapped into a numeric ``casadi.Function``.

    The wrapper is mandatory, not a convenience of this test.  ``Panda.fk``
    composes ``H @ self.rot_z(-pi/4)``, and ``Panda.rot_z`` builds its matrix
    from ``ca.SX_eye(4)``, i.e. a *symbolic* identity.  The result of
    ``Panda().fk(q)`` is therefore a symbolic ``casadi.SX`` object even when
    every entry is a numeric constant, and ``np.array`` refuses to convert it
    ("Implicit conversion of symbolic CasADi type to numeric matrix not
    supported").  ``Panda.forward_flange`` does not have the problem because it
    is wrapped in a ``ca.Function`` internally.  See
    ``tests/test_model.py::test_published_fk_is_symbolic_while_forward_flange_is_numeric``,
    which pins that defect.
    """
    casadi = pytest.importorskip(
        "casadi", reason="the published reference in original/ is a CasADi model"
    )
    symbolic = casadi.SX.sym("q", 7)
    return casadi.Function("published_tool_pose", [symbolic], [published_panda.fk(symbolic)])


@pytest.fixture(scope="session")
def coverage_report():
    """The shared :func:`franka_ik.analysis.coverage_study` measurement.

    ``samples=200, seed=0`` -- the study is fast (~0.4 s) but deterministic, so
    measuring it once and asserting on the same object keeps repeated claims
    consistent instead of re-running it with different sample counts.
    """
    return coverage_study(samples=COVERAGE_SAMPLES, seed=0)


@pytest.fixture(scope="session")
def srs_geometry() -> EquivalentGeometry:
    """The same arm with the shoulder bias removed, i.e. a *true* S-R-S arm.

    With ``bias = 0`` the elbow quadratic must collapse to the classical
    ``cos(theta4) = (d^2 - d_se^2 - d_ew^2) / (2 d_se d_ew)`` of Shimizu et al.
    (2008), which is the reference the reduction is checked against.
    """
    return replace(PAPER_GEOMETRY, bias=0.0)


@pytest.fixture(scope="session")
def paper_geometry() -> EquivalentGeometry:
    """The published geometry of the Panda on this branch."""
    return PAPER_GEOMETRY
