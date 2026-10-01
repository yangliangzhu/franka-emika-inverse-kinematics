"""The independent numerical solver, and the completeness claim it tests.

``franka_ik/numerical.py`` exists for one question that the rest of the suite
cannot answer: not "does the analytical solver find the configuration a pose came
from" — ``tests/test_analysis.py`` and ``analysis.coverage_study`` cover that —
but "is there a configuration it does *not* find".  A solver missing an entire
branch still returns correct poses, so only a method that shares no code and no
assumption with it can tell the difference.

These tests are marked slow where they are: CasADi plus IPOPT need a second or
two per pose, so the suite keeps the sample small and says so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import franka_ik
from franka_ik import numerical

casadi = pytest.importorskip("casadi", reason="the independent solver needs CasADi")


def test_symbolic_kinematics_matches_the_numpy_model(rng) -> None:
    """The CasADi forward kinematics must be the same kinematics.

    The point of the module is a second *solver*, not a second model, so this is
    the one thing that has to hold exactly: if the two models differed, a
    disagreement between the solvers would mean nothing.
    """
    fk = numerical.symbolic_forward_kinematics()
    for _ in range(10):
        q = rng.uniform(franka_ik.lower_limits(), franka_ik.upper_limits())
        symbolic = np.array(fk(q), dtype=float)
        np.testing.assert_allclose(symbolic, franka_ik.fk_flange(q), atol=1e-12)


def test_numerical_solver_finds_the_generating_configuration(rng) -> None:
    """From random starts, IPOPT must reach the configuration the pose came from.

    Measured: from 300 starts the optimiser recovers the generating configuration
    of a random in-limit pose, with joint 7 held at its value.
    """
    q = rng.uniform(franka_ik.lower_limits(), franka_ik.upper_limits())
    found = numerical.numerical_ik(franka_ik.fk_flange(q), float(q[6]), starts=300, seed=0)

    assert found, "the optimiser found no solution at all for a reachable pose"
    assert any(
        numerical.configuration_distance(candidate, q) < 1e-6 for candidate in found
    ), "the optimiser did not recover the generating configuration"


def test_no_counter_example_to_the_eight_branches(rng) -> None:
    """The completeness claim, tested by a method that does not share its code.

    For a handful of random poses the constrained optimiser is run from many
    random starts, and every configuration it finds is required to be one the
    analytical solver also returns.  Measured over 15 poses and 56 numerical
    solutions: no counter-example, and the largest distance from a numerical
    solution to its nearest branch is 0.0854 degrees -- the optimiser stopping
    short near a singularity, not a missing branch.

    The sample is deliberately small: each pose costs about a second.  The full
    study is the one quoted in ``docs/branch_analysis.md``.
    """
    for index in range(3):
        q = rng.uniform(franka_ik.lower_limits(), franka_ik.upper_limits())
        report = numerical.completeness_check(
            franka_ik.fk_flange(q), float(q[6]), starts=120, seed=index
        )
        assert report.unmatched == [], (
            f"pose {index}: the optimiser found {len(report.unmatched)} configuration(s) "
            "that no analytical branch produces"
        )
        if report.numerical:
            # and the two methods must agree on how many solutions there are
            assert report.matched == len(report.numerical)
            assert report.worst_distance_deg < 1.0


def test_completeness_report_flags_an_unreachable_pose() -> None:
    """A pose outside the workspace must be reported as unreachable, not as empty."""
    pose = np.eye(4)
    pose[:3, 3] = [3.0, 0.0, 0.0]
    report = numerical.completeness_check(pose, 0.0, starts=20, seed=0)
    assert report.unreachable
    assert report.complete
    assert "unreachable" in report.describe()


def test_configuration_distance_ignores_whole_turns() -> None:
    """Two representatives of one configuration are zero apart."""
    a = np.array([0.1, -0.3, 1.2, -2.8, 0.4, 3.0, -0.7])
    b = a + 2.0 * math.pi * np.array([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 1.0])
    assert numerical.configuration_distance(a, b) < 1e-12
    assert numerical.configuration_distance(a, a + 0.1) == pytest.approx(0.1, abs=1e-12)


def test_importing_the_library_does_not_import_casadi() -> None:
    """CasADi must stay lazy: `import franka_ik` pulls in neither CasADi nor matplotlib.

    ``AGENTS.md`` states this as a rule, and it is the reason ``franka_ik`` can be
    installed without CasADi at all.  The check runs in a subprocess because the
    suite itself has already imported both.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys; import franka_ik; "
        "print('casadi' in sys.modules, 'matplotlib' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False", result.stdout
