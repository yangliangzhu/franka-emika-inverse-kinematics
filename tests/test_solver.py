"""The eight-branch solver, joint-limit wrapping and branch continuity (facts 3, 5, 9).

``franka_ik/solver.py`` is the part of the repository that *does* something: it
evaluates the eight candidate configurations of a pose, verifies each one against
the forward kinematics, and reports what cannot be wrapped into the real joint
ranges.  Three claims are tested here.

Round trip (fact 3)
    Solving the flange pose of a configuration with that configuration's own
    joint 7 must return the configuration itself.  This is the property that
    fails for the published four-branch solver, and the one the repository's
    coverage study measures -- the solver here is checked on the same
    configurations, not on poses of its own making.

Refusing non-finite input (fact 5)
    The published ``limit_joints`` never terminates on a ``nan``; the
    replacement raises.  The *reproduction* of the published hang lives in
    ``test_branches.py``, because it needs a hard timeout; this module asserts
    the replacement's contract.

Continuity (fact 9)
    ``solve_closest`` must return the branch nearest a reference configuration
    rather than whatever branch happens to come first, which is what makes the
    solver usable for tracking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from franka_ik import model, solver

#: Configurations for the round-trip test.  300 draws is the sample the
#: repository's own studies use; it covers the whole reachable workspace
#: (the sample includes poses with 1 as well as 8 distinct solutions) and keeps
#: the pure-Python eight-branch solve around a second.
ROUND_TRIP_SAMPLES = 300

#: Configurations for the ``solve_closest`` test, as in fact 9.
CLOSEST_SAMPLES = 50

#: Absolute tolerance for "the solver found the configuration it was given".
#: Measured worst deviation 4.7e-13 rad over 300 draws; 1e-6 rad is the threshold
#: the library itself uses to decide that two configurations coincide
#: (``solver._deduplicate``, ``analysis._same_configuration``), so this is the
#: same notion of equality the solver works with -- and 1e-6 rad = 6e-5 degrees,
#: far below any real actuator resolution.
SELF_RECOVERY_TOLERANCE = 1e-6

#: Upper bound on the pose residual of any returned solution.  ``pose_error`` is
#: the largest absolute entrywise difference of ``fk_flange(q)`` against the
#: target, i.e. metres for the position block and a dimensionless rotation error
#: for the orientation block.  It is the solver's own acceptance threshold
#: (``solve(tolerance=1e-9)``), and the measured worst value over 300 round trips
#: is 2.9e-13 -- so the bound is what the solver promises, not a value the tests
#: had to relax.
POSE_ERROR_TOLERANCE = 1e-9

#: Distance by which the test moves a pose to put it well outside the workspace.
#: The arm's reach from the shoulder is at most 0.72 m, so 3 m cannot be reached
#: by any joint-7 value.
OUT_OF_REACH_DISTANCE = 3.0

#: The two joint-7 values at which the published wrist criterion degenerates:
#: ``criteria = r_47[2, 0] * cos(q7)`` is *identically zero* at ``+-90`` degrees,
#: so the published sign test is decided by floating-point noise there.  Joint 7
#: spans ``+-165`` degrees, so both values are inside the real range and are
#: reached routinely.
WRIST_DEGENERATE_Q7 = (math.pi / 2.0, -math.pi / 2.0)

#: A wrist flipped the wrong way misses the pose by O(0.1 .. 2) in homogeneous
#: units, not by a rounding error.  Measured worst residual of the candidate the
#: *published* criterion selects at ``q7 = +-90`` degrees: 1.95 (at +90) and 1.89
#: (at -90).  1e-2 is two orders below that and six orders above the fixed
#: solver's worst residual (1.3e-15), so the two cannot be confused.
WRONG_WRIST_RESIDUAL = 1e-2


def test_num_branches_is_eight() -> None:
    """The published branch count is eight (fact 4).

    Four independent two-way choices (elbow root, arm angle, shoulder flip, and
    the internal wrist criterion) -- the published implementation enumerated
    only the first, the second and the shoulder flip *within* one elbow root,
    which is why it has four entry points instead of eight branches.
    """
    assert solver.NUM_BRANCHES == 8


def test_wrap_to_limits_rejects_non_finite_input() -> None:
    """Fact 5: a ``nan`` raises ``ValueError`` instead of spinning forever.

    The published ``limit_joints`` compares against the angle in a ``while True``
    loop: every comparison with ``nan`` is False, so it subtracts 360 degrees
    forever (reproduced under a hard timeout in ``test_branches.py``).  Here the
    contract of the replacement is asserted for a ``nan`` anywhere in the vector
    and for both infinities.  The defect is written up in ``docs/limitations.md``
    and in ``docs/branch_analysis.md`` §7; this test is the executable record.
    """
    for bad in (math.nan, math.inf, -math.inf):
        angles = np.zeros(model.NUM_JOINTS)
        angles[3] = bad
        with pytest.raises(ValueError):
            solver.wrap_to_limits(angles)
        with pytest.raises(ValueError):
            solver.wrap_to_limits(np.full(model.NUM_JOINTS, bad))


def test_wrap_to_limits_rejects_a_wrong_shape() -> None:
    """A joint vector that is not seven long is rejected, not broadcast.

    Silently broadcasting a 6- or 8-element input would shift every joint by one
    and produce plausible-looking garbage.
    """
    with pytest.raises(ValueError):
        solver.wrap_to_limits(np.zeros(model.NUM_JOINTS - 1))
    with pytest.raises(ValueError):
        solver.wrap_to_limits(np.zeros((model.NUM_JOINTS, 1)))


def test_wrap_to_limits_keeps_in_limit_configurations_fixed(in_limit_configurations) -> None:
    """A configuration already inside the limits comes back unchanged and flagged.

    Bit-exact equality is the correct assertion here: the implementation builds
    the candidates as ``q + 2*pi*k`` and picks the first valid one, which for an
    in-limit ``q`` is ``q + 0.0``.  The real limits matter for this test -- joint
    4 lives in ``[-175, -5]`` degrees, so a configuration drawn on
    ``[-165, 165]`` would *not* be representable and the fixture would be wrong.
    """
    for q in in_limit_configurations:
        wrapped, inside = solver.wrap_to_limits(q)
        assert np.all(inside)
        np.testing.assert_array_equal(wrapped, q)


def test_wrap_to_limits_shifts_whole_turns_back_into_range(in_limit_configurations) -> None:
    """Adding or subtracting full turns does not change the wrapped configuration.

    Each real joint range is narrower than 360 degrees (widest: joint 7 at 330),
    so at most one representative of an angle can fit and the wrap is
    unambiguous -- the wrapped value must come back as the original
    configuration.  Not *bit*-exactly, though: the input here is ``q +- k*2*pi``
    and the wrap adds ``-+ k*2*pi`` back, so two roundings intervene and the
    recovered value carries a couple of ULP (measured worst 4.4e-16 rad, i.e.
    2 ULP of a number of order 1); hence 1e-15 rather than exact equality.
    Only whole-turn shifts the implementation actually searches (``k`` in
    ``-2..2``) are used; a shift of 5 turns is *not* required to be recovered and
    is deliberately not asserted.
    """
    two_pi = 2.0 * math.pi
    for q in in_limit_configurations[:40]:
        for turns in (1, -1, 2, -2):
            wrapped, inside = solver.wrap_to_limits(q + turns * two_pi)
            assert np.all(inside)
            np.testing.assert_allclose(wrapped, q, atol=1e-15)


def test_wrap_to_limits_reports_what_it_cannot_represent() -> None:
    """A joint with no representative in range is flagged, not silently moved.

    Joint 4 is one-sided (``[-175, -5]`` degrees): ``+1 rad`` (= 57.3 degrees) is
    inside no representative, so ``inside[3]`` is False and the reported value is
    the whole-turn representative closest to the middle of the range -- the input
    itself here.  ``-pi/2`` is representable and must be flagged True.  The
    published ``limit_joints`` accepts both (its joint-4 window is
    ``[-271, 91]`` degrees, 362 wide), which is exactly the defect documented in
    ``test_branches.py``.
    """
    angles = np.zeros(model.NUM_JOINTS)
    angles[3] = 1.0
    wrapped, inside = solver.wrap_to_limits(angles)
    assert bool(inside[3]) is False
    assert wrapped[3] == pytest.approx(1.0, abs=1e-15)
    assert np.all(inside[[0, 1, 2, 4, 5, 6]])

    representable = np.zeros(model.NUM_JOINTS)
    representable[3] = -math.pi / 2
    wrapped, inside = solver.wrap_to_limits(representable)
    assert bool(inside[3]) is True
    assert wrapped[3] == pytest.approx(-math.pi / 2, abs=1e-15)

    # Joint 6 is the other one-sided joint: [0, 214] degrees, so 250 is not
    # representable while 100 is.
    angles = np.zeros(model.NUM_JOINTS)
    angles[5] = math.radians(250.0)
    _, inside = solver.wrap_to_limits(angles)
    assert bool(inside[5]) is False
    angles[5] = math.radians(100.0)
    _, inside = solver.wrap_to_limits(angles)
    assert bool(inside[5]) is True


def test_round_trip_recovers_the_sampled_configuration(rng) -> None:
    """Fact 3: ``solve(fk_flange(q), q[6], within_limits_only=True)`` contains ``q``.

    The headline property of the analytical solver: the pose and the joint-7
    value are exactly the information the method is parameterised by, so the
    configuration they came from must be among the answers.  Comparison is
    modulo ``2*pi`` because the solver returns angles wrapped into the real
    ranges while the sample is already in range (they should therefore agree
    exactly; the tolerance covers the ~5e-13 rad of arithmetic, not wrapping).
    Measured 300/300 with a worst deviation of 4.7e-13 rad.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    recovered = 0
    worst = 0.0
    for _ in range(ROUND_TRIP_SAMPLES):
        q = rng.uniform(lower, upper)
        solutions = solver.solve(model.fk_flange(q), float(q[6]), within_limits_only=True)
        best = min(
            (
                float(np.max(np.abs(np.arctan2(np.sin(solution.q - q), np.cos(solution.q - q)))))
                for solution in solutions
            ),
            default=math.inf,
        )
        worst = max(worst, best)
        if best < SELF_RECOVERY_TOLERANCE:
            recovered += 1
    assert recovered == ROUND_TRIP_SAMPLES
    assert worst < SELF_RECOVERY_TOLERANCE


def test_round_trip_solutions_have_small_pose_residual(rng) -> None:
    """Fact 3, second half: every returned solution really reaches the pose.

    ``solve`` drops any branch whose residual exceeds its own 1e-9 tolerance, so
    this asserts that the acceptance test is actually applied and that no
    solution is reported without being verified.  Measured worst residual 2.9e-13
    over 300 configurations, i.e. ~0.3 picometres; the bound of 1e-9 is the
    solver's promise, and the number of solutions checked is reported so a
    regression that returns an empty list cannot pass by vacuity.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    worst = 0.0
    checked = 0
    for _ in range(ROUND_TRIP_SAMPLES):
        q = rng.uniform(lower, upper)
        solutions = solver.solve(model.fk_flange(q), float(q[6]), within_limits_only=True)
        assert solutions, "a configuration inside the limits must have a solution"
        for solution in solutions:
            assert solution.pose_error < POSE_ERROR_TOLERANCE
            worst = max(worst, solution.pose_error)
            checked += 1
    assert worst < POSE_ERROR_TOLERANCE
    # 300 configurations with at least one solution each (the mean is ~3.2).
    assert checked >= ROUND_TRIP_SAMPLES


def test_wrist_selection_recovers_the_target_at_plus_minus_ninety_degrees(rng) -> None:
    """Regression: ``q7 = +-90`` degrees must still produce the right wrist.

    Between the two wrist configurations ``(q5, q6)`` and ``(q5 + pi, -q6)`` the
    published code chooses with ``criteria = r_47[2, 0] * cos(q7)``, a restatement
    of the sign of ``cos(theta_7)`` inherited from ``tan(theta_7) = N / D``.  It
    therefore carries a factor ``cos(q7)`` and **vanishes at q7 = +-90 degrees**,
    where the sign test is decided by floating-point noise: the published code
    then either flips the wrist the wrong way (pose residual ~1.9) or reports no
    solution at all for a reachable pose.

    The test builds targets exactly inside that degeneracy -- a random in-limit
    configuration with joint 7 overwritten by ``+-pi/2`` -- and requires that the
    configuration itself comes back.  Measured with the fix: 60/60 targets per
    sign, worst pose residual 1.3e-15 (at +90) and 9.4e-16 (at -90).
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    for q7 in WRIST_DEGENERATE_Q7:
        for _ in range(30):
            q = rng.uniform(lower, upper)
            q[6] = q7
            pose = model.fk_flange(q)
            solutions = solver.solve(pose, float(q[6]), within_limits_only=False)
            assert solutions, "a reachable pose at q7 = +-90 deg must have a solution"
            for solution in solutions:
                assert solution.pose_error < POSE_ERROR_TOLERANCE
            deviations = [
                float(np.max(np.abs(np.arctan2(np.sin(s.q - q), np.cos(s.q - q)))))
                for s in solutions
            ]
            assert min(deviations) < SELF_RECOVERY_TOLERANCE


def test_published_wrist_criterion_degrades_at_plus_minus_ninety_degrees(rng) -> None:
    """The published criterion alone picks the wrong wrist there; the fix does not.

    ``solve_branch(check_pose=False)`` is the published behaviour exactly (it is
    the branch the library keeps for comparability): it selects the wrist from
    ``criteria`` and reports no residual.  Evaluating that choice against the
    forward kinematics shows what the criterion costs: at ``+-90`` degrees the
    selected candidate misses the pose by up to 1.95, five orders of magnitude
    worse than the residual-based selection used by default.  The residual is
    recomputed here rather than read from the solution, because with
    ``check_pose=False`` the library deliberately reports ``inf``.

    This is the test that would have caught the defect, and it keeps the cause
    visible: it is not that the criterion is wrong in general (it is exact for
    ``|cos(q7)|`` away from zero, which is why the published code works almost
    everywhere) but that it is *undefined* at these two angles.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    for q7 in WRIST_DEGENERATE_Q7:
        worst_published = 0.0
        worst_fixed = 0.0
        for _ in range(30):
            q = rng.uniform(lower, upper)
            q[6] = q7
            pose = model.fk_flange(q)
            for phi_root in (1, -1):
                for q4_root in (1, -1):
                    published = solver.solve_branch(
                        pose, float(q[6]), q4_root=q4_root, phi_root=phi_root, check_pose=False
                    )
                    if published is None:
                        continue
                    residual = float(np.max(np.abs(model.fk_flange(published.q) - pose)))
                    worst_published = max(worst_published, residual)
            for solution in solver.solve(pose, float(q[6]), within_limits_only=False):
                worst_fixed = max(worst_fixed, solution.pose_error)
        assert worst_published > WRONG_WRIST_RESIDUAL
        assert worst_fixed < POSE_ERROR_TOLERANCE


def test_joint_seven_sweep_recovers_every_target(in_limit_configurations) -> None:
    """A sweep of the whole joint-7 range finds and recovers every target.

    Joint 7 is the redundancy parameter, so every one of its values is a distinct
    problem; the sweep is coarse (every 5 degrees from -160 to 160, three
    configurations each, 195 solves) but it includes both ``+-90`` degrees, the
    angles where the published wrist test degenerates.  Both halves are asserted:
    no reachable pose may come back empty, and the generating configuration must
    be among the answers.  Measured: 195/195 recovered, worst pose residual
    3.5e-15.  The reference measurement over a finer grid (every degree, six
    configurations, 1974 solves) is 0 failures with a worst residual of 1.0e-12.
    """
    bases = list(in_limit_configurations[:3])
    trials = 0
    worst = 0.0
    swept = []
    for degrees in range(-160, 161, 5):
        swept.append(degrees)
        for base in bases:
            q = base.copy()
            q[6] = math.radians(float(degrees))
            pose = model.fk_flange(q)
            solutions = solver.solve(pose, float(q[6]), within_limits_only=False)
            assert solutions, f"joint 7 = {degrees} deg produced no solution"
            for solution in solutions:
                worst = max(worst, solution.pose_error)
            deviations = [
                float(np.max(np.abs(np.arctan2(np.sin(s.q - q), np.cos(s.q - q)))))
                for s in solutions
            ]
            assert min(deviations) < SELF_RECOVERY_TOLERANCE
            trials += 1
    assert trials == 65 * len(bases)
    assert 90 in swept and -90 in swept
    assert worst < POSE_ERROR_TOLERANCE


def test_solve_branch_rejects_invalid_root_labels(pose_cases) -> None:
    """The branch labels are ``+1``/``-1``; anything else is an error.

    A silent acceptance of, say, ``q4_root=0`` would select the negative root
    (`q4_root > 0` is False) and quietly return the wrong branch set.
    """
    _, q7, pose = pose_cases[0]
    with pytest.raises(ValueError):
        solver.solve_branch(pose, q7, q4_root=0)
    with pytest.raises(ValueError):
        solver.solve_branch(pose, q7, phi_root=0)
    with pytest.raises(ValueError):
        solver.solve_branch(pose, q7, q4_root=2)


def test_solve_returns_nothing_for_an_unreachable_pose(pose_cases) -> None:
    """A pose 3 m away has no solution, and the solver says so instead of guessing.

    The arm reaches at most 0.72 m from the shoulder, so the elbow quadratic has
    a negative discriminant and every branch returns ``None``.  Asserted with
    ``within_limits_only=False``: an out-of-reach pose must not be "solved" by a
    branch that only looks acceptable after wrapping.
    """
    _, q7, pose = pose_cases[0]
    unreachable = pose.copy()
    unreachable[:3, 3] += np.array([OUT_OF_REACH_DISTANCE, 0.0, 0.0])
    assert solver.solve(unreachable, q7, within_limits_only=False) == []
    assert solver.solve(unreachable, q7, within_limits_only=True) == []
    assert solver.branch_solutions(unreachable, q7) == []
    assert solver.solve_closest(unreachable, q7, pose_cases[0][0]) is None


def test_solve_deduplicates_and_filters_consistently(pose_cases, angdiff) -> None:
    """``deduplicate``/``within_limits_only`` change the *count*, never the solutions.

    Three properties, all of which the branch structure makes easy to get wrong:
    requesting the in-limit subset must not add solutions, skipping
    deduplication must not remove any, and no two deduplicated solutions may
    describe the same configuration (branches coincide at singular poses, which
    is why ``_deduplicate`` exists).  The ``within_limits`` flag is recomputed
    from the model's limits rather than trusted.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    for _, q7, pose in pose_cases[:20]:
        everything = solver.solve(pose, q7, deduplicate=False, within_limits_only=False)
        unique = solver.solve(pose, q7, deduplicate=True, within_limits_only=False)
        in_limits = solver.solve(pose, q7, deduplicate=True, within_limits_only=True)

        assert len(unique) <= len(everything)
        assert len(in_limits) <= len(unique)
        for solution in unique:
            for other in unique:
                if solution is other:
                    continue
                assert float(np.max(angdiff(solution.q, other.q))) > 1e-6
        for solution in in_limits:
            assert solution.within_limits is True
            assert np.all(solution.q >= lower - 1e-9)
            assert np.all(solution.q <= upper + 1e-9)


def test_ik_solution_describes_itself(pose_cases) -> None:
    """``IkSolution`` carries the branch label, the residual and the limit flag.

    ``label`` is what ``analysis.study_pose`` keys its outcomes on and what the
    documentation quotes, so its format is asserted literally; ``describe`` must
    stay printable (it is used by the demo pages).
    """
    _, q7, pose = pose_cases[0]
    solutions = solver.solve(pose, q7, deduplicate=False, within_limits_only=False)
    assert solutions
    labels = set()
    for solution in solutions:
        assert solution.q.shape == (model.NUM_JOINTS,)
        assert solution.q4_root in (1, -1)
        assert solution.phi_root in (1, -1)
        assert isinstance(solution.shoulder_flip, bool)
        assert isinstance(solution.wrist_flipped, bool)
        assert solution.label in {
            f"q4{'+' if q4 > 0 else '-'} phi{'+' if phi > 0 else '-'} {'flip' if flip else 'plain'}"
            for q4 in (1, -1)
            for phi in (1, -1)
            for flip in (False, True)
        }
        assert solution.pose_error < POSE_ERROR_TOLERANCE
        assert isinstance(solution.describe(), str)
        assert solution.label in solution.describe()
        labels.add(solution.label)
    # Different branches, different labels.
    assert len(labels) == len(solutions)


def test_branch_solutions_agrees_with_solve_branch(pose_cases) -> None:
    """``branch_solutions`` is exactly the eight ``solve_branch`` evaluations.

    The two APIs are used interchangeably (the studies use one, the tests the
    other), so they must not drift: the labels returned in bulk must equal the
    labels of the branches that individually return a solution.
    """
    _, q7, pose = pose_cases[0]
    expected = []
    for q4_root in (1, -1):
        for phi_root in (1, -1):
            for shoulder_flip in (False, True):
                solution = solver.solve_branch(
                    pose, q7, q4_root=q4_root, phi_root=phi_root, shoulder_flip=shoulder_flip
                )
                if solution is not None:
                    expected.append(solution.label)
    actual = [solution.label for solution in solver.branch_solutions(pose, q7)]
    assert actual == expected
    assert len(actual) <= solver.NUM_BRANCHES


def test_solve_closest_returns_the_reference_configuration(rng) -> None:
    """Fact 9: given the pose of ``q`` and ``q`` as the reference, return ``q``.

    The eight branches are discrete, so a controller that takes the first one
    jumps between configurations.  Measured 50/50 with a worst deviation of
    1.0e-13 rad, and the tolerance is the library's own "same configuration"
    threshold of 1e-6 rad.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    for _ in range(CLOSEST_SAMPLES):
        q = rng.uniform(lower, upper)
        pose = model.fk_flange(q)
        closest = solver.solve_closest(pose, float(q[6]), q)
        assert closest is not None
        deviation = np.max(np.abs(np.arctan2(np.sin(closest.q - q), np.cos(closest.q - q))))
        assert deviation < SELF_RECOVERY_TOLERANCE


def test_solve_closest_prefers_the_reference_over_the_first_branch(pose_cases) -> None:
    """Fact 9: continuity is not accidental -- the first branch is often wrong.

    ``solve_closest`` must return the reference configuration even when the first
    entry of ``solve()`` is a different, much further branch.  The test first
    finds such a case (measured: 38 of the 60 shared poses have the target among
    the later branches), then asserts both halves: the first branch really is far
    away, and ``solve_closest`` still returns the reference.  Without the first
    assertion the test could pass vacuously on a pose whose target happens to be
    the first solution.
    """
    checked = 0
    for q, q7, pose in pose_cases:
        solutions = solver.solve(pose, q7, within_limits_only=True)
        if not solutions:
            continue
        first = float(
            np.max(np.abs(np.arctan2(np.sin(solutions[0].q - q), np.cos(solutions[0].q - q))))
        )
        if first <= 1e-3:
            continue
        closest = solver.solve_closest(pose, q7, q)
        assert closest is not None
        deviation = np.max(np.abs(np.arctan2(np.sin(closest.q - q), np.cos(closest.q - q))))
        assert deviation < SELF_RECOVERY_TOLERANCE
        checked += 1
    assert checked > 0, "the shared sample must contain a pose whose target is not the first branch"


def test_solve_closest_minimises_the_deviation(pose_cases) -> None:
    """The returned solution is the argmin over all solutions, not just a close one.

    Independent of which branch carries the target, the deviation of the result
    must equal the smallest deviation among the candidates.  The tolerance is
    1e-12 rad because the two sides evaluate the same wrapped difference; a
    larger value would hide a genuine preference for the wrong branch.
    """
    for _, q7, pose in pose_cases[:20]:
        reference = np.array([0.2, -0.4, 0.6, -1.2, 0.3, 1.5, -0.8])
        closest = solver.solve_closest(pose, q7, reference)
        assert closest is not None
        best = min(
            float(np.max(np.abs(np.arctan2(np.sin(s.q - reference), np.cos(s.q - reference)))))
            for s in solver.solve(pose, q7, within_limits_only=True)
        )
        observed = float(
            np.max(np.abs(np.arctan2(np.sin(closest.q - reference), np.cos(closest.q - reference))))
        )
        assert observed == pytest.approx(best, abs=1e-12)


def test_solutions_from_different_joint_seven_values_differ(pose_cases) -> None:
    """Joint 7 parameterises the redundancy: a different ``q7`` is a different problem.

    Solving a pose at a joint-7 value other than the one it was built with may
    legitimately fail (that pose may not be reachable with the wrist at that
    angle), and the solver is allowed to return fewer or different branches --
    but every solution it does return must have joint 7 equal to the requested
    value, and none of them may be the configuration the pose was built from.
    This guards the parameterisation itself: a solver that ignored its ``q7``
    argument would still pass most of the other tests in this file.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    checked = 0
    for q, q7, pose in pose_cases[:10]:
        for offset in (-0.5, 0.5):
            other = q7 + offset
            if not (lower[6] < other < upper[6]):
                continue
            for solution in solver.solve(pose, other, within_limits_only=False):
                assert solution.q[6] == pytest.approx(other, abs=1e-12)
                deviation = float(
                    np.max(np.abs(np.arctan2(np.sin(solution.q - q), np.cos(solution.q - q))))
                )
                # Joint 7 alone already differs by 0.5 rad, so no solution can
                # coincide with the original configuration.
                assert deviation > 0.4
                checked += 1
    assert checked > 0, "at least one sampled pose must be solvable at a neighbouring q7"
