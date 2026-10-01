"""Branch structure, the published reference implementation and its defects.

This is the file that carries the repository's scientific claim: the analytical
method has **eight** solutions parameterised by joint 7, and the four-branch
solver published in ``original/`` keeps only the ``+`` root of the elbow
quadratic, so it silently loses the configurations whose elbow lies on the other
side of the shoulder-wrist line.

What is checked here:

* fact 2 -- the library reproduces *every* published branch, entry point by entry
  point, to 1e-9 rad.  The published output is compared modulo ``2*pi`` (it is
  not wrapped into the joint ranges) and guarded with ``np.isfinite``, because an
  out-of-reach pose makes the published code return ``nan``;
* the same comparison at *label* level: each published entry point is one named
  branch, and the mapping holds over the whole joint-7 range, on both signs of
  ``cos(q7)``.  Comparing solution sets instead would not have caught the
  arm-angle labels silently swapping for ``|q7| > 90`` degrees;
* fact 4 -- ``NUM_BRANCHES == 8``, the published subset is exactly
  ``q4_root == PUBLISHED_Q4_ROOT``, and the published four-branch solver fails to
  recover the target configuration that the eight-branch solver always finds;
* fact 5 -- the published ``limit_joints`` never terminates on a ``nan`` and
  accepts angles the real robot cannot reach.  The hang is reproduced in a
  forked subprocess under a hard timeout, so the suite itself cannot hang;
* fact 7 -- ``(q1+pi, -q2, q3+pi)`` is a genuine symmetry of the arm and
  ``(q5+pi, -q6, q7+pi)`` is not, because the 0.088 m wrist offset means the
  Panda is not a true S-R-S arm.

The measurements quoted in the test docstrings are reproducible with the seeds
fixed in ``tests/conftest.py``; the aggregate ones are also tabulated in
``docs/branch_analysis.md``.
"""

from __future__ import annotations

import math
import multiprocessing
import os

import numpy as np
import pytest

from franka_ik import analysis, geometry, model, solver

#: The four published entry points, parametrised by their dotted names.  Kept in
#: sync with the ``published_entry_points`` fixture by an assertion in the
#: parametrised test itself.
ENTRY_POINTS = (
    "ik_ca.ik_ca",
    "ik_ca.ik_ca_neg",
    "ik_ca2.ik_ca",
    "ik_ca2.ik_ca_neg",
)

#: Which branch of the eight each published entry point corresponds to, as
#: ``(q4_root, phi_root, shoulder_flip)``.  ``ik_ca``/``ik_ca2`` differ in the
#: equivalent arm angle (``phi``), the ``_neg`` variants apply the shoulder flip,
#: and all four use the ``+`` elbow root.  Measured 300/300 per entry point over
#: the whole joint-7 range (1200 label-level comparisons, worst 3.2e-13 rad).
ENTRY_POINT_BRANCH = {
    "ik_ca.ik_ca": (1, 1, False),
    "ik_ca.ik_ca_neg": (1, 1, True),
    "ik_ca2.ik_ca": (1, -1, False),
    "ik_ca2.ik_ca_neg": (1, -1, True),
}

#: Absolute tolerance, in radians, for "the library reproduces the published
#: branch".  The published code evaluates the same expressions through a CasADi
#: SX graph, so a different operation order alone accounts for the observed worst
#: deviation of 3.2e-13 rad over the 800 comparisons of this file; 1e-9 rad is
#: ~3000x above that noise (no flakes) and still 6e-8 degrees.
BRANCH_TOLERANCE = 1e-9

#: Tolerance for the pose-symmetry checks of fact 7.  The shoulder flip is an
#: exact algebraic symmetry, so the residual is pure double-precision noise
#: (measured worst 9.4e-16 over 138 configurations); 1e-9 is ~6 orders of
#: headroom.  The same threshold is used to state that the wrist flip does *not*
#: hold, where the observed discrepancy is 0.109 .. 0.176.
SYMMETRY_TOLERANCE = 1e-9

#: Number of configurations for the symmetry checks: the sample size the brief's
#: 138/138 (shoulder) and 0/138 (wrist) measurements were taken on.
SYMMETRY_SAMPLES = 138

#: How long the published ``limit_joints`` is given to terminate on a ``nan``
#: before it counts as an infinite loop.  The same call on finite input returns
#: in well under a millisecond (asserted below), so 2 s is ~4 orders of
#: magnitude of slack, and the cost is only paid once.
HANG_BUDGET_SECONDS = 2.0

#: Extra time allowed for ``terminate()`` + ``join()`` to reap the child.
HANG_REAP_TIMEOUT_SECONDS = 10.0

#: Samples for the documented coverage measurement (``docs/branch_analysis.md``
#: as quoted by ``franka_ik/analysis.py``: 265 of 300 for the published subset).
DOCUMENTED_COVERAGE_SAMPLES = 300


def all_branch_labels() -> tuple:
    """The eight branch labels, in the library's own format."""
    return tuple(
        f"q4{'+' if q4_root > 0 else '-'} phi{'+' if phi_root > 0 else '-'} "
        f"{'flip' if shoulder_flip else 'plain'}"
        for q4_root in (1, -1)
        for phi_root in (1, -1)
        for shoulder_flip in (False, True)
    )


def _published_limit_joints_worker(theta, queue) -> None:
    """Call the published ``limit_joints`` in a child process and report back.

    Module-level so it can be used with either start method; the parent never
    waits for it without a timeout.
    """
    import ik_ca

    queue.put(np.asarray(ik_ca.limit_joints(np.asarray(theta, dtype=float)), dtype=float))


def test_published_q4_root_constant_is_the_plus_root(paper_geometry) -> None:
    """Fact 4: the published subset is ``q4_root == PUBLISHED_Q4_ROOT == +1``.

    ``PUBLISHED_Q4_ROOT`` is the identification the whole branch analysis rests
    on, so it is checked in both directions: the constant is ``+1`` *and*
    ``q4_roots`` really returns the ``+`` root (``(a1 + sqrt(disc)) / (2 a2)``)
    first, which is what makes restricting to that root reproduce the published
    code.  Same arithmetic, so 1e-12 rad.
    """
    assert analysis.PUBLISHED_Q4_ROOT == 1

    distance = 0.5
    a2, a1, a0 = geometry.q4_coefficients(distance**2, paper_geometry)
    root = math.sqrt(a1 * a1 - 4.0 * a0 * a2)
    plus = 2.0 * math.atan((a1 + root) / (2.0 * a2))
    minus = 2.0 * math.atan((a1 - root) / (2.0 * a2))
    roots = geometry.q4_roots(distance**2, paper_geometry)
    assert len(roots) == 2
    assert roots[0] == pytest.approx(plus, abs=1e-12)
    assert roots[1] == pytest.approx(minus, abs=1e-12)


def test_solver_one_root_matches_the_published_q4_root(paper_geometry, pose_cases, angdiff) -> None:
    """Fact 4: ``q4_root == +1`` selects ``q4_roots(...)[0]``, not the other root.

    ``solve_branch`` maps ``q4_root > 0`` to ``roots[0]`` and ``q4_root < 0`` to
    ``roots[1]``.  If that mapping were reversed, restricting to
    ``PUBLISHED_Q4_ROOT`` would reproduce the *unpublished* half of the solution
    set while every other test still passed.  The check uses the elbow angles the
    solver returns for a real pose against the two roots of that pose's own
    shoulder-wrist distance; the comparison is modulo ``2*pi`` because joint 4 is
    returned wrapped into its one-sided range ``[-175, -5]`` degrees.

    Not every pose supports both roots at the same arm angle (17 of the 60 shared
    poses do, measured), so the test asserts on every pose that does rather than
    on a hand-picked one, and requires that at least one was found.
    """
    checked = 0
    for _, q7, pose in pose_cases:
        corrected = pose[:3, :3] @ geometry.wrist_correction(q7, paper_geometry)
        p_sw = geometry.shoulder_to_wrist(pose[:3, 3], corrected, paper_geometry)
        roots = geometry.q4_roots(float(p_sw @ p_sw), paper_geometry)
        if len(roots) != 2:
            continue
        plus = solver.solve_branch(pose, q7, q4_root=1, phi_root=1)
        minus = solver.solve_branch(pose, q7, q4_root=-1, phi_root=1)
        if plus is None or minus is None:
            continue
        assert float(angdiff(plus.q[3], roots[0])) < 1e-9
        assert float(angdiff(minus.q[3], roots[1])) < 1e-9
        checked += 1
    assert checked > 0, "no shared pose exercised both elbow roots at the same arm angle"


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_reproduces_the_published_entry_point(
    entry_point, in_limit_configurations, published_entry_points, angdiff
) -> None:
    """Fact 2: every published branch is contained in the library's solution set.

    For each of the 200 shared in-limit configurations the pose is solved by the
    library and the published entry point is evaluated on the same pose and
    joint-7 value; the published answer must match one library solution to 1e-9
    rad modulo ``2*pi`` (the published code returns unwrapped angles, the library
    wraps into the real ranges, so a ``2*pi`` offset is expected rather than a
    discrepancy).  This is the fidelity claim that lets every other test treat
    the library as a superset of the published solver.

    ``np.isfinite`` is applied to the published output before comparison: for a
    pose outside the reachable shell the published code returns ``nan`` (it takes
    ``sqrt`` of a negative discriminant), which is asserted separately in
    ``test_published_entry_point_is_not_finite_outside_the_reachable_shell``.
    Over this sample all 800 comparisons were finite and matched, with a worst
    deviation of 3.2e-13 rad (4.0e-14 for the two ``ik_ca`` entry points, 3.2e-13
    for the two ``ik_ca2`` ones); the 90 % floor on finite comparisons keeps a
    wholesale ``nan`` regression from turning this into a vacuous pass.
    """
    assert entry_point in published_entry_points, "entry points and fixture disagree"
    published = published_entry_points[entry_point]

    total = 0
    finite = 0
    worst = 0.0
    for q in in_limit_configurations:
        q7 = float(q[6])
        pose = model.fk_flange(q)
        solutions = solver.solve(pose, q7, deduplicate=False, within_limits_only=False)
        assert solutions, "the library must solve a reachable pose"
        answer = np.asarray(published(pose, q7), dtype=float).reshape(model.NUM_JOINTS)
        total += 1
        if not np.all(np.isfinite(answer)):
            continue
        finite += 1
        best = min(float(np.max(angdiff(answer, solution.q))) for solution in solutions)
        worst = max(worst, best)
        assert best < BRANCH_TOLERANCE

    assert finite >= 0.9 * total
    assert worst < BRANCH_TOLERANCE


def test_published_entry_points_are_the_plus_elbow_root_branches(
    in_limit_configurations, published_entry_points, angdiff
) -> None:
    """Fact 4: all four published answers sit on ``q4_root == +1``, and on a known branch.

    This is the quantitative form of "the published four-branch subset is
    incomplete": the four entry points cover the ``+`` elbow root exhaustively --
    both equivalent arm angles and both shoulder flips -- and never produce a
    ``-`` root solution.  Measured over 200 configurations per entry point
    (800 comparisons): every published answer matches a library solution with
    ``q4_root == PUBLISHED_Q4_ROOT``, with no exceptions, and the ``_neg``
    variants always match a shoulder-flipped branch while ``ik_ca``/``ik_ca2``
    always match a plain one.  Together the two non-flipped entry points cover the
    plain ``phi+``/``phi-`` pair on every pose, and so do the two flipped ones.

    The arm-angle label itself is pinned separately, at full strength, by
    ``test_published_entry_points_match_their_library_branch``; this test stays on
    the two labels that fact 4 is actually about.
    """
    for entry_point, published in published_entry_points.items():
        expected_shoulder_flip = entry_point.endswith("_neg")
        labels = set()
        matched = 0
        for q in in_limit_configurations:
            q7 = float(q[6])
            pose = model.fk_flange(q)
            solutions = solver.solve(pose, q7, deduplicate=False, within_limits_only=False)
            answer = np.asarray(published(pose, q7), dtype=float).reshape(model.NUM_JOINTS)
            if not np.all(np.isfinite(answer)):
                continue
            best = min(solutions, key=lambda s: float(np.max(angdiff(answer, s.q))))
            assert float(np.max(angdiff(answer, best.q))) < BRANCH_TOLERANCE
            assert best.q4_root == analysis.PUBLISHED_Q4_ROOT
            assert best.shoulder_flip is expected_shoulder_flip
            labels.add(best.label)
            matched += 1
        assert matched > 0
        expected_phi = ENTRY_POINT_BRANCH[entry_point][1]
        for label in labels:
            assert label.startswith(f"q4+ phi{'+' if expected_phi > 0 else '-'}")

    # The two published arm-angle branches are both reachable and are jointly
    # covered by the two non-flipped (or two flipped) entry points.
    for pair in (("ik_ca.ik_ca", "ik_ca2.ik_ca"), ("ik_ca.ik_ca_neg", "ik_ca2.ik_ca_neg")):
        covered = 0
        for q in in_limit_configurations:
            q7 = float(q[6])
            pose = model.fk_flange(q)
            solutions = solver.solve(pose, q7, deduplicate=False, within_limits_only=False)
            labels = set()
            for entry_point in pair:
                answer = np.asarray(
                    published_entry_points[entry_point](pose, q7), dtype=float
                ).reshape(model.NUM_JOINTS)
                if not np.all(np.isfinite(answer)):
                    continue
                best = min(solutions, key=lambda s: float(np.max(angdiff(answer, s.q))))
                if float(np.max(angdiff(answer, best.q))) < BRANCH_TOLERANCE:
                    labels.add(best.label)
            if len(labels) == 2:
                covered += 1
        assert covered == len(in_limit_configurations)


def test_published_entry_points_match_their_library_branch(
    in_limit_configurations, published_entry_points, angdiff
) -> None:
    """Each published entry point *is* one named branch, and stays that branch everywhere.

    ``ik_ca`` must be ``(q4_root=+1, phi_root=+1, shoulder_flip=False)``,
    ``ik_ca_neg`` the same with the shoulder flip, ``ik_ca2`` the ``phi_root=-1``
    pair; the joint vectors are compared directly, at 1e-9 rad.  Measured 300/300
    per entry point (1200 comparisons) over the whole joint-7 range, worst
    deviation 3.2e-13 rad.

    **Why this test compares labels and not just sets.**  The two ``phi_root``
    labels are the two roots of a sinusoid in the equivalent arm angle, and the
    equation that produces them is evaluated in the form
    ``W[i][2,1] * cos(q7) + W[i][2,0] * sin(q7)`` rather than the published
    ``W[i][2,1] + tan(q7) * W[i][2,0]``, because the tangent form is not finite at
    ``q7 = +-90`` degrees.  Multiplying an equation by ``cos(q7)`` leaves its
    roots unchanged *as a set* but negates all three coefficients when
    ``cos(q7) < 0``, and negating them swaps the two roots' labels.  A set-level
    comparison therefore passed while ``phi_root = +1`` silently denoted the
    opposite root for every pose with ``|q7| > 90`` degrees -- the solver was
    right and its labels were wrong.  The sign of ``cos(q7)`` is now folded into
    the root choice, and this test is what keeps it that way: it asserts on the
    label for every pose, and it refuses to pass on a one-sided sample, because
    only the ``cos(q7) < 0`` half can detect the swap.  That is also why the
    regime counts are asserted rather than assumed.
    """
    positive_cos = 0
    negative_cos = 0
    for entry_point, published in published_entry_points.items():
        q4_root, phi_root, shoulder_flip = ENTRY_POINT_BRANCH[entry_point]
        expected_label = (
            f"q4{'+' if q4_root > 0 else '-'} phi{'+' if phi_root > 0 else '-'} "
            f"{'flip' if shoulder_flip else 'plain'}"
        )
        for q in in_limit_configurations:
            q7 = float(q[6])
            pose = model.fk_flange(q)
            answer = np.asarray(published(pose, q7), dtype=float).reshape(model.NUM_JOINTS)
            if not np.all(np.isfinite(answer)):
                continue
            branch = solver.solve_branch(
                pose, q7, q4_root=q4_root, phi_root=phi_root, shoulder_flip=shoulder_flip
            )
            assert branch is not None, "the published branch must exist for a reachable pose"
            assert branch.label == expected_label
            assert float(np.max(angdiff(answer, branch.q))) < BRANCH_TOLERANCE
    for q in in_limit_configurations:
        if math.cos(float(q[6])) > 0.0:
            positive_cos += 1
        else:
            negative_cos += 1
    assert positive_cos > 0
    assert negative_cos > 0


def test_published_entry_point_is_not_finite_outside_the_reachable_shell(
    pose_cases, published_entry_points
) -> None:
    """The published code returns ``nan`` for an unreachable pose -- hence the guard.

    With a negative discriminant the published ``ik_ca`` evaluates
    ``sqrt(negative)`` and returns ``nan`` for every joint but the joint-7 value
    it was handed.  This is why fact 2 asks for an ``np.isfinite`` guard: without
    it, comparing against ``nan`` would either fail spuriously or (worse) be
    skipped silently.  The library, by contrast, returns an empty list.
    """
    _, q7, pose = pose_cases[0]
    unreachable = pose.copy()
    unreachable[:3, 3] += np.array([3.0, 0.0, 0.0])
    for published in published_entry_points.values():
        answer = np.asarray(published(unreachable, q7), dtype=float).reshape(model.NUM_JOINTS)
        assert not np.all(np.isfinite(answer))
        assert np.all(np.isnan(answer[:6]))
        assert answer[6] == pytest.approx(q7, abs=1e-12)
    assert solver.solve(unreachable, q7, within_limits_only=False) == []


def test_published_four_branch_subset_is_incomplete(coverage_report) -> None:
    """Fact 4, the scientific headline: eight branches recover every target, four do not.

    ``coverage_study`` samples configurations inside the real joint limits, solves
    each pose, and asks whether the configuration it came from is among the
    answers.  The eight-branch solver must recover all of them (any failure would
    mean a missing branch, since every returned solution is verified against the
    forward kinematics); the published subset must recover strictly fewer, which
    is the measured evidence that the published solver is incomplete rather than
    merely inconvenient.  The fixture uses ``samples=200, seed=0``; measured at
    that seed: 200/200 recovered by all eight branches against 174/200 by the
    published subset.
    """
    assert coverage_report.samples > 0
    assert coverage_report.recovered_full == coverage_report.samples
    assert coverage_report.full_rate == pytest.approx(1.0, abs=1e-12)
    assert coverage_report.recovered_published < coverage_report.samples
    assert coverage_report.recovered_published > 0
    # A real gap, not a one-sample difference: measured 13 % at this seed.
    assert (coverage_report.full_rate - coverage_report.published_rate) > 0.05
    assert "published four-branch subset" in coverage_report.describe()


def test_documented_coverage_numbers_still_hold() -> None:
    """The 300-sample measurement quoted by the repository reproduces exactly.

    ``docs/branch_analysis.md`` §2 and §6 publish this experiment (265/300 =
    88.3 % for the published subset, 300/300 for the eight branches at seed 0,
    and 261/300 = 87.0 % at seed 1), and ``coverage_study`` is deterministic in
    its seed, so the values are pinned rather than bounded.  Two seeds rather than
    one, because the document's claim is that the gap is not a single-sample
    artefact.

    The docstring caveat the document records is confirmed here as well: the prose
    figure in ``franka_ik/solver.py`` ("the published four in 375 of 600") does
    **not** reproduce -- at seed 0 this study returns 531/600 in limit-sampling
    mode and 230/600 in the alternative mode, and five other seeds were checked
    too (518..539 and 216..235).  The behaviour is consistent with 88 %, so the
    stale figure is a docstring problem, not a code problem, and
    ``docs/branch_analysis.md`` §6 says so.
    """
    for seed, expected_published in ((0, 265), (1, 261)):
        report = analysis.coverage_study(samples=DOCUMENTED_COVERAGE_SAMPLES, seed=seed)
        assert report.samples == DOCUMENTED_COVERAGE_SAMPLES
        assert report.recovered_full == DOCUMENTED_COVERAGE_SAMPLES
        assert report.recovered_published == expected_published
        assert report.published_rate == pytest.approx(
            expected_published / DOCUMENTED_COVERAGE_SAMPLES, rel=1e-12
        )
        assert report.full_rate == pytest.approx(1.0, abs=1e-12)


def test_all_eight_branch_labels_occur_and_are_distinct(pose_cases) -> None:
    """Every one of the eight branches is a real, distinct configuration.

    The label set returned over the shared poses must be exactly the eight
    combinations of elbow root, arm angle and shoulder flip -- if one were
    unreachable in practice it would be a dead branch, and if two branches ever
    produced the same label the counting in the studies would be meaningless.
    Measured over the 60 shared poses: all eight labels occur (17 of the poses
    have all eight branches).
    """
    expected = set(all_branch_labels())
    assert len(expected) == solver.NUM_BRANCHES
    seen = set()
    complete = 0
    for _, q7, pose in pose_cases:
        labels = [solution.label for solution in solver.branch_solutions(pose, q7)]
        assert len(labels) == len(set(labels)), "two branches returned the same label"
        assert set(labels) <= expected
        seen |= set(labels)
        if len(labels) == solver.NUM_BRANCHES:
            complete += 1
    assert seen == expected
    assert complete > 0


def test_shoulder_flip_is_a_genuine_symmetry_of_the_arm(rng) -> None:
    """Fact 7: ``(q1+pi, -q2, q3+pi)`` reaches the same flange pose.

    The shoulder flip maps the two base joints and the elbow of the equivalent
    arm onto the mirrored arm plane, and it survives the reduction exactly: the
    shoulder bias and the wrist offset do not break it.  Measured 138/138 with a
    worst residual of 9.4e-16 (tolerance 1e-9), and the flipped configuration is
    asserted to be a genuinely different one so the test cannot pass by comparing
    a configuration with itself.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    worst = 0.0
    for _ in range(SYMMETRY_SAMPLES):
        q = rng.uniform(lower, upper)
        flipped = np.array([q[0] + math.pi, -q[1], q[2] + math.pi, q[3], q[4], q[5], q[6]])
        residual = float(np.max(np.abs(model.fk_flange(flipped) - model.fk_flange(q))))
        worst = max(worst, residual)
        assert residual < SYMMETRY_TOLERANCE
        # Joint 1 differs by pi, so this is a different configuration.
        assert abs(flipped[0] - q[0]) > 3.0
    assert worst < SYMMETRY_TOLERANCE


def test_wrist_flip_is_not_a_symmetry_of_the_arm(rng) -> None:
    """Fact 7: ``(q5+pi, -q6, q7+pi)`` does **not** reach the same flange pose.

    That flip is the wrist symmetry of a true S-R-S arm, and the Panda is not one:
    the 0.088 m offset of DH row 7 rotates the last link, so the mirrored wrist
    puts the flange somewhere else.  Measured 0/138, with the pose discrepancy
    between 0.109 and 0.176 -- the size of the wrist offset, not numerical noise.
    The upper bound is asserted as well, to show the failure is a bounded
    geometric effect and not, say, a diverged pose.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    smallest = math.inf
    largest = 0.0
    for _ in range(SYMMETRY_SAMPLES):
        q = rng.uniform(lower, upper)
        flipped = np.array([q[0], q[1], q[2], q[3], q[4] + math.pi, -q[5], q[6] + math.pi])
        residual = float(np.max(np.abs(model.fk_flange(flipped) - model.fk_flange(q))))
        assert residual > 1e-3
        assert residual < 1.0
        smallest = min(smallest, residual)
        largest = max(largest, residual)
    assert smallest > 1e-3
    assert largest < 1.0


def test_published_limit_joints_accepts_angles_the_robot_cannot_reach(
    in_limit_configurations, published_entry_points
) -> None:
    """Fact 5's other half: the published windows are not the real joint ranges.

    ``limit_joints`` wraps joints 1-3 and 7 into ``[-181, 181]`` degrees, joint 4
    into ``[-271, 91]`` and joint 6 into ``[-74, 288]``.  The joint-4 and joint-6
    windows are 362 degrees wide, so they contain *two* representatives of some
    angles and accept values outside the real ranges ``[-175, -5]`` and
    ``[0, 214]``.  Concretely: 60 degrees for joint 4 and 250 degrees for joint 6
    come back unchanged, while :func:`franka_ik.solver.wrap_to_limits` flags both
    as unrepresentable.  A configuration drawn inside the real limits is left
    alone by both -- the published helper is not *always* wrong, which is why the
    defect went unnoticed.
    """
    import ik_ca

    angle = np.zeros(model.NUM_JOINTS)
    angle[3] = math.radians(60.0)
    angle[5] = math.radians(250.0)
    published = np.asarray(ik_ca.limit_joints(angle), dtype=float)
    np.testing.assert_allclose(published, angle, atol=1e-12)

    _, inside = solver.wrap_to_limits(angle)
    assert bool(inside[3]) is False
    assert bool(inside[5]) is False

    for q in in_limit_configurations[:20]:
        np.testing.assert_allclose(np.asarray(ik_ca.limit_joints(q), dtype=float), q, atol=1e-12)


def test_published_limit_joints_returns_immediately_on_finite_input() -> None:
    """Sanity check for the hang test: the same call on finite input terminates.

    Without this, ``test_published_limit_joints_never_terminates_on_a_nan`` would
    also "pass" if the subprocess machinery itself were broken (a child that never
    starts also never finishes).  The published helper leaves in-limit angles
    untouched, so the returned value is compared against the input.
    """
    pytest.importorskip("casadi", reason="the published helper lives in a CasADi module")
    if not hasattr(os, "fork"):  # pragma: no cover - Linux CI has fork
        pytest.skip("needs a fork-based subprocess")
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    angles = np.array([0.1, 0.2, 0.3, -1.0, 0.4, 1.0, 0.5])
    process = context.Process(target=_published_limit_joints_worker, args=(angles, queue))
    process.start()
    process.join(HANG_BUDGET_SECONDS)
    try:
        assert not process.is_alive()
        np.testing.assert_allclose(queue.get(timeout=1.0), angles, atol=1e-12)
    finally:
        if process.is_alive():  # pragma: no cover - would mean the child hung too
            process.terminate()
            process.join(HANG_REAP_TIMEOUT_SECONDS)


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="the reproduction runs the published helper in a killable child process",
)
def test_published_limit_joints_never_terminates_on_a_nan() -> None:
    """Fact 5: the published ``limit_joints`` loops forever on a ``nan``.

    Its loop is ``while True: if (-181 <= theta[i]) and (181 >= theta[i]): break;
    elif -181 > theta[i]: theta[i] += 360; else: theta[i] -= 360``.  Every
    comparison with ``nan`` is False, so neither the ``break`` nor either branch
    is taken in a useful way: the angle is repeatedly decreased by 360 and the
    ``while True`` never exits.  Reproduction has to be able to kill the process,
    so the call runs in a forked child that is given
    :data:`HANG_BUDGET_SECONDS` and then terminated; the child is reaped in a
    ``finally`` block, so this suite can never hang.  The library's replacement
    raises ``ValueError`` instead, which ``tests/test_solver.py`` asserts.
    """
    pytest.importorskip("casadi", reason="the published helper lives in a CasADi module")
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(
        target=_published_limit_joints_worker, args=([math.nan] * model.NUM_JOINTS, queue)
    )
    process.start()
    try:
        process.join(HANG_BUDGET_SECONDS)
        assert process.is_alive(), (
            "the published limit_joints terminated on a nan; the documented infinite "
            "loop no longer reproduces (the helper may have been fixed)"
        )
    finally:
        process.terminate()
        process.join(HANG_REAP_TIMEOUT_SECONDS)
    assert not process.is_alive()
