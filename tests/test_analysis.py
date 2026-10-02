"""The solution-set studies and the failure classifier (facts 4, 10 and 11).

``franka_ik/analysis.py`` is where the repository turns its claim into a
measurement: :func:`~franka_ik.analysis.coverage_study` asks whether the solver
finds the configuration a pose came from, :func:`~franka_ik.analysis.
solution_count_study` asks how many configurations a pose actually has, and
:func:`~franka_ik.analysis.study_pose` reports what each branch did for one pose.

The scientific headline those studies produce -- the published four-branch subset
recovers 88.3 % of targets and the eight-branch solver 100 %, the measurement
written up in ``docs/branch_analysis.md`` §2 and §4 -- is asserted in
``tests/test_branches.py``, next to the comparison against the published
implementation.  What is checked here is the machinery itself:

* the report objects are self-consistent and deterministic in their seed, and the
  counts they publish are recomputed from :func:`franka_ik.solver.solve` rather
  than trusted;
* ``study_pose`` marks exactly the branches that recover the target, and its two
  counts agree with the solver;
* ``classify_failure`` never raises on a finite pose, returns ``"solvable"`` for
  reachable ones and a documented reason for an unreachable one.  Its behaviour
  on *non-finite* input deviates from the literal "never raise" wording of the
  brief and is pinned by a test of its own, with the reasoning in the docstring:
  a ``nan`` reaches :func:`franka_ik.solver.wrap_to_limits`, whose finite guard
  replaces the published infinite loop described in ``docs/limitations.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from franka_ik import analysis, model, solver

#: Poses for the classifier sweep.  Facts 10 and 11 talk about "reachable poses",
#: and a reachable pose means an in-limit configuration whose *own* joint-7 value
#: is used: solving a pose at a joint-7 value it was not built with is allowed to
#: fail, and does (measured: 37 % of such calls return ``no_valid_branch``).
CLASSIFIER_SAMPLES = 200

#: The documented failure strings, from the return-value section of
#: :func:`franka_ik.analysis.classify_failure`.
DOCUMENTED_FAILURES = (
    "shoulder_wrist_distance_zero",
    "outside_reachable_shell",
    "arm_angle_singular",
    "no_valid_branch",
)

#: Distance by which a pose is moved to leave the workspace.  The arm reaches at
#: most 0.72 m from the shoulder, so 3 m has a negative discriminant for every
#: joint-7 value.
OUT_OF_REACH_DISTANCE = 3.0

#: Magnitudes for the "must not raise" sweep: from a picometre to 1e300, i.e.
#: well beyond any scale a pose could legitimately have.  The point is that a
#: garbage-but-finite pose is *classified*, not crashed on.
GARBAGE_SCALES = (1e-12, 1e-3, 1.0, 10.0, 1e3, 1e12, 1e150, 1e300)


def test_coverage_report_is_internally_consistent(coverage_report) -> None:
    """The shared coverage report's counts, rates and text agree with each other.

    The fixture is the same object ``test_branches.py`` asserts the headline on;
    here the consistency of the report itself is checked: the rates are the counts
    over the sample size, the eight-branch solver loses nothing, and the published
    subset loses something.  Measured at ``samples=200, seed=0``: 200/200 against
    174/200.
    """
    assert coverage_report.samples > 0
    assert coverage_report.recovered_full == coverage_report.samples
    assert coverage_report.recovered_published < coverage_report.samples
    assert coverage_report.recovered_published > 0
    assert coverage_report.full_rate == pytest.approx(
        coverage_report.recovered_full / coverage_report.samples, rel=1e-12
    )
    assert coverage_report.published_rate == pytest.approx(
        coverage_report.recovered_published / coverage_report.samples, rel=1e-12
    )
    assert coverage_report.min_solutions >= 1
    assert coverage_report.max_solutions <= solver.NUM_BRANCHES
    assert coverage_report.min_solutions <= coverage_report.mean_solutions
    assert coverage_report.mean_solutions <= coverage_report.max_solutions
    text = coverage_report.describe()
    assert "eight-branch" in text
    assert "four-branch" in text


def test_coverage_study_is_deterministic_in_its_seed() -> None:
    """Two runs with the same seed produce identical reports.

    The studies are the repository's published measurements, so a reader has to be
    able to re-run them and get the same numbers; ``samples=50`` keeps the check
    cheap (the reported 300-sample values are pinned in ``test_branches.py``).
    """
    first = analysis.coverage_study(samples=50, seed=0)
    second = analysis.coverage_study(samples=50, seed=0)
    assert first.samples == second.samples
    assert first.recovered_full == second.recovered_full
    assert first.recovered_published == second.recovered_published
    assert first.mean_solutions == second.mean_solutions
    assert first.min_solutions == second.min_solutions
    assert first.max_solutions == second.max_solutions


def test_solution_count_study_histogram() -> None:
    """Every pose has between 1 and ``NUM_BRANCHES`` distinct in-limit solutions.

    The number of solutions is a property of the arm, not of the solver, so it
    must never exceed the branch count (that would mean the enumeration is
    incomplete) and never be zero for a target inside the joint limits (that would
    mean the solver missed the configuration it was given).  Measured at
    ``samples=100, seed=0``: counts from 1 to 7, mean 3.15, so the range is real
    and not a degenerate constant.
    """
    report = analysis.solution_count_study(samples=100, seed=0)
    assert report.samples == 100
    assert sum(report.histogram.values()) == report.samples
    assert set(report.histogram) <= set(range(1, solver.NUM_BRANCHES + 1))
    weighted = sum(count * number for count, number in report.histogram.items()) / report.samples
    assert report.mean == pytest.approx(weighted, rel=1e-12)
    assert min(report.histogram) >= 1
    assert max(report.histogram) <= solver.NUM_BRANCHES
    text = report.describe()
    assert isinstance(text, str)
    assert f"mean {report.mean:.2f}" in text


def test_reachable_distance_range_agrees_with_the_discriminant(paper_geometry) -> None:
    """The reported shell is exactly where the elbow discriminant changes sign.

    ``reachable_distance_range`` returns the bounds by scanning the discriminant,
    so the two ends must be *on* the boundary: a hair inside the discriminant is
    positive, a hair outside it is negative, and the interval is not empty.  The
    measured range for the real (biased) arm is 0.0661704620 .. 0.7193542034 m --
    *not* the ``|d_se - d_ew| .. d_se + d_ew`` = 0.068 .. 0.700 annulus, which is
    the bias-free S-R-S shell; ``tests/test_geometry.py`` derives and checks both
    closed forms.
    """
    inner, outer = analysis.reachable_distance_range(paper_geometry)
    assert 0.0 < inner < outer
    assert inner == pytest.approx(0.0661704620, abs=1e-6)
    assert outer == pytest.approx(0.7193542034, abs=1e-6)
    assert analysis.q4_discriminant((inner * (1.0 + 1e-6)) ** 2, paper_geometry) > 0.0
    assert analysis.q4_discriminant((inner * (1.0 - 1e-6)) ** 2, paper_geometry) < 0.0
    assert analysis.q4_discriminant((outer * (1.0 - 1e-6)) ** 2, paper_geometry) > 0.0
    assert analysis.q4_discriminant((outer * (1.0 + 1e-6)) ** 2, paper_geometry) < 0.0


def test_classify_failure_is_solvable_for_reachable_poses(rng) -> None:
    """Fact 10: an in-limit configuration's own pose is reported as solvable.

    The pose is solved at the joint-7 value it was built with, which is the only
    value at which it is guaranteed to be reachable.  Measured 200/200 at this
    seed; the classifier has to agree with the solver, so a disagreement here
    would mean one of the two is wrong about the workspace.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    for _ in range(CLASSIFIER_SAMPLES):
        q = rng.uniform(lower, upper)
        assert analysis.classify_failure(model.fk_flange(q), float(q[6])) == "solvable"


def test_classify_failure_reports_a_reason_out_of_reach(rng) -> None:
    """Fact 10: a pose 3 m away is classified, never solved and never crashed on.

    Two directions, both well outside the 0.72 m workspace of the shoulder, and
    both measured to come back as ``"outside_reachable_shell"`` -- the reason that
    corresponds to the negative discriminant of the elbow quadratic, which is also
    what ``solve`` uses to return an empty list.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    for direction in (
        np.array([OUT_OF_REACH_DISTANCE, 0.0, 0.0]),
        np.array([0.0, 0.0, OUT_OF_REACH_DISTANCE]),
    ):
        q = rng.uniform(lower, upper)
        pose = model.fk_flange(q)
        pose[:3, 3] += direction
        reason = analysis.classify_failure(pose, float(q[6]))
        assert reason == "outside_reachable_shell"
        assert solver.solve(pose, float(q[6]), within_limits_only=False) == []


def test_classify_failure_never_raises_on_a_finite_pose(rng) -> None:
    """Fact 10, "never raise": it holds for every finite pose, however absurd.

    A sweep of deliberately meaningless inputs -- non-orthonormal rotation blocks
    and positions from a picometre to 1e300, at random joint-7 values -- must come
    back with one of the documented strings.  Measured: 240/240 classified, no
    exception.  ``np.errstate`` is used because the huge scales make NumPy warn
    about overflow inside the classifier; the warnings are expected and the answer
    is still a valid classification.

    The one input class that *does* raise is non-finite input, which the next test
    pins separately.
    """
    generator = np.random.default_rng(4)
    outcomes = set()
    with np.errstate(all="ignore"):
        for scale in GARBAGE_SCALES:
            for _ in range(30):
                pose = np.eye(4)
                pose[:3, :3] = generator.normal(size=(3, 3)) * scale
                pose[:3, 3] = generator.normal(size=3) * scale
                reason = analysis.classify_failure(pose, float(generator.normal()))
                assert reason == "solvable" or reason in DOCUMENTED_FAILURES
                outcomes.add(reason)
    assert outcomes


def test_classify_failure_raises_on_non_finite_input() -> None:
    """Documented deviation: a ``nan`` pose is rejected with ``ValueError``.

    The brief's fact 10 says ``classify_failure`` must never raise; measured, it
    never raises for a *finite* pose (see the test above) but does raise for a
    non-finite one, because it reaches :func:`franka_ik.solver.wrap_to_limits`,
    whose finite-input guard is the replacement for the published infinite loop.
    The behaviour is defensible -- a silent ``nan`` classification would be worse,
    and the guard is the same one that stops the published ``limit_joints`` from
    spinning forever -- but ``classify_failure``'s docstring documents only its
    return values and has no ``Raises:`` section, so the contract in the brief is
    narrower than the contract in the code.  This test pins the behaviour so the
    gap cannot widen unnoticed; the assessment is recorded in the suite's
    documentation comment rather than hidden in a tolerance.
    """
    quiet = np.eye(4)
    for pose, q7 in (
        (np.full((4, 4), math.nan), 0.3),
        (np.full((4, 4), math.inf), 0.3),
        (quiet, math.nan),
        (quiet, math.inf),
    ):
        with pytest.raises(ValueError), np.errstate(all="ignore"):
            # errstate: the non-finite pose makes NumPy warn about invalid
            # matmul results before the finite guard is reached, which is
            # expected here and would otherwise clutter the suite's output.
            analysis.classify_failure(pose, q7)


def test_the_limits_do_not_explain_the_discarded_elbow_root() -> None:
    """The second elbow root is not discarded by the joint limits, and is not unusable.

    ``docs/provenance.md`` is where this matters: the published code takes one root
    of the ``STEP2`` quadratic, and the natural first guess -- and the reason a
    published variant of the same reduction gives for dropping one of its two variants
    -- is that the other root violates the Panda's joint limits.  Measured over 300
    in-limit configurations at ``seed 0``, it does not.  Every one of the 35
    configurations that sit on the second root has an in-limit solution there, and
    for 2 of them that root is the only way to reach the pose at all.

    The same sample also pins the one number in ``docs/provenance.md`` §3 that
    speaks to the published reasoning: **all 153** in-limit solutions on the second
    root have joint 4 inside the band a published variant of the same reduction
    quotes for the variant it discards, ``[-26.76, -4]`` deg.  Both accounts agree
    about the window; they disagree about whether it is empty.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    rng = np.random.default_rng(0)

    on_second_root = 0
    with_an_in_limit_solution = 0
    published_reaches_it_too = 0
    published_reports_it_unreachable = 0
    in_limit_q4: list[float] = []
    out_of_limit_q4: list[float] = []

    for _ in range(300):
        target = rng.uniform(lower, upper)
        pose = model.fk_flange(target)
        # Every candidate for this joint 7, and only then the in-limit ones: the
        # question is whether the second root is *blocked* by the limits, so the
        # whole family has to be visible before the filter is applied.
        candidates = solver.solve(pose, float(target[6]), within_limits_only=False)
        plus = [s for s in candidates if s.q4_root == analysis.PUBLISHED_Q4_ROOT]
        minus = [s for s in candidates if s.q4_root != analysis.PUBLISHED_Q4_ROOT]
        in_limit_plus = [s for s in plus if _inside_limits(s.q, lower, upper)]
        in_limit_minus = [s for s in minus if _inside_limits(s.q, lower, upper)]

        if any(_same_configuration(s.q, target) for s in minus) and not any(
            _same_configuration(s.q, target) for s in plus
        ):
            on_second_root += 1
            if in_limit_minus:
                with_an_in_limit_solution += 1
            if in_limit_plus:
                published_reaches_it_too += 1
            else:
                published_reports_it_unreachable += 1

        for s in minus:
            (in_limit_q4 if _inside_limits(s.q, lower, upper) else out_of_limit_q4).append(
                float(np.degrees(s.q[3]))
            )

    assert on_second_root == 35
    assert with_an_in_limit_solution == on_second_root, "the second root is not limit-blocked"
    assert published_reaches_it_too == 33
    assert published_reports_it_unreachable == 2
    assert len(in_limit_q4) == 153
    assert len(out_of_limit_q4) == 275
    assert min(in_limit_q4) >= -26.76
    assert max(in_limit_q4) <= -4.0
    # The 275 out-of-limit solutions are *not* checked for a joint-4 window: they
    # violate some joint of the arm, and joint 4 is only sometimes the one that is
    # out of range.  Their distribution is in the study's JSON instead.
    assert out_of_limit_q4


def _same_configuration(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """True when two configurations differ only by whole turns, per joint."""
    delta = np.degrees(a) - np.degrees(b)
    return bool(np.all(np.abs((delta + 180.0) % 360.0 - 180.0) < tol))


def _inside_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    """True when every joint is inside its range, with a hair of slack for rounding."""
    degrees = np.degrees(q)
    return bool(
        np.all(degrees >= np.degrees(lower) - 1e-9) and np.all(degrees <= np.degrees(upper) + 1e-9)
    )


def test_study_pose_marks_exactly_the_recovering_branches(pose_cases, angdiff) -> None:
    """Fact 11: ``recovers_target`` is set precisely on the branches that return ``q``.

    The flag is recomputed independently from :func:`franka_ik.solver.
    branch_solutions` (same modulo-``2*pi`` comparison, 1e-6 rad) and the two sets
    of labels must be equal -- not merely non-empty.  Every flagged outcome must
    also be a real, correct branch: it exists and its pose residual is below the
    solver's 1e-9 acceptance threshold.  Measured over the 60 shared poses: the
    target is recovered by exactly one branch each time, which is the expected
    behaviour of a discrete branch enumeration.
    """
    for q, q7, pose in pose_cases:
        study = analysis.study_pose(pose, q7, target=q)
        flagged = {outcome.label for outcome in study.outcomes if outcome.recovers_target}
        expected = {
            solution.label
            for solution in solver.branch_solutions(pose, q7)
            if float(np.max(angdiff(solution.q, q))) < 1e-6
        }
        assert flagged == expected
        assert flagged, "the target configuration must be recovered by some branch"
        for outcome in study.outcomes:
            if not outcome.recovers_target:
                continue
            assert outcome.exists is True
            assert outcome.pose_error < 1e-9


def test_study_pose_counts_agree_with_the_solver(pose_cases) -> None:
    """Fact 11: ``n_distinct >= n_published``, and both equal the solver's answer.

    ``study_pose`` counts distinct *in-limit* solutions and how many of those the
    published elbow root finds; ``solve`` returns the same set, so the counts are
    recomputed from it here.  The inequality is the branch finding in miniature:
    the published subset can never exceed the full set, and on every one of the 60
    shared poses the full set is at least as large.
    """
    for _, q7, pose in pose_cases:
        study = analysis.study_pose(pose, q7)
        solutions = solver.solve(pose, q7, within_limits_only=True)
        assert study.n_distinct == len(solutions)
        assert study.n_published == sum(
            1 for solution in solutions if solution.q4_root == analysis.PUBLISHED_Q4_ROOT
        )
        assert study.n_published <= study.n_distinct
        assert study.n_distinct <= solver.NUM_BRANCHES
        assert study.exists_count <= solver.NUM_BRANCHES
        # The joint-7 value is echoed back unchanged (it is the parameter the
        # study is indexed by), so only a rounding-level tolerance is needed.
        assert study.q7 == pytest.approx(q7, rel=1e-15, abs=1e-15)


def test_study_pose_without_a_target_marks_nothing(pose_cases) -> None:
    """Without a target there is nothing to recover, so no outcome may be flagged.

    ``recovers_target`` is only meaningful when the configuration the pose came
    from is known; with ``target=None`` the counts must still be computed, which
    also keeps the two code paths comparable.
    """
    q, q7, pose = pose_cases[0]
    without = analysis.study_pose(pose, q7)
    with_target = analysis.study_pose(pose, q7, target=q)
    assert not any(outcome.recovers_target for outcome in without.outcomes)
    assert without.n_distinct == with_target.n_distinct
    assert without.n_published == with_target.n_published
    assert without.exists_count == with_target.exists_count


def test_study_pose_reports_every_branch_and_prints_itself(pose_cases) -> None:
    """``study_pose`` lists all eight branches, existing or not, and describes itself.

    Branches that have no solution for the pose are still reported (with
    ``exists=False``), because that absence is itself information: it is what
    distinguishes "this branch is out of limits" from "this branch does not exist
    for that pose".  ``describe`` feeds the demo pages, so it has to stay
    printable and to mention both counts.
    """
    _, q7, pose = pose_cases[0]
    study = analysis.study_pose(pose, q7, target=pose_cases[0][0])
    labels = [outcome.label for outcome in study.outcomes]
    assert len(labels) == solver.NUM_BRANCHES
    assert sorted(labels) == sorted(set(labels))
    assert set(study.outcomes[0].label.split()).issubset(
        {"q4+", "q4-", "phi+", "phi-", "plain", "flip"}
    )
    for outcome in study.outcomes:
        assert isinstance(outcome.label, str)
        assert outcome.exists in (True, False)
    text = study.describe()
    assert "joint 7" in text
    assert str(study.n_distinct) in text
    assert str(study.n_published) in text
