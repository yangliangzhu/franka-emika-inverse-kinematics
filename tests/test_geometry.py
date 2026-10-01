"""The S-R-S reduction and the elbow quadratic (facts 6 and 8).

``franka_ik/geometry.py`` is the part of the derivation that can be checked
without any solver: the two obstructions that make the Panda "almost" an S-R-S
arm are removed by (1) rotating the target by :math:`R_z(-q_7) R_y(-\\beta)
R_z(q_7)` and (2) replacing the S-R-S elbow law by a quadratic in
:math:`\\tan(\\theta_4/2)`.  This module tests both steps -- ``STEP1`` and
``STEP2`` of ``docs/method.md``, the quadratic being derived in the PDF's appendix
*特殊方程的求解* -- against their closed forms, plus the geometry constants
against the DH table of :mod:`franka_ik.model`.  The law the quadratic has to
collapse to when the bias is removed is eq. (12) of Shimizu et al. (2008), the
elbow law of a true S-R-S arm.

The one place where the brief and the code disagree is the reachable shell of
``||x_sw||``: 0.068 m to 0.700 m is the shell of the *unbiased* arm, i.e.
``|d_se - d_ew|`` to ``d_se + d_ew``, and it is exactly what the shell collapses
to when ``bias = 0``.  With the real bias of 0.0825 m the shell widens to
0.0661704620 m .. 0.7193542034 m; both facts are asserted and explained below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from franka_ik import analysis, geometry, model

#: The values the reduction is built on, straight from the published file.  The
#: library stores them as decimal literals, so comparing against the same
#: literals is exact and 1e-15 is already generous.
DOCUMENTED_GEOMETRY = {
    "d_bs": 0.333,
    "d_se": 0.316,
    "d_ew": 0.384,
    "d_wt": 0.107,
    "offset": 0.088,
    "bias": 0.0825,
}

#: Tolerance for identities that are the *same formula* evaluated twice (an
#: angle, a hypotenuse, a rotation product).  The measured residuals are at the
#: 1e-16 level, i.e. double-precision noise on values of order 1.
IDENTITY_TOLERANCE = 1e-15

#: Tolerance for quantities that go through a trigonometric round trip
#: (acos/atan, matrix products with a determinant).  Measured worst residuals
#: 1e-16..5e-15 depending on the expression, so 1e-12 leaves ~3 orders of
#: headroom while being ~9 orders below anything a modelling error would cause.
TRIG_TOLERANCE = 1e-12


def unbiased_shell(geometry_parameters: geometry.EquivalentGeometry) -> tuple:
    """Closed form of the reachable shell of ``||x_sw||`` for a true S-R-S arm.

    With ``bias = 0`` the quadratic degenerates to ``(d_se - d_ew)^2 <= d^2 <=
    (d_se + d_ew)^2``: the arm can fold until the two links overlap and stretch
    until they are collinear.  This is the annulus the brief quotes as
    0.068 m .. 0.700 m.
    """
    shortening = abs(geometry_parameters.d_se - geometry_parameters.d_ew)
    extension = geometry_parameters.d_se + geometry_parameters.d_ew
    return (shortening, extension)


def biased_shell(geometry_parameters: geometry.EquivalentGeometry) -> tuple:
    """Closed form of the reachable shell for a shoulder bias ``b``.

    ``disc(D) = a1^2 - 4 a0 a2`` (``D = ||x_sw||^2``) is a downward parabola in
    ``p = S^2 - D``, with ``S = d_se + d_ew`` and ``T = d_se - d_ew``:

        disc(D) = 16 b^2 S^2 - 4 p (p + k),   k = 4 b^2 + T^2 - S^2,

    so ``disc = 0`` is ``p^2 + k p - 4 b^2 S^2 = 0`` and the two roots
    ``p = (-k +- sqrt(k^2 + 16 b^2 S^2)) / 2`` give the shell
    ``D = S^2 - p``.  Written down here instead of being searched, which is what
    makes it a check on :func:`franka_ik.analysis.reachable_distance_range` and
    on :func:`franka_ik.geometry.q4_discriminant`.
    """
    bias = geometry_parameters.bias
    total = geometry_parameters.d_se + geometry_parameters.d_ew
    difference = geometry_parameters.d_se - geometry_parameters.d_ew
    constant = 4.0 * bias**2 + difference**2 - total**2
    discriminant = math.sqrt(constant**2 + 16.0 * bias**2 * total**2)
    inner = (-constant + discriminant) / 2.0  # larger p  -> smaller D
    outer = (-constant - discriminant) / 2.0  # smaller p -> larger D
    return (math.sqrt(total**2 - inner), math.sqrt(total**2 - outer))


def test_paper_geometry_is_the_documented_arm(paper_geometry) -> None:
    """``PAPER_GEOMETRY`` holds the six published numbers, in metres.

    These six numbers are the whole reduction; a typo here silently produces a
    solver for a different robot, so they are asserted literally against the
    values printed in ``franka_ik/geometry.py`` (and in the PDF's parameter
    table).  1e-15 absolute is exact for decimal literals.
    """
    for name, value in DOCUMENTED_GEOMETRY.items():
        assert getattr(paper_geometry, name) == pytest.approx(value, abs=1e-15)


def test_geometry_parameters_are_the_ones_in_the_dh_table(paper_geometry) -> None:
    """Every reduction parameter can be traced to a row of ``DH_PARAMETERS``.

    This is the link between the two modules that would otherwise drift apart:
    ``d_bs`` is the ``d`` of DH row 1, ``d_se`` of row 3, ``d_ew`` of row 5,
    ``d_wt`` of row 8, ``offset`` is the ``a`` of row 7 and ``bias`` the
    magnitude of the ``a`` of row 4.  Tolerance 1e-15: the same literals.
    """
    table = model.DH_PARAMETERS
    assert paper_geometry.d_bs == pytest.approx(table[0, 1], abs=1e-15)
    assert paper_geometry.d_se == pytest.approx(table[2, 1], abs=1e-15)
    assert paper_geometry.d_ew == pytest.approx(table[4, 1], abs=1e-15)
    assert paper_geometry.d_wt == pytest.approx(table[7, 1], abs=1e-15)
    assert paper_geometry.offset == pytest.approx(table[6, 0], abs=1e-15)
    assert paper_geometry.bias == pytest.approx(abs(table[3, 0]), abs=1e-15)
    # The bias is not an invention: the two shoulder rows carry +-0.0825.
    assert table[3, 0] == pytest.approx(-table[4, 0], abs=1e-15)


def test_wrist_offset_angle_and_effective_length(paper_geometry) -> None:
    """Fact 8: ``beta = atan2(0.088, 0.107)`` and the last link is ``hypot`` of them.

    ``beta`` is the angle the 0.088 m wrist offset adds to the last link, and
    ``hypot(0.088, 0.107)`` is the length that replaces ``d_wt`` once the offset
    is absorbed.  The library and the test evaluate the same ``math`` functions
    on the same arguments, so the tolerance is relative and at machine level.
    """
    expected_beta = math.atan2(0.088, 0.107)
    expected_length = math.hypot(0.088, 0.107)
    assert geometry.wrist_offset_angle(paper_geometry) == pytest.approx(expected_beta, rel=1e-15)
    assert geometry.effective_wrist_length(paper_geometry) == pytest.approx(
        expected_length, rel=1e-15
    )
    # 0.1385388032... m, i.e. ~0.032 m longer than d_wt: the offset is not
    # negligible, which is why the arm is not a true S-R-S arm.
    assert expected_length > paper_geometry.d_wt


def test_wrist_correction_at_zero_is_the_identity_rotated_by_beta(paper_geometry) -> None:
    """Fact 8: ``wrist_correction(0)`` is a pure rotation about y by ``-beta``.

    For ``q7 = 0`` the conjugation ``Rz(-q7) Ry(-beta) Rz(q7)`` collapses to
    ``Ry(-beta)``.  The sign convention of ``geometry.rot_y`` matters here: a
    flipped sign mirrors the correction and every branch returns a wrong pose
    (the module docstring records a 0.865 pose residual found exactly this way),
    so the test compares against the explicit matrix as well as the trace.
    """
    beta = geometry.wrist_offset_angle(paper_geometry)
    correction = geometry.wrist_correction(0.0, paper_geometry)

    explicit = np.array(
        [
            [math.cos(beta), 0.0, math.sin(beta)],
            [0.0, 1.0, 0.0],
            [-math.sin(beta), 0.0, math.cos(beta)],
        ]
    )
    np.testing.assert_allclose(correction, explicit, atol=IDENTITY_TOLERANCE)
    np.testing.assert_allclose(correction, geometry.rot_y(-beta), atol=IDENTITY_TOLERANCE)
    # Trace 1 + 2 cos(beta) identifies the rotation angle as beta.
    assert np.trace(correction) == pytest.approx(1.0 + 2.0 * math.cos(beta), abs=TRIG_TOLERANCE)


def test_wrist_correction_is_orthogonal_with_determinant_one(rng, paper_geometry) -> None:
    """Fact 8: the correction is a rotation for random ``q7``.

    ``Rz(-q7) Ry(-beta) Rz(q7)`` is a conjugation of a rotation, so it must stay
    in SO(3): orthogonal and with determinant +1 (not -1, which would be a
    reflection and would break the handedness of the whole construction).
    Tolerance 1e-12: the products are 3x3, so the observed residuals are 1e-16
    and 1e-12 is pure headroom.
    """
    for q7 in rng.uniform(-math.pi, math.pi, size=64):
        correction = geometry.wrist_correction(float(q7), paper_geometry)
        np.testing.assert_allclose(correction @ correction.T, np.eye(3), atol=TRIG_TOLERANCE)
        assert np.linalg.det(correction) == pytest.approx(1.0, abs=TRIG_TOLERANCE)
        # A conjugation preserves the angle: trace is a constant of q7.
        assert np.trace(correction) == pytest.approx(
            1.0 + 2.0 * math.cos(geometry.wrist_offset_angle(paper_geometry)),
            abs=TRIG_TOLERANCE,
        )


def test_wrist_correction_is_the_documented_conjugation(rng, paper_geometry) -> None:
    """``wrist_correction(q7)`` equals ``Rz(-q7) Ry(-beta) Rz(q7)`` recomputed.

    Guards against a reordering of the three factors, which would still be a
    valid rotation but would conjugate the wrong axis.  Same formula, so 1e-15.
    """
    beta = geometry.wrist_offset_angle(paper_geometry)
    for q7 in rng.uniform(-math.pi, math.pi, size=16):
        expected = geometry.rot_z(-q7) @ geometry.rot_y(-beta) @ geometry.rot_z(q7)
        np.testing.assert_allclose(
            geometry.wrist_correction(float(q7), paper_geometry),
            expected,
            atol=IDENTITY_TOLERANCE,
        )


def test_shoulder_to_wrist_vector_decomposition(rng, paper_geometry) -> None:
    """``x_sw = x - [0,0,d_bs] - R_corrected [0,0,l_wt]``, and it reconstructs ``x``.

    The round trip is the part that matters: the vector is what the elbow
    quadratic is solved for, so an error here (e.g. using ``d_wt`` instead of the
    effective length) would move the whole solution set.  Same formula twice,
    hence 1e-15.
    """
    length = geometry.effective_wrist_length(paper_geometry)
    for _ in range(16):
        position = rng.uniform(-0.5, 0.5, size=3)
        corrected = geometry.wrist_correction(float(rng.uniform(-math.pi, math.pi)), paper_geometry)
        p_sw = geometry.shoulder_to_wrist(position, corrected, paper_geometry)
        expected = (
            position
            - np.array([0.0, 0.0, paper_geometry.d_bs])
            - corrected @ np.array([0.0, 0.0, length])
        )
        np.testing.assert_allclose(p_sw, expected, atol=IDENTITY_TOLERANCE)
        np.testing.assert_allclose(
            p_sw
            + np.array([0.0, 0.0, paper_geometry.d_bs])
            + corrected @ np.array([0.0, 0.0, length]),
            position,
            atol=IDENTITY_TOLERANCE,
        )


def test_q4_coefficients_are_the_published_expressions(rng, paper_geometry) -> None:
    """Fact 6: ``(a2, a1, a0)`` are the three published lines of ``ik_ca.py``.

    Compared against the CasADi source verbatim, including the sign convention of
    the quadratic ``a2 x^2 - a1 x + a0 = 0``.  ``a1`` is positive for the real
    geometry, which is what makes the ``+`` root the one the published solver
    kept.  Same arithmetic, so 1e-15 on values of order 0.5.
    """
    for distance_squared in rng.uniform(0.0, 1.0, size=32):
        a2, a1, a0 = geometry.q4_coefficients(float(distance_squared), paper_geometry)
        bias, d_se, d_ew = paper_geometry.bias, paper_geometry.d_se, paper_geometry.d_ew
        assert a2 == pytest.approx(4 * bias**2 + (d_se - d_ew) ** 2 - distance_squared, abs=1e-15)
        assert a1 == pytest.approx(4 * bias * (d_se + d_ew), abs=1e-15)
        assert a0 == pytest.approx((d_se + d_ew) ** 2 - distance_squared, abs=1e-15)
        assert a1 > 0.0


def test_q4_discriminant_is_the_determinant_of_the_quadratic(rng, paper_geometry) -> None:
    """``q4_discriminant`` equals ``a1^2 - 4 a0 a2`` for the coefficients.

    A trivial identity, kept because every reachability statement in the
    repository is expressed through this scalar: a mismatch between the two
    functions would make the failure classifier disagree with the root finder.
    """
    for distance_squared in rng.uniform(0.0, 1.0, size=32):
        a2, a1, a0 = geometry.q4_coefficients(float(distance_squared), paper_geometry)
        assert geometry.q4_discriminant(float(distance_squared), paper_geometry) == pytest.approx(
            a1 * a1 - 4.0 * a0 * a2, abs=1e-15
        )


def test_q4_collapses_to_the_srs_elbow_law(rng, srs_geometry) -> None:
    """Fact 6: with ``bias = 0`` the two roots are ``+-arccos(cos theta4)``.

    This is the check that the quadratic is an *extension* of the classical
    closed form of Shimizu et al. (2008) and not a different formula: for a true
    S-R-S arm, ``cos(theta4) = (d^2 - d_se^2 - d_ew^2) / (2 d_se d_ew)``.  Both
    roots are tested because the unbiased quadratic is symmetric in
    ``tan(theta4/2)``.  The distances are spread evenly over the shell interior
    rather than drawn at random, so the test covers every distance once and
    cannot flake; the 1e-3 m margin keeps the sample away from the two shell
    boundaries, where the ``tan(theta4/2)`` parameterisation is ill-conditioned
    (the roots pile up at ``+-pi``) -- that regime has its own test below.
    Measured worst deviation on this grid: 5.2e-15, against a tolerance of 1e-12.
    """
    d_se, d_ew = srs_geometry.d_se, srs_geometry.d_ew
    inner, outer = unbiased_shell(srs_geometry)
    for distance in np.linspace(inner + 1e-3, outer - 1e-3, 200):
        law = math.acos((distance**2 - d_se**2 - d_ew**2) / (2.0 * d_se * d_ew))
        roots = geometry.q4_roots(float(distance) ** 2, srs_geometry)
        assert len(roots) == 2
        assert min(abs(roots[0] - law), abs(roots[0] + law)) < TRIG_TOLERANCE
        assert min(abs(roots[1] - law), abs(roots[1] + law)) < TRIG_TOLERANCE
        # One root on each side of the shoulder-wrist line.
        assert abs(roots[0] - roots[1]) > 1e-6


def test_q4_roots_satisfy_the_quadratic(paper_geometry, srs_geometry) -> None:
    """Substituting ``x = tan(theta4/2)`` makes the quadratic vanish.

    The roots are returned as *angles*, so this is the only test that checks the
    ``2 atan(x)`` back-transformation as well as the root formula.  The residual
    is not normalised by the coefficients (which are O(0.5) here), and it is
    sensitive to how close the distance is to a shell boundary: as the elbow
    folds, one root's ``tan(theta4/2)`` diverges and the substitution loses
    precision.  Measured worst residual on an evenly spaced grid (200 distances)
    for the *biased* arm: 5.4e-9 at a 1e-3 m margin, 2.2e-11 at 0.02 m.  The test
    therefore uses a 0.02 m margin and a 1e-9 tolerance, which leaves ~45x
    headroom and still covers 97 % of the shell; the unbiased arm is
    well-conditioned everywhere (worst 5.2e-15 on the same grid).  The degenerate
    boundary itself is covered by the test below.
    """
    for parameters in (paper_geometry, srs_geometry):
        inner, outer = (
            unbiased_shell(parameters) if parameters.bias == 0.0 else biased_shell(parameters)
        )
        for distance in np.linspace(inner + 0.02, outer - 0.02, 200):
            a2, a1, a0 = geometry.q4_coefficients(float(distance) ** 2, parameters)
            roots = geometry.q4_roots(float(distance) ** 2, parameters)
            assert roots, "an interior distance must have at least one root"
            for angle in roots:
                value = math.tan(angle / 2.0)
                assert abs(a2 * value * value - a1 * value + a0) < 1e-9


def test_q4_roots_returns_two_roots_inside_the_shell(paper_geometry) -> None:
    """Fact 6: two distinct roots when the discriminant is positive.

    Four interior distances of the biased arm, chosen across the shell.  The
    second half of the test is the one that matters scientifically: at
    ``d = d_se + d_ew`` (the fully stretched arm) the *biased* quadratic still
    has a positive discriminant ``a1^2``, so it has two roots -- the
    "one root at the boundary" picture only holds for the unbiased arm.
    """
    for distance in (0.1, 0.2, 0.5, 0.7):
        roots = geometry.q4_roots(distance**2, paper_geometry)
        assert len(roots) == 2
        assert abs(roots[0] - roots[1]) > 1e-6
        assert geometry.q4_discriminant(distance**2, paper_geometry) > 0.0

    total = paper_geometry.d_se + paper_geometry.d_ew
    assert geometry.q4_discriminant(total**2, paper_geometry) > 0.0
    assert len(geometry.q4_roots(total**2, paper_geometry)) == 2


def test_q4_roots_returns_one_root_at_the_unbiased_outer_shell(srs_geometry) -> None:
    """Fact 6: exactly one root when the discriminant is zero.

    At ``d = d_se + d_ew`` with ``bias = 0`` the quadratic degenerates to
    ``a2 x^2 + a0 = 0`` with ``a0 = 0`` and ``a1 = 0``: the discriminant is
    exactly zero, both roots coincide at ``theta4 = 0``, and ``q4_roots``
    reports the single configuration instead of a duplicate.  1e-12 on the
    angle: the value is ``2 atan(0)``, i.e. exactly zero up to one rounding.
    """
    total = srs_geometry.d_se + srs_geometry.d_ew
    discriminant = geometry.q4_discriminant(total**2, srs_geometry)
    assert discriminant == pytest.approx(0.0, abs=1e-15)
    roots = geometry.q4_roots(total**2, srs_geometry)
    assert len(roots) == 1
    assert roots[0] == pytest.approx(0.0, abs=TRIG_TOLERANCE)


def test_q4_roots_is_empty_outside_the_reachable_shell(paper_geometry, srs_geometry) -> None:
    """Fact 6: no roots when the distance is outside the shell.

    ``q4_roots`` must return an empty list rather than raising or returning a
    complex number, because that empty list is how :func:`franka_ik.solver.
    solve_branch` reports an unreachable pose.  Two distances on each side of the
    shell, plus the matching negative discriminant.
    """
    biased_inner, biased_outer = biased_shell(paper_geometry)
    for parameters, distance in (
        (paper_geometry, biased_inner - 0.01),
        (paper_geometry, biased_outer + 0.05),
        (srs_geometry, 0.01),
        (srs_geometry, 0.9),
    ):
        assert geometry.q4_discriminant(distance**2, parameters) < 0.0
        assert geometry.q4_roots(distance**2, parameters) == []


def test_q4_roots_at_the_fully_folded_degenerate_boundary(srs_geometry) -> None:
    """The fully folded pose ``d = |d_se - d_ew|`` is a documented degenerate case.

    With ``bias = 0`` and ``d = |d_se - d_ew|`` the coefficients are ``a2 = 0``
    and ``a1 = 0``: the equation collapses to ``a0 = 0`` and the true solution
    ``theta4 = +-pi`` is at infinity in the ``tan(theta4/2)`` parameterisation,
    which is why ``geometry.py`` guards ``abs(a2) < 1e-15`` and returns no roots
    (the guard is marked ``pragma: no cover - degenerate geometry``).  A hair
    inside the shell the two roots reappear and pile up at ``+-pi``.  This is a
    parameterisation limit, not a wrong answer, and it is asserted so that a
    future change to the guard is noticed.
    """
    difference = abs(srs_geometry.d_se - srs_geometry.d_ew)
    assert geometry.q4_coefficients(difference**2, srs_geometry)[0] == pytest.approx(0.0, abs=1e-15)
    assert geometry.q4_roots(difference**2, srs_geometry) == []

    slightly_inside = difference**2 * (1.0 + 1e-9)
    roots = geometry.q4_roots(slightly_inside, srs_geometry)
    assert len(roots) == 2
    for angle in roots:
        assert abs(abs(angle) - math.pi) < 1e-3


def test_reachable_shell_of_the_unbiased_arm_is_the_srs_annulus(srs_geometry) -> None:
    """Fact 6: with ``bias = 0`` the shell is exactly 0.068 m .. 0.700 m.

    ``reachable_distance_range`` scans ``||x_sw||`` on a grid; with the library's
    default resolution the step is ``(0.7 - 0.068) / 20000 = 3.2e-5`` m, so the
    analytic bounds can only be matched to one grid step.  Tolerance 1e-4 m
    covers that (measured: the scan returns the bounds to 1e-16 for this
    geometry, because both ends of the span are feasible and on-grid).
    """
    inner, outer = unbiased_shell(srs_geometry)
    assert inner == pytest.approx(abs(0.316 - 0.384), abs=1e-15)
    assert outer == pytest.approx(0.316 + 0.384, abs=1e-15)

    measured_inner, measured_outer = analysis.reachable_distance_range(srs_geometry)
    assert measured_inner == pytest.approx(inner, abs=1e-4)
    assert measured_outer == pytest.approx(outer, abs=1e-4)


def test_shoulder_bias_widens_the_reachable_shell(paper_geometry) -> None:
    """The real arm's shell is 0.0661704620 .. 0.7193542034 m, not 0.068 .. 0.700.

    Fact 6 quotes the 0.068/0.700 annulus as "matching ``|d_se - d_ew|`` and
    ``d_se + d_ew``"; that is the shell of the arm *without* the shoulder bias
    (see the test above).  For ``bias = 0.0825`` the quadratic's discriminant is
    not centred on that annulus and the reachable shell widens on both sides --
    the arm reaches 1.8 mm closer to the shoulder and 19.4 mm further away, which
    is a real workspace difference and the reason :func:`franka_ik.geometry.
    q4_discriminant` cannot be replaced by the closed-form S-R-S annulus.
    ``q4_roots`` and :func:`franka_ik.analysis.reachable_distance_range` are
    checked against the analytic bounds derived in :func:`biased_shell`.
    """
    inner, outer = biased_shell(paper_geometry)
    measured_inner, measured_outer = analysis.reachable_distance_range(paper_geometry)
    # The scan grid is (0.7 - 0.068)/20000 = 3.2e-5 m, hence the 1e-4 tolerance.
    assert measured_inner == pytest.approx(inner, abs=1e-4)
    assert measured_outer == pytest.approx(outer, abs=1e-4)

    unbiased_inner, unbiased_outer = unbiased_shell(paper_geometry)
    assert inner < unbiased_inner - 1e-3
    assert outer > unbiased_outer + 1e-3

    # The discriminant must change sign exactly at those bounds.
    assert geometry.q4_discriminant((inner * (1 + 1e-6)) ** 2, paper_geometry) > 0.0
    assert geometry.q4_discriminant((inner * (1 - 1e-6)) ** 2, paper_geometry) < 0.0
    assert geometry.q4_discriminant((outer * (1 - 1e-6)) ** 2, paper_geometry) > 0.0
    assert geometry.q4_discriminant((outer * (1 + 1e-6)) ** 2, paper_geometry) < 0.0


def test_link_offset_and_equivalent_link_vectors(rng, paper_geometry, srs_geometry) -> None:
    """``delta = -tan(theta4/2) bias`` and the links grow by it, and by nothing else.

    ``delta`` is the whole "family of S-R-S arms" parameterisation: the links of
    the equivalent arm are ``[0, d_se + delta, 0]`` and ``[0, 0, d_ew + delta]``.
    For ``bias = 0`` the vectors must be *exactly* the published links, which is
    why that half of the test asserts bit equality rather than a tolerance: the
    subtraction ``d_se + 0.0`` is exact.
    """
    for theta4 in rng.uniform(-math.pi, math.pi, size=32):
        delta = geometry.link_offset(float(theta4), paper_geometry)
        assert delta == pytest.approx(-math.tan(theta4 / 2.0) * paper_geometry.bias, abs=1e-15)
        l_se, l_ew = geometry.equivalent_link_vectors(float(theta4), paper_geometry)
        np.testing.assert_allclose(
            l_se, np.array([0.0, paper_geometry.d_se + delta, 0.0]), atol=1e-15
        )
        np.testing.assert_allclose(
            l_ew, np.array([0.0, 0.0, paper_geometry.d_ew + delta]), atol=1e-15
        )
    # Odd in theta4, and zero for an unbiased shoulder.
    assert geometry.link_offset(0.7, paper_geometry) == pytest.approx(
        -geometry.link_offset(-0.7, paper_geometry), abs=1e-15
    )
    l_se, l_ew = geometry.equivalent_link_vectors(0.7, srs_geometry)
    np.testing.assert_array_equal(l_se, np.array([0.0, 0.316, 0.0]))
    np.testing.assert_array_equal(l_ew, np.array([0.0, 0.0, 0.384]))


def test_equivalent_joint_rotation_uses_the_reduction_convention(rng) -> None:
    """``R_i(theta) = Rz(theta) Rx(alpha_i)`` with the seven published alphas.

    The reduction works in *standard* DH rotations with
    ``alpha = [-1, 1, 1, -1, 1, 1, 0] pi/2`` -- a different convention from the
    modified-DH transforms of :mod:`franka_ik.model`, which is exactly the sort of
    mismatch the numerical model comparison in ``test_model.py`` exists to catch.
    Joint 7 has ``alpha = 0``, so it is a pure ``Rz``.  Same formula, so 1e-15.
    """
    np.testing.assert_allclose(
        geometry.DEFAULT_ALPHAS,
        np.array([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]) * (math.pi / 2.0),
        atol=1e-15,
    )
    theta = float(rng.uniform(-math.pi, math.pi))
    for index in range(1, 8):
        expected = geometry.rot_z(theta) @ geometry.rot_x(float(geometry.DEFAULT_ALPHAS[index - 1]))
        np.testing.assert_allclose(
            geometry.equivalent_joint_rotation(theta, index), expected, atol=IDENTITY_TOLERANCE
        )
    np.testing.assert_allclose(
        geometry.equivalent_joint_rotation(theta, 7), geometry.rot_z(theta), atol=IDENTITY_TOLERANCE
    )
    with pytest.raises(IndexError):
        geometry.equivalent_joint_rotation(theta, 8)


def test_equivalent_joint_rotation_is_a_rotation(rng) -> None:
    """Every equivalent joint rotation is orthogonal with determinant +1.

    Cheap sanity check on the convention: if ``rot_x`` or ``rot_z`` in
    :mod:`franka_ik.geometry` had a wrong sign, the product would still be a
    rotation, but a bad *order* (``Rx Rz`` instead of ``Rz Rx``) would make the
    reduction inconsistent with the model -- which the solver tests catch.  Here
    only the SO(3) property is asserted, with 1e-12.
    """
    for index in range(1, 8):
        for theta in rng.uniform(-math.pi, math.pi, size=8):
            rotation = geometry.equivalent_joint_rotation(float(theta), index)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=TRIG_TOLERANCE)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=TRIG_TOLERANCE)
