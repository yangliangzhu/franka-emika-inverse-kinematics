"""The NumPy model against the published CasADi model (fact 1).

``franka_ik/model.py`` claims to be ``original/panda.py`` evaluated in NumPy
"and nothing else changed".  These tests are what makes that claim checkable:

* the DH table and the joint limits are compared entry by entry against the
  published class, so a typo in a link length or an angle cannot slip through;
* ``fk_flange``, ``fk_tool`` and ``jacobian`` are compared against the published
  CasADi functions over a fixed sample of 50 configurations in ``[-2.5, 2.5]``;
* the Jacobian is *additionally* verified against central finite differences of
  ``fk_flange``, because agreeing with the published code is not the same as
  being the derivative of the frame the solver targets.

Two API facts worth knowing before reading the tests:

* ``Panda.fk`` is a Python method, so ``Panda().fk(q)`` returns a symbolic
  ``casadi.SX`` rather than a number: unlike ``forward_flange``, it is not
  wrapped in a ``casadi.Function``, and ``Panda.rot_z`` starts from
  ``ca.SX_eye(4)``.  The comparison therefore goes through the wrapper built by
  the ``published_tool_pose`` fixture, and the defect itself is pinned by
  ``test_published_fk_is_symbolic_while_forward_flange_is_numeric``.
* ``Panda.jacobian`` differentiates the **tool** point (DH row 9, the extra
  0.1034 m), while ``Panda.jacobian_flange`` and ``franka_ik.model.jacobian``
  both describe the **flange** (DH row 8) that the analytical IK solves for.
"""

from __future__ import annotations

import numpy as np
import pytest

from franka_ik import model

#: Sample size for the model-versus-CasADi comparisons.  The published code is
#: evaluated through CasADi's SX graph, so each configuration costs a function
#: call; 50 draws are enough to see a wrong link length (which would show up as a
#: large constant error) and cheap enough to stay a fraction of a second.
MODEL_SAMPLES = 50

#: Joint range of those draws.  Deliberately *wider* than the real limits: the
#: forward kinematics has to be right everywhere, and a bug in a limit-specific
#: branch would be invisible if the sample stayed inside the real ranges.
MODEL_SAMPLE_RANGE = 2.5

#: Absolute tolerance for model-versus-CasADi comparisons, in metres / radians.
#: Both sides evaluate the same closed-form expressions with a different
#: operation order (NumPy right-multiplication vs CasADi SX), so the observed
#: worst disagreement is 3.3e-16 for the poses and 5.6e-16 for the Jacobian --
#: pure double-precision noise.  1e-12 is ~4 orders above that noise (so the
#: test cannot flake) and still ~1e-9 mm in physical units, far below anything a
#: real modelling error would produce.
MODEL_TOLERANCE = 1e-12

#: Step of the central-difference Jacobian check.  1e-6 rad balances the two
#: error terms of a central difference: truncation ~h^2 = 1e-12 and cancellation
#: ~eps/h = 1e-10.  The measured worst disagreement over the sample is 1.9e-10,
#: so 1e-7 leaves ~2 orders of headroom while remaining far tighter than any
#: analytic mistake (the tool/flange confusion this guards against is ~1e-1).
FINITE_DIFFERENCE_STEP = 1e-6
FINITE_DIFFERENCE_TOLERANCE = 1e-7


def test_dh_table_is_a_verbatim_copy_of_the_published_table(published_panda) -> None:
    """``DH_PARAMETERS`` must equal ``Panda.dh_params`` entry by entry.

    The library documents the table as "taken verbatim from panda.py", so this
    is one of the few places where bit-exact equality is the *correct*
    assertion: both arrays are built from the same decimal literals, and any
    difference is a real edit to the model, not rounding.
    """
    np.testing.assert_array_equal(model.DH_PARAMETERS, published_panda.dh_params)


def test_joint_limits_are_the_ones_the_robot_reports(published_panda) -> None:
    """The degree limits must match the published bounds, one-sided joints included.

    Joint 4 is ``[-175, -5]`` and joint 6 is ``[0, 214]`` degrees; both are
    one-sided, which is what makes the wrapping in :mod:`franka_ik.solver`
    non-trivial and what the published ``limit_joints`` gets wrong.  Tolerance is
    1e-12 degrees: the published class stores radians and the comparison goes
    through ``np.degrees``, so only the degree/radian round trip contributes
    (~1e-14), not a modelling difference.
    """
    np.testing.assert_allclose(
        model.LOWER_LIMITS_DEG, np.degrees(published_panda.lower_bounds), atol=1e-12
    )
    np.testing.assert_allclose(
        model.UPPER_LIMITS_DEG, np.degrees(published_panda.upper_bounds), atol=1e-12
    )

    assert model.LOWER_LIMITS_DEG.shape == (model.NUM_JOINTS,)
    assert model.UPPER_LIMITS_DEG.shape == (model.NUM_JOINTS,)
    # The two one-sided joints are the interesting ones: joint 4 cannot be
    # positive at all and joint 6 cannot be negative.
    assert model.UPPER_LIMITS_DEG[3] == -5.0
    assert model.LOWER_LIMITS_DEG[5] == 0.0
    assert np.all(model.LOWER_LIMITS_DEG < model.UPPER_LIMITS_DEG)


def test_limit_helpers_return_the_same_angles_in_radians() -> None:
    """``lower_limits``/``upper_limits``/``limits`` are consistent with the degrees.

    ``np.radians`` of the same input array is deterministic, so the tolerance is
    1e-15 radians -- one order above double-precision noise on values of order 3.
    """
    lower, upper = model.lower_limits(), model.upper_limits()
    np.testing.assert_allclose(lower, np.radians(model.LOWER_LIMITS_DEG), atol=1e-15)
    np.testing.assert_allclose(upper, np.radians(model.UPPER_LIMITS_DEG), atol=1e-15)
    limits_lower, limits_upper = model.limits()
    np.testing.assert_array_equal(limits_lower, lower)
    np.testing.assert_array_equal(limits_upper, upper)


def test_flange_pose_matches_the_published_forward_kinematics(rng, published_panda) -> None:
    """``fk_flange`` == ``Panda().forward_flange`` over 50 seeded configurations.

    Fact 1, first half.  The published ``forward_flange`` stops after DH row 8,
    which is the flange frame the analytical IK targets; if the two models ever
    drifted, every solution in the suite would be validated against the wrong
    frame.  Measured worst disagreement over these 50 draws: 3.3e-16, against a
    tolerance of 1e-12.
    """
    worst = 0.0
    for _ in range(MODEL_SAMPLES):
        q = rng.uniform(-MODEL_SAMPLE_RANGE, MODEL_SAMPLE_RANGE, size=model.NUM_JOINTS)
        expected = np.array(published_panda.forward_flange(q), dtype=float)
        worst = max(worst, float(np.max(np.abs(model.fk_flange(q) - expected))))
    assert worst < MODEL_TOLERANCE


def test_tool_pose_matches_the_published_fk(rng, published_tool_pose) -> None:
    """``fk_tool`` == ``Panda().fk(q)``, through the mandatory CasADi wrapper.

    Fact 1, second half.  ``published_tool_pose`` is ``Panda.fk`` wrapped into a
    ``casadi.Function``; the wrapper is required *because of a defect in the
    published model*, not because of the test (see the fixture's docstring and
    ``test_published_fk_is_symbolic_while_forward_flange_is_numeric``).
    Measured worst disagreement over these 50 draws: 3.3e-16, and the same
    value is obtained by extracting the symbolic result with ``casadi.DM``.
    """
    worst = 0.0
    for _ in range(MODEL_SAMPLES):
        q = rng.uniform(-MODEL_SAMPLE_RANGE, MODEL_SAMPLE_RANGE, size=model.NUM_JOINTS)
        expected = np.array(published_tool_pose(q), dtype=float)
        worst = max(worst, float(np.max(np.abs(model.fk_tool(q) - expected))))
    assert worst < MODEL_TOLERANCE


def test_published_fk_is_symbolic_while_forward_flange_is_numeric(rng, published_panda) -> None:
    """The published ``Panda.fk`` cannot be used numerically without a wrapper.

    ``Panda.forward_flange`` is a ``casadi.Function`` and returns a numeric
    ``DM``.  ``Panda.fk`` is a plain Python method whose last operation is
    ``H @ self.rot_z(-pi/4)``, and ``Panda.rot_z`` starts from ``ca.SX_eye(4)``,
    so its return value is a symbolic ``SX`` -- which ``np.array`` rejects with
    "Implicit conversion of symbolic CasADi type to numeric matrix not
    supported".

    This is a real defect of the published file (the notebook only ever calls
    ``forward_flange``, which is why it went unnoticed), and it is pinned here so
    the next reader does not rediscover it as a mysterious exception: the suite
    keeps using the wrapper from :func:`tests.conftest.published_tool_pose`.  The
    exception type is a bare ``Exception``, so it is identified by its message;
    the assertion would need revisiting only if CasADi changed that message.
    """
    q = rng.uniform(-MODEL_SAMPLE_RANGE, MODEL_SAMPLE_RANGE, size=model.NUM_JOINTS)
    assert type(published_panda.forward_flange(q)).__name__ == "DM"
    assert type(published_panda.fk(q)).__name__ == "SX"
    with pytest.raises(Exception, match="Implicit conversion"):
        np.array(published_panda.fk(q))


def test_tool_pose_is_the_flange_pose_with_the_factory_tool_rotation(rng) -> None:
    """``fk_tool`` is the flange frame translated by DH row 9, then rotated.

    The tool pose is *not* ``fk_flange @ RotZ(-pi/4)``: DH row 9 adds a 0.1034 m
    translation along the flange z-axis, and only then does the factory
    ``RotZ(-pi/4)`` apply.  Getting that wrong is exactly the kind of frame
    confusion this test exists to prevent (the 0.1034 m shows up as a 0.07 m
    position error), so both halves -- orientation and origin -- are asserted.
    The two sides are the same arithmetic, hence the 1e-15 tolerance.
    """
    for _ in range(10):
        q = rng.uniform(-MODEL_SAMPLE_RANGE, MODEL_SAMPLE_RANGE, size=model.NUM_JOINTS)
        frames = model.forward_kinematics(q)
        flange = model.fk_flange(q)
        tool = model.fk_tool(q)
        np.testing.assert_allclose(flange, frames[model.FLANGE_ROW], atol=1e-15)
        np.testing.assert_allclose(tool, frames[-1] @ model.rot_z(model.TOOL_ROTATION), atol=1e-15)
        # Orientation: the flange frame with the factory tool rotation.
        np.testing.assert_allclose(
            tool[:3, :3],
            flange[:3, :3] @ model.rot_z(model.TOOL_ROTATION)[:3, :3],
            atol=1e-15,
        )
        # Origin: the flange origin plus the rigid DH row-9 offset, which the
        # tool rotation (a pure rotation about z) leaves alone.
        np.testing.assert_allclose(
            tool[:3, 3],
            flange[:3, 3] + flange[:3, :3] @ np.array([0.0, 0.0, 0.1034]),
            atol=1e-15,
        )


def test_jacobian_matches_the_published_flange_jacobian(rng, published_panda) -> None:
    """``jacobian`` == ``Panda().jacobian_flange``, not ``Panda().jacobian``.

    Fact 1 named ``Panda().jacobian`` as the counterpart of
    ``franka_ik.model.jacobian``; that is not what the code does, and the
    difference is not rounding.  ``Panda.jacobian`` takes its linear part at the
    *tool* point (``dh_fk[-1]``, i.e. DH row 9, 0.1034 m beyond the flange),
    ``Panda.jacobian_flange`` and ``franka_ik.model.jacobian`` both take it at the
    flange (``dh_fk[7]``).  The library's own docstring calls its function the
    Jacobian "of the flange frame ... the frame the analytical IK solves for", so
    ``jacobian_flange`` is the honest counterpart and it matches to 5.6e-16
    (tolerance 1e-12).  The test below pins the exact relation to
    ``Panda().jacobian``.
    """
    worst_flange = worst_tool = 0.0
    for _ in range(MODEL_SAMPLES):
        q = rng.uniform(-MODEL_SAMPLE_RANGE, MODEL_SAMPLE_RANGE, size=model.NUM_JOINTS)
        jacobian = model.jacobian(q)
        worst_flange = max(
            worst_flange,
            float(np.max(np.abs(jacobian - np.array(published_panda.jacobian_flange(q))))),
        )
        worst_tool = max(
            worst_tool,
            float(np.max(np.abs(jacobian - np.array(published_panda.jacobian(q))))),
        )
    assert worst_flange < MODEL_TOLERANCE
    # The two published Jacobians really are different functions: the linear
    # block differs by roughly |p_tool - p_flange| = 0.1034 m, i.e. by O(1e-1).
    assert worst_tool > 1e-2


def test_jacobian_is_the_flange_not_the_tool_point_jacobian(rng, published_panda) -> None:
    """The published ``jacobian`` equals ours plus ``axis_i x (p_tool - p_flange)``.

    This is the strongest way to state the frame convention: rather than
    asserting "they differ", it asserts the *exact* transformation between the
    two, which is what a reader of the paper needs in order to use this Jacobian
    for a tool point.  Measured worst disagreement: 2.2e-16 (tolerance 1e-12).
    """
    worst = 0.0
    largest_linear_gap = 0.0
    for _ in range(MODEL_SAMPLES):
        q = rng.uniform(-MODEL_SAMPLE_RANGE, MODEL_SAMPLE_RANGE, size=model.NUM_JOINTS)
        frames = model.forward_kinematics(q)
        flange, tool = frames[model.FLANGE_ROW], frames[-1]
        prediction = model.jacobian(q).copy()
        for index in range(model.NUM_JOINTS):
            # Linear velocity of a point r away from the flange along the axis.
            prediction[:3, index] += np.cross(frames[index][:3, 2], tool[:3, 3] - flange[:3, 3])
        published = np.array(published_panda.jacobian(q), dtype=float)
        worst = max(worst, float(np.max(np.abs(prediction - published))))
        largest_linear_gap = max(
            largest_linear_gap,
            float(np.max(np.abs(prediction[:3] - model.jacobian(q)[:3]))),
        )
    assert worst < MODEL_TOLERANCE
    assert largest_linear_gap > 1e-2


def test_jacobian_linear_part_matches_finite_differences(in_limit_configurations) -> None:
    """The linear block really is ``d p_flange / d q``, checked independently.

    Agreement with the published code is not proof of correctness (both could be
    wrong in the same way), so the flange position is differentiated numerically
    with central differences over configurations drawn inside the real limits.
    Tolerance 1e-7 (measured worst 1.9e-10): see
    :data:`FINITE_DIFFERENCE_TOLERANCE`.
    """
    step = FINITE_DIFFERENCE_STEP
    for q in in_limit_configurations[:20]:
        jacobian = model.jacobian(q)
        for index in range(model.NUM_JOINTS):
            basis = np.zeros(model.NUM_JOINTS)
            basis[index] = 1.0
            forward = model.fk_flange(q + step * basis)[:3, 3]
            backward = model.fk_flange(q - step * basis)[:3, 3]
            derivative = (forward - backward) / (2.0 * step)
            np.testing.assert_allclose(
                jacobian[:3, index], derivative, atol=FINITE_DIFFERENCE_TOLERANCE
            )


def test_jacobian_angular_part_is_the_joint_axis_and_its_rotation_rate(
    in_limit_configurations,
) -> None:
    """The angular block is joint ``i``'s axis, and ``omega = vee(R' R^T)``.

    Two claims in one test because they must hold simultaneously: the axis has to
    be the z-axis of the joint frame the reduction uses (otherwise the analytical
    construction and the numeric Jacobian describe different arms), and the
    rotation rate has to agree with the finite-difference angular velocity
    (otherwise the block could be a plausible-looking wrong axis).  The axis
    comparison is bit-exact because it is the same expression; the finite
    difference uses the same 1e-7 tolerance as the linear part (measured worst
    1.9e-10).
    """
    step = FINITE_DIFFERENCE_STEP
    for q in in_limit_configurations[:20]:
        jacobian = model.jacobian(q)
        frames = model.forward_kinematics(q)
        rotation = frames[model.FLANGE_ROW][:3, :3]
        for index in range(model.NUM_JOINTS):
            axis = frames[index][:3, 2]
            np.testing.assert_array_equal(jacobian[3:, index], axis)

            basis = np.zeros(model.NUM_JOINTS)
            basis[index] = 1.0
            forward = model.fk_flange(q + step * basis)[:3, :3]
            backward = model.fk_flange(q - step * basis)[:3, :3]
            rate = (forward - backward) / (2.0 * step) @ rotation.T
            omega = np.array([rate[2, 1], rate[0, 2], rate[1, 0]])
            np.testing.assert_allclose(jacobian[3:, index], omega, atol=FINITE_DIFFERENCE_TOLERANCE)


def test_manipulability_follows_its_definition(in_limit_configurations) -> None:
    """``manipulability`` is ``sqrt(det(J J^T))`` and is positive in the workspace.

    Recomputed from the Jacobian rather than trusted, with a 1e-12 relative
    tolerance: both sides take a square root of the same determinant, so only
    evaluation order differs.  Every configuration inside the real limits is
    non-singular for the flange frame, hence the strict ``> 0``.
    """
    for q in in_limit_configurations[:20]:
        jacobian = model.jacobian(q)
        determinant = float(np.linalg.det(jacobian @ jacobian.T))
        np.testing.assert_allclose(model.manipulability(q), np.sqrt(abs(determinant)), rtol=1e-12)
        assert model.manipulability(q) > 0.0


def test_manipulability_vanishes_at_a_singular_configuration() -> None:
    """The all-zero configuration is singular, so the measure is zero.

    ``q = 0`` puts the arm in the fully stretched, wrist-aligned posture: the
    Jacobian loses rank and ``det(J J^T)`` is not even positive numerically.
    This is the case the ``determinant > 0`` guard in ``manipulability`` exists
    for.  The measure is asserted below 1e-12 rather than at exactly 0.0 because
    "zero" here means "not distinguishable from zero in double precision".
    """
    jacobian = model.jacobian(np.zeros(model.NUM_JOINTS))
    assert np.linalg.matrix_rank(jacobian) < 6
    assert model.manipulability(np.zeros(model.NUM_JOINTS)) < 1e-12


def test_forward_kinematics_returns_all_frames_and_validates_its_input() -> None:
    """``forward_kinematics`` returns nine frames and rejects a wrong shape.

    Nine, not seven: DH rows 8 and 9 are the rigid flange and tool offsets, so
    ``[7]`` is the flange and ``[8]`` the tool frame.  Indexing those rows
    correctly is load-bearing for :func:`franka_ik.model.fk_flange`, so the shape
    and the frames' rigid structure are asserted here.
    """
    q = np.zeros(model.NUM_JOINTS)
    frames = model.forward_kinematics(q)
    assert frames.shape == (model.DH_PARAMETERS.shape[0], 4, 4)
    assert model.FLANGE_ROW == 7

    # Frames 8 and 9 are the same orientation, translated along z by the DH
    # distances: the last two rows carry no joint angle.
    np.testing.assert_allclose(frames[8][:3, :3], frames[7][:3, :3], atol=1e-15)
    np.testing.assert_allclose(
        frames[8][:3, 3] - frames[7][:3, 3],
        frames[7][:3, :3] @ np.array([0, 0, 0.1034]),
        atol=1e-15,
    )

    with pytest.raises(ValueError):
        model.forward_kinematics(np.zeros(model.NUM_JOINTS - 1))
    with pytest.raises(ValueError):
        model.forward_kinematics(np.zeros((model.NUM_JOINTS, 1)))
