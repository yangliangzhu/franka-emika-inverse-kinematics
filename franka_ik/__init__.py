"""Analytical inverse kinematics for the Franka Emika Panda.

The repository implements the method derived in ``franka解析反解方法.pdf``: reduce
the Panda to an equivalent S-R-S arm and solve that arm in closed form with the
method of Shimizu et al. (2008), parameterised by joint 7.

Modules
-------
:mod:`franka_ik.model`      modified-DH model, forward kinematics, Jacobian
:mod:`franka_ik.geometry`   the SRS-equivalent geometry and the elbow quadratic
:mod:`franka_ik.solver`     the eight-branch analytical inverse kinematics
:mod:`franka_ik.analysis`   studies of the solution set (coverage, counts)
:mod:`franka_ik.numerical`  an independent optimiser, used to test completeness
"""

from __future__ import annotations

__version__ = "0.2.0"

from .analysis import (
    PUBLISHED_Q4_ROOT,
    BranchOutcome,
    CoverageReport,
    PoseStudy,
    SolutionCountReport,
    classify_failure,
    coverage_study,
    reachable_distance_range,
    solution_count_study,
    study_pose,
)
from .geometry import (
    PAPER_GEOMETRY,
    EquivalentGeometry,
    effective_wrist_length,
    equivalent_link_vectors,
    link_offset,
    q4_coefficients,
    q4_discriminant,
    q4_roots,
    shoulder_to_wrist,
    wrist_correction,
    wrist_offset_angle,
)
from .model import (
    DH_PARAMETERS,
    LOWER_LIMITS_DEG,
    NUM_JOINTS,
    TOOL_ROTATION,
    UPPER_LIMITS_DEG,
    fk_flange,
    fk_tool,
    forward_kinematics,
    jacobian,
    joint_frames,
    lower_limits,
    manipulability,
    upper_limits,
)
from .numerical import (
    CompletenessReport,
    completeness_check,
    numerical_ik,
    symbolic_forward_kinematics,
)
from .solver import (
    NUM_BRANCHES,
    IkSolution,
    branch_solutions,
    solve,
    solve_branch,
    solve_closest,
    wrap_to_limits,
)

__all__ = [
    "__version__",
    # model
    "NUM_JOINTS",
    "DH_PARAMETERS",
    "LOWER_LIMITS_DEG",
    "UPPER_LIMITS_DEG",
    "lower_limits",
    "upper_limits",
    "forward_kinematics",
    "fk_flange",
    "fk_tool",
    "jacobian",
    "manipulability",
    "joint_frames",
    "TOOL_ROTATION",
    # geometry
    "EquivalentGeometry",
    "PAPER_GEOMETRY",
    "wrist_offset_angle",
    "effective_wrist_length",
    "wrist_correction",
    "shoulder_to_wrist",
    "q4_coefficients",
    "q4_discriminant",
    "q4_roots",
    "equivalent_link_vectors",
    "link_offset",
    # solver
    "NUM_BRANCHES",
    "IkSolution",
    "wrap_to_limits",
    "solve_branch",
    "branch_solutions",
    "solve",
    "solve_closest",
    # analysis
    "PUBLISHED_Q4_ROOT",
    "BranchOutcome",
    "PoseStudy",
    "study_pose",
    "CoverageReport",
    "coverage_study",
    "SolutionCountReport",
    "solution_count_study",
    "reachable_distance_range",
    "classify_failure",
    # numerical cross-check (imports CasADi lazily, never at import time)
    "CompletenessReport",
    "completeness_check",
    "numerical_ik",
    "symbolic_forward_kinematics",
]
