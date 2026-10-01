# API reference

Every name in `franka_ik.__all__`, grouped by the module that defines it, with the exact
signature and a one-line example that was run to produce the result in the comment.

Shared setup for all examples below:

```python
import numpy as np
import franka_ik as fk

q    = np.array([0.3, -0.4, 0.2, -1.2, 0.1, 1.0, 0.5])   # radians, inside the limits
pose = fk.fk_flange(q)                                   # the 4x4 flange pose of q
```

Wherever the signature shows `geometry=PAPER_GEOMETRY`, the real default is
`EquivalentGeometry(d_bs=0.333, d_se=0.316, d_ew=0.384, d_wt=0.107, offset=0.088,
bias=0.0825)` — shortened here for readability.

`franka_ik.__version__` is `"0.2.0"`.

---

## `franka_ik.model` — modified-DH model, forward kinematics, Jacobian

| symbol | signature | what it does | example |
|---|---|---|---|
| `NUM_JOINTS` | `= 7` | number of actuated joints | `fk.NUM_JOINTS` → `7` |
| `DH_PARAMETERS` | `= ndarray (9, 4)` | modified-DH rows `[a, d, alpha, theta_offset]`; rows 8–9 are the flange and tool offsets | `fk.DH_PARAMETERS[6]` → `[0.088, 0.0, 1.5708, 0.0]` (the wrist offset) |
| `LOWER_LIMITS_DEG` | `= ndarray (7,)` | joint lower limits in degrees | `fk.LOWER_LIMITS_DEG[3]` → `-175.0` |
| `UPPER_LIMITS_DEG` | `= ndarray (7,)` | joint upper limits in degrees | `fk.UPPER_LIMITS_DEG[5]` → `214.0` |
| `lower_limits()` | `() -> np.ndarray` | the same limits in radians | `np.degrees(fk.lower_limits())[0]` → `-165.0` |
| `upper_limits()` | `() -> np.ndarray` | the same limits in radians | `fk.upper_limits().shape` → `(7,)` |
| `forward_kinematics(q)` | `(q) -> np.ndarray` | all nine link frames, base first | `fk.forward_kinematics(q).shape` → `(9, 4, 4)` |
| `fk_flange(q)` | `(q) -> np.ndarray` | the frame `solve` solves for | `np.round(fk.fk_flange(q)[:3, 3], 3)` → `[0.223, 0.182, 0.891]` |
| `fk_tool(q)` | `(q) -> np.ndarray` | flange plus `TransZ(0.1034)·Rz(-π/4)` | `np.allclose(fk.fk_tool(q), fk.fk_flange(q) @ fk.model.trans_z(0.1034) @ fk.model.rot_z(-np.pi/4))` → `True` |
| `jacobian(q)` | `(q) -> np.ndarray` | 6×7 geometric Jacobian of the flange, linear part first | `fk.jacobian(q).shape` → `(6, 7)` |
| `manipulability(q)` | `(q) -> float` | Yoshikawa's `sqrt(det(J Jᵀ))` | `round(fk.manipulability(q), 4)` → `0.0436` |

`forward_kinematics` raises `ValueError` when `q` is not a seven-element sequence.

## `franka_ik.geometry` — the S-R-S-equivalent geometry

| symbol | signature | what it does | example |
|---|---|---|---|
| `EquivalentGeometry` | `(d_bs, d_se, d_ew, d_wt, offset, bias)` | frozen dataclass of the six geometric parameters | `fk.PAPER_GEOMETRY.bias` → `0.0825` |
| `PAPER_GEOMETRY` | `= EquivalentGeometry(...)` | the parameters of the published implementation | `fk.PAPER_GEOMETRY.d_wt` → `0.107` |
| `wrist_offset_angle(geometry=PAPER_GEOMETRY)` | `-> float` | `β = atan2(offset, d_wt)`, in radians | `round(np.degrees(fk.wrist_offset_angle()), 4)` → `39.4349` |
| `effective_wrist_length(geometry=PAPER_GEOMETRY)` | `-> float` | `ℓ_wt = hypot(offset, d_wt)`, in metres | `round(fk.effective_wrist_length(), 8)` → `0.1385388` |
| `wrist_correction(q7, geometry=PAPER_GEOMETRY)` | `-> np.ndarray` | `STEP1`: the 3×3 `Rz(-q7) Ry(-β) Rz(q7)` to post-multiply onto the target orientation | `fk.wrist_correction(q[6]).shape` → `(3, 3)` |
| `shoulder_to_wrist(position, corrected_rotation, geometry=PAPER_GEOMETRY)` | `-> np.ndarray` | `x_sw`, the equivalent arm's shoulder-to-wrist vector | see the snippet below |
| `q4_coefficients(distance_squared, geometry=PAPER_GEOMETRY)` | `-> Tuple[float, float, float]` | `STEP2`: `(a₂, a₁, a₀)` of the quadratic in `tan(θ₄/2)` | `tuple(round(v, 6) for v in fk.q4_coefficients(0.1))` → `(-0.068151, 0.231, 0.39)` |
| `q4_discriminant(distance_squared, geometry=PAPER_GEOMETRY)` | `-> float` | `a₁² − 4a₀a₂`; negative means out of reach | `round(fk.q4_discriminant(0.1), 6)` → `0.159677` |
| `q4_roots(distance_squared, geometry=PAPER_GEOMETRY, tolerance=1e-12)` | `-> List[float]` | **both** elbow angles, `+` root first; `[]` when out of reach, one entry for a double root | `[round(np.degrees(r), 3) for r in fk.q4_roots(0.1)]` → `[-155.607, 102.092]` |
| `equivalent_link_vectors(theta4, geometry=PAPER_GEOMETRY)` | `-> Tuple[np.ndarray, np.ndarray]` | `(l_se, l_ew)` of the equivalent arm at this `θ₄` | `fk.equivalent_link_vectors(1.0)[0]` → `array([0., 0.27093, 0.])` |

The `STEP1` chain in full, including the two lines a caller usually needs:

```python
corrected = pose[:3, :3] @ fk.wrist_correction(q[6])
p_sw      = fk.shoulder_to_wrist(pose[:3, 3], corrected)
roots     = fk.q4_roots(float(p_sw @ p_sw))      # -> [-1.2, 0.266] rad for this q
```

## `franka_ik.solver` — the analytical inverse kinematics

| symbol | signature | what it does | example |
|---|---|---|---|
| `NUM_BRANCHES` | `= 8` | how many candidates the solver enumerates | `fk.NUM_BRANCHES` → `8` |
| `IkSolution` | dataclass `(q, q4_root, phi_root, shoulder_flip, wrist_flipped, pose_error, within_limits)` | one solution plus everything known about how it was found | `fk.solve(pose, q[6])[0].label` → `'q4+ phi+ plain'` |
| `IkSolution.label` | property `-> str` | compact branch identifier, used in reports | `fk.solve(pose, q[6], within_limits_only=True)[0].describe()` → `'q4+ phi+ flip    q = [ 17.189, …] deg   pose error 3.75e-16   in limits'` |
| `IkSolution.describe(precision=3)` | `(precision=3) -> str` | one-line summary in degrees | see above |
| `wrap_to_limits(q, lower=None, upper=None, tolerance=1e-9)` | `-> Tuple[np.ndarray, np.ndarray]` | map angles into the real ranges; returns `(wrapped, inside)` | `fk.wrap_to_limits(q + 2*np.pi)[1].all()` → `True` |
| `solve_branch(pose, q7, q4_root=1, phi_root=1, shoulder_flip=False, geometry=PAPER_GEOMETRY, check_pose=True)` | `-> Optional[IkSolution]` | one of the eight branches, or `None` if it does not exist | `fk.solve_branch(pose, q[6], q4_root=-1).pose_error` → `9.02e-16` |
| `branch_solutions(pose, q7, geometry=PAPER_GEOMETRY, check_pose=True)` | `-> List[IkSolution]` | all existing branches, in a fixed order, undeduplicated | `len(fk.branch_solutions(pose, q[6]))` → `8` |
| `solve(pose, q7, geometry=PAPER_GEOMETRY, tolerance=1e-9, within_limits_only=False, deduplicate=True)` | `-> List[IkSolution]` | the useful entry point: verified, optionally limit-filtered, deduplicated | `len(fk.solve(pose, q[6], within_limits_only=True))` → `1` |
| `solve_closest(pose, q7, reference, geometry=PAPER_GEOMETRY, within_limits_only=True)` | `-> Optional[IkSolution]` | the solution nearest a reference configuration, for tracking | `np.max(np.abs(fk.solve_closest(pose, q[6], q).q - q))` → `8.33e-16` |

The three entry points differ only in how much they filter:

```python
branches = fk.branch_solutions(pose, q[6])                    # up to 8, unchecked for limits
sols     = fk.solve(pose, q[6], within_limits_only=True)      # verified poses, inside limits
best     = fk.solve_closest(pose, q[6], reference=previous_q) # one solution, near previous_q
```

`check_pose` does two jobs, which is worth knowing before switching it off: it records
`IkSolution.pose_error`, and it decides which of the two wrist candidates `(q₅, q₆)` /
`(q₅+π, −q₆)` is kept — the one that actually reaches the pose. With `check_pose=False` the
published `criteria` sign test makes that choice instead, so the branch can come back wrong
(it is the failure the published code shows at `q₇ = ±90°`, see `docs/limitations.md` §7).
`IkSolution.wrist_flipped` says which candidate was kept either way.

## `franka_ik.analysis` — studies of the solution set

| symbol | signature | what it does | example |
|---|---|---|---|
| `PoseStudy` | dataclass `(q7, outcomes, n_distinct, n_published)` | per-branch detail for one pose | `study.exists_count` → `8` |
| `PoseStudy.exists_count` | property `-> int` | how many branches produced a configuration | see above |
| `PoseStudy.describe(precision=3)` | `-> str` | multi-line per-branch report | see the snippet below |
| `study_pose(pose, q7, target=None, geometry=PAPER_GEOMETRY, tolerance=1e-9)` | `-> PoseStudy` | evaluate every branch, optionally checking whether it recovers a known target | `fk.study_pose(pose, q[6], target=q).n_distinct` → `1` |
| `CoverageReport` | dataclass `(samples, recovered_full, recovered_published, mean_solutions, min_solutions, max_solutions)` | result of a coverage study | `fk.coverage_study(samples=50, seed=0).samples` → `50` |
| `CoverageReport.full_rate` | property `-> float` | fraction recovered by all eight branches | `round(rep.full_rate, 3)` → `1.0` |
| `CoverageReport.published_rate` | property `-> float` | fraction recovered by the published four | `round(rep.published_rate, 3)` → `0.74` |
| `CoverageReport.describe(precision=3)` | `-> str` | the four-line report quoted in `docs/branch_analysis.md` | see below |
| `coverage_study(samples=300, seed=0, geometry=PAPER_GEOMETRY, use_real_limits=True)` | `-> CoverageReport` | the measurement behind the completeness claim: is a sampled configuration among the solutions? | `fk.coverage_study(samples=300, seed=0).recovered_full` → `300` |
| `SolutionCountReport` | dataclass `(samples, histogram, mean)` | histogram of solutions per pose | `cnt.histogram[2]` → `108` |
| `SolutionCountReport.describe()` | `-> str` | ASCII histogram | see below |
| `solution_count_study(samples=300, seed=0, geometry=PAPER_GEOMETRY)` | `-> SolutionCountReport` | how many distinct in-limit solutions a pose has | `round(fk.solution_count_study(samples=300, seed=0).mean, 2)` → `3.15` |
| `reachable_distance_range(geometry=PAPER_GEOMETRY)` | `-> Tuple[float, float]` | closed-form shell of `‖x_sw‖` | `tuple(round(v, 5) for v in fk.reachable_distance_range())` → `(0.06617, 0.71935)` |
| `classify_failure(pose, q7, geometry=PAPER_GEOMETRY)` | `-> str` | why a pose has no solution | `fk.classify_failure(pose, q[6])` → `'solvable'` |

```python
print(fk.study_pose(pose, q[6], target=q).describe())
```
```
joint 7 = 28.648 deg
  q4+ phi+ flip    pose error 3.75e-16   in limits, TARGET
  q4+ phi+ plain   pose error 3.75e-16
  q4- phi+ flip    pose error 9.02e-16
  ...
  distinct in-limit solutions: 1 (published four-branch subset finds 1)
```

```python
print(fk.coverage_study(samples=300, seed=0).describe())
```
```
300 random configurations
  published four-branch subset recovers the target: 265/300 (88.3 %)
  full eight-branch solver recovers the target:     300/300 (100.0 %)
  distinct in-limit solutions per pose: mean 3.15, range 1-8
```

```python
print(fk.solution_count_study(samples=300, seed=0).describe())
```
```
distinct in-limit solutions for 300 random poses (mean 3.15):
  1 solution(s):    29  ######
  2 solution(s):   108  ######################
  ...
```

---

## `franka_ik.report` and `franka_ik.viz` — figures and demo pages

The presentation layer. Neither module is re-exported at package level: import them by name,
and note that `franka_ik.viz` is the only part of the package that needs matplotlib.

| symbol | signature | what it does |
|---|---|---|
| `report.build_demo_site(out_dir="demos", samples=300, seed=0, geometry=PAPER_GEOMETRY, rich_min_solutions=6)` | `-> List[Path]` | write the three self-contained HTML pages (`index`, `branches`, `coverage`) and return their paths |
| `report.build_index_page(samples=300, seed=0, …)` | `-> str` | the HTML of the overview page |
| `report.build_branches_page(pose, q7, target=None, …)` | `-> str` | the HTML of one pose with all of its solutions drawn |
| `report.build_coverage_page(samples=300, seed=0, …)` | `-> str` | the HTML of the coverage comparison |
| `viz.plot_arm_3d(ax, q, …)` | | draw one configuration into a 3D axes |
| `viz.plot_branches_3d(pose, q7, target=None, …)` | | draw every solution of a pose on top of each other |
| `viz.plot_elbow_quadratic(distance, …)` | | plot the `STEP2` quadratic and mark its two roots |
| `viz.plot_coverage_comparison(report, …)` | | bar chart from a `CoverageReport` |
| `viz.plot_solution_histogram(report, …)` | | histogram from a `SolutionCountReport` |
| `viz.plot_joint_trajectories(times, naive, continuous, …)` | | joint trajectories: naive branch picking against `solve_closest` |
| `viz.import_mplot3d()`, `viz.use_headless_backend()`, `viz.save(figure, path, …)`, `viz.show()` | | matplotlib plumbing used by the examples |
| `viz.BRANCH_COLORS` | `= tuple` | the per-branch colour of every figure |

```python
from franka_ik.report import build_demo_site

build_demo_site("demos", samples=300, seed=0)     # writes demos/{index,branches,coverage}.html
```

## Worked examples

| file | what it walks through |
|---|---|
| `examples/01_forward_kinematics.py` | the modified-DH model, `fk_flange` vs `fk_tool`, the real joint ranges, manipulability |
| `examples/02_srs_reduction.py` | `STEP1` and `STEP2` on a real configuration: the wrist correction, and the elbow quadratic that reproduces joint 4 |
| `examples/03_branches.py` | the eight branches, the four the published solver keeps, and `solve_closest` for tracking |

Run them from the repository root, for example
`MPLBACKEND=Agg python3 examples/03_branches.py --save-dir /tmp/branches`.

---

## Branch naming

`IkSolution.label` is built from three of the four decisions that select a branch, in the order
`q4{+,-} phi{+,-} {plain,flip}` — for example `q4- phi+ flip`. The fourth decision, the internal
wrist flip, is reported separately as `IkSolution.wrist_flipped` because it is decided per
branch rather than enumerated. `phi+` is the first root of the `STEP3` sinusoid and `phi-` the
second, in the order the published implementation numbers them — the sign of `cos(q₇)` is
folded into that choice so the correspondence holds over the whole joint-7 range. See
`docs/method.md` §5.3 and §7.

## Symbols that are deliberately *not* exported

`franka_ik.geometry` also defines `rot_x`, `rot_y`, `rot_z`, `equivalent_joint_rotation`,
`link_offset` and `DEFAULT_ALPHAS`; `franka_ik.model` defines `link_transform`, `trans_x`,
`trans_z`, `joint_frames`, `limits`, `FLANGE_ROW` and `TOOL_ROTATION`; `franka_ik.analysis`
defines `BranchOutcome` and `PUBLISHED_Q4_ROOT`; and `franka_ik.report` / `franka_ik.viz` above
are whole modules outside the re-export list. These are part of the implementation (and
`PUBLISHED_Q4_ROOT` is the constant that pins the published four-branch subset), but they are
not re-exported at package level. Import them from their module if you need them:
`from franka_ik.geometry import equivalent_joint_rotation`.
