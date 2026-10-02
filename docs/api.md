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

## `franka_ik.numerical` — the independent optimiser, and the completeness check

Nothing here is needed to *use* the solver. It exists so that "these eight branches are all of
them" can be tested against something that does not share the derivation's assumptions. CasADi is
imported inside the functions, never at module import, so `import franka_ik` stays CasADi-free.

| symbol | signature | what it does | example |
|---|---|---|---|
| `DEFAULT_SOLUTION_TOLERANCE` | `= 1e-8` | largest entry of `fk_flange(q) - pose` at which an IPOPT result counts as reaching the pose | `fk.DEFAULT_SOLUTION_TOLERANCE` → `1e-08` |
| `symbolic_forward_kinematics()` | `-> casadi.Function` | the modified-DH model rebuilt in CasADi, for the optimiser | `f(q)` reproduces `fk_flange(q)` to `1e-12` |
| `numeric_solver()` | `-> casadi.Function` | the NLP and its IPOPT solver, built once and cached | `numeric_solver()` |
| `numerical_ik(pose, q7, starts=400, seed=0, tolerance=DEFAULT_SOLUTION_TOLERANCE)` | `-> List[np.ndarray]` | configurations reaching `pose` with joint 7 fixed, found from `starts` random in-limit starting points | `len(fk.numerical_ik(pose, q7))` → `1` for a typical pose |
| `configuration_distance(a, b)` | `-> float` | largest whole-turn-free `|Δq|`, in radians, over the seven joints | `round(fk.numerical.configuration_distance(q, q), 12)` → `0.0` |
| `CompletenessReport` | dataclass `(q7, numerical, analytical, matched, unmatched, unreachable, distances_deg)` | the two solution sets and how they line up | — |
| `completeness_check(pose, q7, starts=400, seed=0, tolerance=1e-3, counter_example_tolerance=0.05)` | `-> CompletenessReport` | the completeness test: `unmatched` holds the IPOPT solutions further than `counter_example_tolerance` from **every** branch | `rep.unmatched` → `[]` over 20 poses |

`configuration_distance` is in the module's `__all__` but is not re-exported at package level;
reach it as `fk.numerical.configuration_distance`.

The two tolerances are not redundant. IPOPT drives the *pose* error to zero, and near a kinematic
singularity a joint can move a long way without moving the tool, so the optimiser stops a fraction
of a degree from the configuration it is standing on. `tolerance` is how close a numerical
solution has to be to count as *being* a branch; `distances_deg` keeps the full list so the margin
is visible rather than hidden; `counter_example_tolerance` is where a result stops being
"converged imprecisely" and becomes an unexplained configuration. Measured over 10 poses × 400
starts: 41 solutions, 0 counter-examples, worst distance 0.0855 deg on one near-singular pose
where that offset moves the pose by 7.8 × 10⁻⁹. See `docs/provenance.md` §4.

---

## `franka_ik.swift_viz` and `franka_ik.swift_app` — the 3D viewers

Not re-exported at package level and not needed to use the solver: these are what
`examples/07`-`11` are built from, and they need the optional ``viz`` extra
(``uv sync --extra viz``). Every function imports `roboticstoolbox`, `swift` and
`spatialgeometry` inside itself, never at module level, so `import franka_ik` stays free
of the visualisation stack.

| symbol | signature | what it does | example |
|---|---|---|---|
| `require_viz()` | `-> VizDependencies` | the three modules as a named tuple `(rtb, swift, sg)` | `sv.require_viz().sg` |
| `find_franka_description()` | `-> Optional[Path]` | searches `FRANKA_IK_FRANKA_DESCRIPTION` and the usual checkout locations | — |
| `find_urdf()` | `-> Optional[Path]` | `FRANKA_IK_URDF`, else a `.urdf` inside a `franka_description` checkout, else `None` | — |
| `load_arm(urdf=None, check=True)` | `-> Tuple[object, Optional[Path]]` | loads the bundled Panda description, or a URDF, and verifies it | `arm, path = sv.load_arm()` |
| `check_kinematics(arm, samples=5)` | `-> float` | largest disagreement with `fk_tool` over the 4x4; **raises** `ArmKinematicsError` above `1e-3` | `sv.check_kinematics(arm)` → `1.7e-16` |
| `ArmKinematicsError` | `RuntimeError` | raised when the model is not this arm | — |
| `arm_keypoints(q)` | `-> List[np.ndarray]` | nine points: base, the seven joint origins, the tool | `len(sv.arm_keypoints(q))` → `9` |
| `joint_axes(q)` | `-> List[Tuple[np.ndarray, np.ndarray]]` | `(origin, unit axis)` per joint, from the DH frames | `sv.joint_axes(q)[3]` |
| `ArmSkeleton(radius_scale=1.0, alpha=1.0, colour=None, show_joint_axes=False, name=None)` | class | eight capsule segments and nine spheres; `.shapes` to add, `.update(q)` to move | `sv.ArmSkeleton(alpha=0.22, colour=[0.2, 0.45, 0.85])` |
| `link_collision_shapes(arm)` | `-> List[object]` | the loaded model's own collision primitives -- 30 cylinders and spheres from the bundled Panda description, so a realistic arm with **no URDF and no meshes** | `len(sv.link_collision_shapes(arm))` → `30` |
| `update_link_collision_shapes(shapes, arm, q)` | `-> None` | places them at `link_pose @ shape_pose` using ``arm.fkine_all`` | — |
| `arm_plane_outline(q, radius=0.004)` | `-> Optional[object]` | closed outline through shoulder, elbow, wrist | `sv.arm_plane_outline(q)` |
| `manipulability_ellipsoid(q, scale=0.06, centre=None)` | `-> object` | the ellipsoid, built once at a fixed size | — |
| `update_manipulability_ellipsoid(ellipsoid, q, centre=None)` | `-> float` | moves it and returns `model.manipulability(q)` | — |
| `frame_axes(pose, length=0.12)` | `-> object` | the axes of any 4x4 frame | `sv.frame_axes(sv._se3().Trans([0,0,1]))` |
| `flange_axes(q, length=0.12)` | `-> object` | the **flange** frame: the pose the solver takes and returns, and the frame every residual is measured in | `sv.flange_axes(q)` |
| `tool_axes(q, length=0.12)` | `-> object` | the factory **tool** frame: `fk_flange @ TransZ(0.1034) @ R_z(-45°)`, i.e. 0.1034 m past the flange | `sv.tool_axes(q)` |
| `tool_stem(q, radius=0.0035)` / `update_tool_stem(stem, q)` | `-> object` / `-> None` | a thin capsule covering the flange-to-tool 0.1034 m, so a tool marker is not left floating | — |
| `reachable_shell_markers(count=300, seed=0, radius=0.004, inner=False)` | `-> List[object]` | markers exactly on the closed-form shell | `len(sv.reachable_shell_markers())` → `300` |
| `add_shapes(env, shapes)` | `-> int` | adds shapes one at a time, because `Swift.add` silently ignores a list; each one blocks until the browser mounts it | `sv.add_shapes(env, skeleton.shapes)` → `17` |
| `add_cloud(env, shapes, name=None)` | `-> AssemblyHandle` | adds a **group** as one assembly: one message instead of one per shape, and `env.remove(handle)` takes it off again. The group can still be moved; a colour must be set before it goes in | a 150-marker cloud: 87 s one at a time, under a second as a group |
| `camera_presets()` / `apply_camera(env, name)` | `-> Dict` / `-> None` | four presets, and applying one | `sorted(sv.camera_presets())` → `['front', 'iso', 'side', 'top']` |
| `launch_env(headless=False, browser=None, realtime=True)` | `-> object` | launches Swift, with the install/browser hint attached to any failure | — |
| `detect_browser()` / `is_wsl()` | `-> Optional[str]` / `-> bool` | browser discovery, for WSL | — |

```python
from franka_ik import swift_viz as sv

arm, urdf = sv.load_arm()          # bundled Panda description; no download
env = sv.launch_env(headless=True)
skeleton = sv.ArmSkeleton()
sv.add_shapes(env, skeleton.shapes)  # one shape at a time: Swift.add swallows a list
skeleton.update(q)                 # pose writes only: lengths and radii are fixed
env.close()
```

```python
from franka_ik import swift_app

q, q7 = swift_app.resolve_pose(args)      # --pose / --q7, in radians
env, skeleton = swift_app.scene(args)     # a launched env with one skeleton in it
```

| `franka_ik.swift_app` symbol | signature | what it does |
|---|---|---|
| `ArmShapes` | class | one arm, either a skeleton or the collision geometry, with `.shapes` and `.update(q)` either way |
| `build_arm_shapes(args, arm=None, *, alpha=None)` | `-> ArmShapes` | builds the arm that ``--model skeleton|collision`` asks for; ``alpha`` overrides ``--mesh-alpha``, for a faded reference arm |
| `scene(args, arm=None)` | `-> Tuple[object, ArmShapes]` | launches Swift with one arm already in it |
| `POSES` | `Dict[str, List[float]]` | the four sample configurations, in degrees |
| `CAMERA_CHOICES` | `Tuple[str, ...]` | the camera presets the examples put on a radio, in listed order |
| `resolve_pose(args)` | `-> Tuple[np.ndarray, float]` | ``(q, q7)`` in radians from ``--pose`` / ``--q7`` |
| `joint7_window(pose, fallback=0.0, samples=361)` | `-> Tuple[float, float]` | the joint-7 interval, in degrees, that still reaches ``pose`` inside the limits -- found by asking the solver, because it is usually much narrower than the joint's own range |
| `common_parser(description)` | `-> ArgumentParser` | the shared flags, ``--model`` included |
| `add_readout(env, lines, name="readout", elements=None)` | `-> Label` | adds a text readout and returns it; `set_readout(label, lines)` replaces its lines |
| `add_slider(env, low=, high=, value=, label=, step=, unit=, precision=, name=, elements=)` | `-> Slider` | a slider whose **callback does nothing**: `interaction_loop` reads `slider.value`, which is the live value |
| `read_slider(slider, fallback)` | `-> float` | the slider's live value, or `fallback` before the browser has sent one |
| `add_radio(env, label=, options=, on_select=, checked=, name=, elements=)` | `-> Radio` | a radio group calling ``on_select(index)``; Swift's own empty first event is ignored |
| `add_button(env, label, on_click, elements=None)` | `-> Button` | a button; Swift fires the callback once as the page attaches |
| `add_camera_radio(env, initial="iso", elements=None)` | `-> Radio` | the camera-preset radio, with `initial` selected |
| `interaction_loop(env, slider=None, initial=None, on_change=None, on_frame=None, steps=None, dt=0.05, tolerance=1e-9)` | `-> None` | `hold` plus a slider: renders every frame, calls `on_change(value)` when the live value moves, and calls `on_frame(value)` every frame -- a returned value is written back to the slider, which is how a play button sweeps |

The interaction helpers exist because Swift's UI is not a plain event source, and the slider is
the sharp end of that. **Swift's slider JavaScript assigns `value` before `min`/`max`**
(`swift/public/js/ui.js`, `Slider.update`) while its markup starts the input at `0..100`, so a
range that does not contain 0 has its initial value clamped to the nearer end and reported once,
on attach, as a change -- measured, a joint-7 slider built with `value=-58.78` over
`[-77.92, -50.42]` came back as `-50.42`, its maximum. Reading `.value` instead of the callback
argument is half the answer; `add_slider` and `interaction_loop` write the intended value back
once the element is in the scene, which sticks because by then the browser has the real range.
A value written from Python **is** visible to the next read, in both directions measured:
`42.0` reads back as `42`, and a drag to `-70` arrives as `-70`.
`examples/10_swift_singularities.py` fixes the other end of the design: its slider has
`step=1e-5` and brackets a failure window `1e-4` degrees wide, so `interaction_loop`'s `tolerance`
has to be below the step or the feature is invisible. `tests/test_swift_app.py` pins all of it
without a browser.

```python
arm, urdf = sv.load_arm()
shapes = sv.link_collision_shapes(arm)          # 30 primitives, no URDF involved
sv.add_shapes(env, shapes)
sv.update_link_collision_shapes(shapes, arm, q)
```

Three behaviours are deliberate and easy to mistake for bugs. `update` writes **poses only** --
a length or radius written after construction marks the shape changed, and Swift's headless
client does not acknowledge that update -- so anything whose size must change is rebuilt through
the public `add`/`remove` API. `arm_keypoints` draws the **tool** frame, not the flange: the
flange lies 0.107 m along joint 7's axis, inside the tool stem. And `check_kinematics` **raises**
rather than warns when a URDF turns out to be a different arm, because the alternative is a
plausible picture of the wrong robot.

The three end frames are worth keeping apart, because mixing them is visible *only* as a marker
hanging in space, which is how it was reported. Measured (pinned by `tests/test_swift_viz.py`):

| frame | where it is | what draws it |
|---|---|---|
| wrist (`forward_kinematics[6]`) | joint 7's origin | the skeleton's second-to-last key point, the meshes' `link7` |
| **flange** (`fk_flange`) | 0.107 m further along the flange's `z` (the DH `d7`) | **the pose marker** in every example, since this is the pose the solver takes |
| tool (`fk_tool`) | 0.1034 m past the flange, rotated −45° about its `z` | the small marker, with `tool_stem` drawing the distance |

The meshes draw neither offset: `link7.dae`'s geometry reaches `z = +0.1068` in the link7 frame,
i.e. the flange and no more (measured with `trimesh`, an ad-hoc install: `trimesh.load(...).bounds`).
The skeleton's last segment, on the other hand, is the tool stem, so it ends 0.1034 m beyond the
flange. Both facts are why the examples draw *both* frames rather than choosing one.

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
defines `BranchOutcome` and `PUBLISHED_Q4_ROOT`; and `franka_ik.report`, `franka_ik.viz`,
`franka_ik.swift_viz` and `franka_ik.swift_app` are whole modules outside the re-export list --
`franka_ik.swift_app` in particular is imported by the examples for its `POSES` table and its
argument parsing, not by the library. These are part of the implementation (and
`PUBLISHED_Q4_ROOT` is the constant that pins the published four-branch subset), but they are
not re-exported at package level. Import them from their module if you need them:
`from franka_ik.geometry import equivalent_joint_rotation`.
