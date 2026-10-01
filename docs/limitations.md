# Limitations and failure modes

Everything in this file is either a measured number or a property of the code that can be
checked by reading it. Where a claim is empirical rather than proven, it says so.

---

## 1. What the solver promises, and what it does not

`solve(pose, q7)` returns **verified** solutions: every candidate is checked against
`model.fk_flange` and dropped if its residual exceeds `tolerance` (default `1e-9`). So a
solution you get back is right. What is *not* promised:

* that the list is complete in the mathematical sense. The measured claim is that the
  eight-branch enumeration recovers the generating configuration in 300 of 300 random trials
  (`docs/branch_analysis.md`), not that no configuration can ever be missed;
* that a solution exists. `solve` can legitimately return `[]`.
* that the list is the best one, or continuous, or collision-free. It is a discrete set.

## 2. The caller must choose joint 7, and that choice decides everything

The redundancy of a seven-axis arm is one-dimensional, and this method spends it on `q₇`. The
price is that a *reachable pose can still be unsolvable* for the `q₇` the caller asked for.
Measured on 200 poses generated from random in-limit configurations:

| `q₇` supplied | poses with at least one in-limit solution |
|---|---|
| the target's own `q₇` | 200/200 |
| target `q₇` + 2.9° | 170/200 |
| target `q₇` + 8.6° | 139/200 |
| target `q₇` + 17.2° | 119/200 |
| target `q₇` + 34.4° | 89/200 |

A *fixed* `q₇` is much worse — that is the point of the parameterisation, and also its cost:

| fixed `q₇` | poses solved (of the same 200 reachable poses) |
|---|---|
| −165° | 55/200 |
| −110° | 61/200 |
| −55° | 72/200 |
| 0° | 78/200 |
| +55° | 77/200 |
| +110° | 82/200 |
| +165° | 71/200 |

Practical reading: at a fixed `q₇` only about a third of reachable poses are solvable at all,
so `q₇` has to be tracked rather than held constant. In a tracking loop, feed the previous
`q₇` (or the previous configuration's `q₇`) and use `solve_closest` to stay on one branch; if a
solution disappears, sweep `q₇` rather than declaring the pose unreachable.

## 3. The solver works on the **flange** frame, not the tool frame

`model.fk_flange` is the frame the analytical IK solves for. The nominal tool frame adds two
rigid transforms that are easy to forget:

```math
T_{\text{tool}} = T_{\text{flange}}\;\operatorname{TransZ}(0.1034)\;R_z(-\pi/4),
```

so the conversion back is `T_flange = T_tool @ rot_z(+π/4) @ trans_z(-0.1034)`. Verified to
5.6 × 10⁻¹⁷ against `model.fk_flange` on random configurations. Dropping the 0.1034 m term is
silent: it produces a pose that is 0.1034 m away from the intended one, and `solve` happily
returns verified solutions *for that wrong pose* (1 solution instead of 3, and not the target
configuration, in a spot check). Nothing in the API can detect the mistake.

## 4. Joint limits, wrapping, and the published `limit_joints`

The Panda's limits are not symmetric — joint 4 is `[-175°, -5°]` and joint 6 is `[0°, 214°]`
(`franka_ik.model.LOWER_LIMITS_DEG`, `UPPER_LIMITS_DEG`) — so "wrap into range" is not the
usual symmetric fold.

`original/ik_ca.py::limit_joints` gets this wrong twice.

**The windows are not the robot's.** It uses `[-181°, 181°]` for the general joints,
`[-271°, 91°]` for joint 4 and `[-74°, 288°]` for joint 6. Those two special windows are 362°
wide, i.e. *wider than a full turn*, so the loop can stop at either of two representatives, and
neither of them matches the range the robot actually reports (`[-175°, -5°]` and `[0°, 214°]`).

**It hangs on a `nan`.**

```python
while True:
    if (-181 <= theta[i]) and (181 >= theta[i]): break
    elif -181 > theta[i]: theta[i] += 360
    else:                 theta[i] -= 360
```

If `theta[i]` is `nan`, every comparison is `False`, so the loop takes the `else` branch,
subtracts 360 forever and never terminates. Feeding it a `nan` and killing it after 5 seconds
finds it still spinning — this is not a slow return, it is a non-terminating loop. A robot
controller that calls the published solver with a `nan` (which is what an out-of-reach target
produces) hangs instead of reporting that the pose is unreachable.

`franka_ik.solver.wrap_to_limits` replaces it:

```python
wrap_to_limits(q)                  # -> (wrapped, inside), inside is a per-joint bool array
wrap_to_limits([nan] * 7)          # -> ValueError: joint angles must be finite
```

and, unlike the original, it never silently projects: a joint whose range cannot contain the
angle is returned at the closest wrapped value with `inside[joint] = False`, and `solve`
reports `IkSolution.within_limits` rather than dropping the information.

**Its own limit.** `wrap_to_limits` searches `q + 2πk` for `k ∈ {-2, -1, 0, 1, 2}`. An angle
more than two turns outside the range is not found and is reported as outside (verified with
`q₁ + 8π`). For joint values coming out of an analytical solver that is more than enough, but
it is not a general angle-normalisation routine.

## 5. The published `Panda.fk` cannot be evaluated numerically

A second defect of the published files, and a quieter one than the `nan` hang. `original/panda.py`
ends `fk` with

```python
Tool = self.rot_z(-pi/4)
return H @ Tool
```

and `Panda.rot_z` is built from `ca.SX_eye(4)` — a *symbolic* identity matrix. So `Panda().fk(q)`
returns a CasADi `SX` even when `q` is an ordinary NumPy array: every entry is a numeric
constant, but the type is symbolic, and

```python
np.array(Panda().fk(q), dtype=float)
# Exception: Implicit conversion of symbolic CasADi type to numeric matrix not supported.
```

It takes a while to find, because nothing about the call looks wrong and the error message
points at NumPy rather than at the model. `Panda.forward_flange(q)` does not have the problem —
it is built through `casadi.Function` and evaluates to a `DM` — and the original
`original/test.ipynb` only ever calls `forward_flange`, which is why it was never noticed.

The workaround is to wrap it once:

```python
import casadi as ca
qs = ca.SX.sym("q", 7)
tool = ca.Function("fk_tool", [qs], [Panda().fk(qs)])
np.array(tool(q), dtype=float)      # works
```

Wrapped that way it agrees with `franka_ik.model.fk_tool` to **3.33 × 10⁻¹⁶** over 100 random
in-limit configurations, so it is the *type* that is wrong in the original, not the model.

The pair of defects is worth remembering together: `limit_joints` (§4) makes a function hang,
`Panda.fk` makes one un-usable numerically, and neither raises a clear error at the point of the
mistake. Both live in the published files, both are fixed in `franka_ik/`, and both are pinned
by the tests that cross-check the library against `original/`.

## 6. Only one of the two S-R-S symmetries survives

An S-R-S arm has two two-fold symmetries: flip the shoulder, and flip the wrist. On the Panda
only the first is real. Measured over 138 random configurations:

| transformation | same flange pose? |
|---|---|
| shoulder: `(q₁, q₂, q₃) → (q₁+π, −q₂, q₃+π)` | **138/138** |
| wrist: `(q₅+π, −q₆, q₇+π)` | **0/138** |
| wrist: `(q₅+π, −q₆)` | **0/138** |

The reason is the one that runs through this whole repository: the Panda is not a true S-R-S
arm, and the offset that `STEP1` compensates is exactly what breaks the wrist symmetry. The
consequence for a caller is practical: a second solution cannot be manufactured by mirroring;
it has to come out of the branch enumeration (`shoulder_flip=True` is the only symmetry the
solver exposes).

Note that this is *not* the same thing as `IkSolution.wrist_flipped`. That flag records **which
of the two solutions of `q₆ = ±arccos(·)` was kept for a fixed `q₇`** — a choice inside the
decomposition of one branch, not a symmetry that produces a second solution (`docs/method.md`
§6). The two ideas are easy to confuse and they behave in opposite ways: the symmetry is absent
(0/138), while the wrist choice is present in every branch and simply has to be made correctly.

## 7. `q₇ = ±90°`: a defect of the published code that this library repairs

This is kept as a worked example of a limitation that turned out to be *fixable*, because the
diagnosis is more useful than the conclusion.

### What the published solver does

At exactly `q₇ = ±90°` the published implementation can return a wrong configuration or none at
all. Sweeping the whole joint-7 range — 6 random in-limit configurations × 329 values of `q₇`
from −164° to +164° × the 4 published entry points, i.e. **7896 calls** — produces 16 calls whose
returned configuration has a pose residual above 10⁻⁶, and **all 16 are at `q₇ = ±90°`**:

| trial | `q₇` | entry points that returned a wrong pose | worst pose residual |
|---|---|---|---|
| 0 | −90° | all four | 1.44 |
| 1 | +90° | `ik_ca`, `ik_ca_neg` | 1.57 |
| 3 | +90° | `ik_ca`, `ik_ca_neg` | 1.90 |
| 4 | +90° | all four | 1.73 |
| 5 | −90° | all four | 1.82 |

Everywhere else in the range the published code is exact. The residual of about 1.9 is the
signature of a wrist flipped the wrong way: `π` radians of error on one joint, which is what the
numbers are.

### Why

The published code decides between the two wrist configurations with

```python
criteria = r_47[2, 0] * cos(q7)
if criteria < 0:
    q5 += pi
    q6 = -q6
```

and `criteria = sin(q₆)·cos²(q₇)` (`docs/method.md` §6), so the quantity that carries the
decision vanishes **identically** at `q₇ = ±90°`. There the sign is decided by floating-point
noise, and it comes out wrong about half the time. The method is fine; the sign test is not.

The tangent in the `STEP3` coefficients also blows up at `±90°`, which is where the published
code's own comment points (`#. tan(q7) 为无穷则需要调整`). That is a red herring, and it can be
shown to be one: solving with the tangent-free coefficient form of §5.4 but leaving the
*published* sign test in charge of the wrist — `branch_solutions(..., check_pose=False)` — still
fails at `±90°` and nowhere else.

| `q₇`, `check_pose=False` | branches landing on a wrong pose | trials whose target was found |
|---|---|---|
| `+90°` exactly | 140/532 | 72/100 |
| `−90°` exactly | 120/532 | 77/100 |
| `+90° − 10⁻⁹` | 0/532 | 100/100 |
| random `q₇` | 0/532 | 100/100 |

So the tangent was never the problem; the sign test is.

### What `franka_ik` does instead

* The `STEP3` coefficients are evaluated as `W[2,1]·cos q₇ + W[2,0]·sin q₇`, which is the tangent
  form multiplied by `cos q₇` — exact, since it scales all three coefficients by the same
  constant — and stays finite at `±90°`.
* The wrist is not decided by a sign test at all: both candidates `(q₅, q₆)` and `(q₅+π, −q₆)`
  are evaluated and the one that reaches the pose is kept. That costs one extra forward-kinematics
  call per branch and is exact. The published criterion is still used when `check_pose=False`, so
  the two remain comparable.

Measured after the change, on 100 random in-limit targets per case:

| `q₇` | target configuration recovered | branches with a correct pose |
|---|---|---|
| `+90°` exactly | 100/100 | 532/532 |
| `−90°` exactly | 100/100 | 532/532 |
| `+90° − 10⁻⁹` | 100/100 | 532/532 |

Before the change the same measurement gave 71/100 and 75/100 with 384/532 and 400/532 correct
branches, so this was a real failure and not a precaution.
### The one thing that had to be handled carefully

Multiplying the `STEP3` coefficients by `cos q₇` negates them when `cos q₇ < 0`, which swaps the
two arm angles: `φ₊` and `φ₋` trade places for `|q₇| > 90°`. The *solution set* is unaffected
(the same eight candidates come out), but the *labels* would have silently stopped matching the
published implementation in half of the joint-7 range — a change nobody would have noticed,
because every returned pose is still correct. The root choice therefore folds in the sign of
`cos q₇`, so `phi_root = +1` means the same published branch at every `q₇`. With that in place
the label-level comparison is exact everywhere: 1200 of 1200 comparisons over 300
configurations, 684 of them with `cos q₇ > 0` and 516 with `cos q₇ < 0`, worst deviation
1.30 × 10⁻¹³ rad, none above 10⁻⁹.

## 8. Singular and degenerate configurations

`solve_branch` returns `None` — not an exception, not a wrong answer — in four situations, and
`analysis.classify_failure` names them:

| `classify_failure` result | cause |
|---|---|
| `shoulder_wrist_distance_zero` | `‖x_sw‖² ≤ 10⁻¹⁸`: the wrist sits on the shoulder |
| `outside_reachable_shell` | negative discriminant of the elbow quadratic |
| `arm_angle_singular` | `\|c₂\| > ρ` in the `STEP3` sinusoid, or a vanishing amplitude in the reference-plane solve |
| `no_valid_branch` | branches exist but none passes the pose test and the limits |

Over 500 random points placed in the workspace those categories occur in practice
(446 `outside_reachable_shell`, 27 `solvable`, 15 `arm_angle_singular`, 12 `no_valid_branch`),
so a caller should treat an empty result as normal and ask `classify_failure` why.

Not every pose supports all eight candidate branches, either. A `(q4_root, phi_root)` pair
exists only when the `STEP3` sinusoid has a solution *and* the reference-plane arc-sines are in
range, so the candidate list is often shorter than eight even for a perfectly reachable pose.
Measured over 200 random in-limit configurations, `len(branch_solutions(pose, q7))` was **4 for
124 poses and 8 for 76** — never anything in between, because when the second elbow root admits
an arm angle at all, both of its `φ` values and both shoulder flips do as well. So a short
candidate list is normal; an empty one is what means "no solution".

At a wrist singularity (`q₆ = 0`, i.e. `sin q₆ = 0`) the branches that differ only in the
wrist collapse onto each other; `solve` merges configurations that agree to within `1e-6` per
joint, so the returned list is shorter there. That is intended, but it means "the number of
solutions" is not a continuous function of the pose.

## 9. Two DH conventions, reconciled numerically rather than by inspection

`franka_ik/model.py` uses modified (Craig) DH;
`franka_ik/geometry.py` uses the standard DH rotation `Rz(θ)Rx(α)` of the S-R-S paper
(`franka_ik.geometry.DEFAULT_ALPHAS`). The repository deliberately does **not** try to rewrite
one into the other. It reduces the arm in the convention the derivation was written in, and
checks the result against the modified-DH forward kinematics:

* the poses agree to 3.3 × 10⁻¹⁶ (`fk_flange` and `fk_tool`, 300 random configurations);
* the branch-by-branch comparison against the published CasADi implementation agrees to
  1.30 × 10⁻¹³ rad over 1200 comparisons spanning the whole joint-7 range.

This is a design choice with a cost: a reader who assumes one convention while reading the
other gets no type error and no exception, only wrong numbers. `model.jacobian` illustrates the
subtlety — it matches the published `Panda.jacobian_flange` to 5.6 × 10⁻¹⁶, while the published
`Panda.jacobian` differentiates the **tool** frame and differs from it by 0.103.

### The sign slip that this design is meant to catch

`geometry.rot_y` uses `[[c, 0, −s], [0, 1, 0], [s, 0, c]]`. Its docstring records why the sign
is called out: during the port, flipping it mirrored the wrist correction and **every** branch
returned a wrong pose — a pose residual of 0.865 on the pose where it was found, against
1 × 10⁻¹⁶ when correct. Re-running that experiment (flip `rot_y` inside the correction, solve 20
random poses) reproduces the failure at the same order of magnitude: worst residual 1.25, i.e.
a completely wrong pose. Nothing about the code is wrong when this happens; the *numbers* are.
The verification habit that catches it is the one the library already uses: always look at
`IkSolution.pose_error`.

## 10. Numerical tolerances you may want to change

| tolerance | default | where | what it means |
|---|---|---|---|
| pose residual | `1e-9` | `solve(tolerance=…)` | a branch is dropped below this |
| same-configuration | `1e-6` per joint | `solve(deduplicate=True)` | two branches count as one |
| root coincidence | `1e-12` relative | `q4_roots(tolerance=…)` | a double root is reported once |
| limit slack | `1e-9` | `wrap_to_limits(tolerance=…)` | how far outside a limit still counts as inside |

Changing `solve`'s pose tolerance is the one that changes results rather than presentation: the
measured residual distribution has median 4.4 × 10⁻¹⁶ and worst 2.0 × 10⁻¹⁴ over 1104 branch
evaluations, so the default `1e-9` sits about five orders of magnitude above the noise and is
not at risk of rejecting good branches.

## 11. What is deliberately not implemented

* **Redundancy resolution.** The S-R-S paper's second half — using the remaining degree of
  freedom to stay away from joint limits while following a task — is not implemented. This
  repository stops at "here are the configurations for this `q₇`". Choosing among them is
  `solve_closest` and nothing more.
* **Velocity-level IK, dynamics, collision checking.** Out of scope; only `model.jacobian` and
  `model.manipulability` are provided, and they are for analysis, not for a controller.
* **Tool-frame IK.** See §3: convert first.
* **Non-`PAPER_GEOMETRY` robots.** `EquivalentGeometry` is a parameter, so the reduction itself
  is not Panda-specific, but nothing else is tested with other values.

## 12. `q₂ = 0` is a coordinate singularity of the reduction

**Severity: real, but it needs an exactly-zero joint angle to trigger.**

At `q₂ = 0` the shoulder loses a degree of freedom: joints 1 and 3 become coaxial, so
their individual values stop being determined by the pose — only their combination is
meaningful — and the closed form's reconstruction of the shoulder orientation collapses.
The matrix element it reads joint 1 from is `r₀₃[1,1] = -sin θ₁ sin θ₂`, which is zero for
*any* `θ₁` when `sin θ₂ = 0`, so `atan2(0, 0)` returns 0 and the branch comes back with
`q₁ = 0` instead of the true value. Because every branch is verified against the forward
kinematics before it is returned, that wrong branch is discarded — and if it was the only
one inside the joint limits, the solver reports no solution at all.

Reproducer (a configuration inside the limits, whose pose is exact):

```python
import numpy as np
from franka_ik import model, solve, classify_failure

q = np.radians([36.0, 0.0, -24.0, -100.0, 18.0, 126.0, 48.0])
pose = model.fk_flange(q)

solve(pose, float(q[6]), within_limits_only=True)   # -> []  (empty)
classify_failure(pose, float(q[6]))                 # -> 'no_valid_branch'
```

The eight branches come back with `q₂ = 0` and a pose residual of `2.55e-2`, or with
`q₂ = ±106.23°` / `±73.70°` and a residual of `1e-16` but **outside** the `[-100°, 100°]`
range of joint 2. Perturbing the generating configuration by `±0.01°` in joint 2 restores
two solutions, including the target itself.

Two things are worth saying about this.

* **The published implementation behaves the same way.** It uses the identical
  reconstruction, so this is a property of the derivation, not of this reimplementation.
  It is not one of the four defects listed in §4, §5 and §7 that `franka_ik` repairs — it
  is a limitation that remains.
* **The coverage study cannot see it.** The condition `q₂ = 0` has measure zero, so a
  random sample never hits it (`coverage_study(300, seed=0)` still reports 300/300). It
  shows up in practice when a *planned* path crosses the singular value, which is how it
  was found: a joint-space interpolation in `examples/06_tracking.py` passed through
  `q₂ = 0`. That example therefore keeps joint 2 away from zero and says so.

A proper fix is a dedicated branch for `sin θ₂ ≈ 0`, in the spirit of the remedy Shimizu
et al. give for the shoulder singularity in their Section II-E: fix one of the two
indeterminate joints and recover the other from the shoulder position. That is a real piece
of work and it is deliberately not attempted here; the honest statement is that the solver
is complete for `q₂ ≠ 0` and singular at `q₂ = 0`.
