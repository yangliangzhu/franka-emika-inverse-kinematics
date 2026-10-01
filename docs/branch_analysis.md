# Branch analysis: the published solver has four branches, the arm has eight

The 2020 derivation is correct. The **branch enumeration built on top of it is incomplete**, and
this document is the measurement that shows it: the elbow quadratic of `STEP2` has two roots,
the published code keeps one of them, and keeping both takes the target-recovery rate from
**88.3 %** to **100 %**.

The framing matters, so it is stated once, up front: this is a *completion* of the original
method, not a refutation of it. Every configuration the published four-branch solver returns is
correct to machine precision (pose residual ≤ 2.0 × 10⁻¹⁴); it simply does not return all of
them.

---

## 1. What the published implementation believes

`original/README.md` describes two solvers, each with two entry points, and
`docs/original_notes_zh.md` states the count outright: four.

| evidence | what it says |
|---|---|
| `original/ik_ca.py` | branch 1, with the two signs of joint 2 (`ik_ca`, `ik_ca_neg`) |
| `original/ik_ca2.py` | branch 2, the same two signs (`ik_ca`, `ik_ca_neg`) |
| `docs/original_notes_zh.md` | "作者目前认为 franka 解析反解存在四种可能的分支" — the author believed there were four |

Two files × two entry points = four. The same four appear in `franka_ik` as the
`q4_root = +1` half of the branch table, which is what `analysis.PUBLISHED_Q4_ROOT` names.

## 2. Why the obvious test cannot detect the problem

The tempting test is *"does the returned configuration reach the target pose?"* — and it is
useless here, because **every** branch satisfies it. The paper's derivation does not produce
approximate answers that can be spotted by a residual; it produces a discrete set of exact
solutions, and a solver that computes four of eight of them computes four *right* answers.

The only test that catches a missing branch is therefore:

> take a configuration `q`, compute `T = fk_flange(q)`, solve for `T` **at the same joint 7**,
> and ask whether `q` itself is among the returned solutions.

A solver that misses the branch containing `q` fails that question while passing every
pose-residual check. Both numbers below are from that test
(`analysis.coverage_study`), on 300 configurations drawn uniformly inside the real joint limits
of the Panda, seed 0, each solved at its own `q₇`:

```python
from franka_ik import coverage_study
print(coverage_study(samples=300, seed=0).describe())
```

```
300 random configurations
  published four-branch subset recovers the target: 265/300 (88.3 %)
  full eight-branch solver recovers the target:     300/300 (100.0 %)
  distinct in-limit solutions per pose: mean 3.15, range 1-8
```

## 3. Where the missing 11.7 % go

The published code's elbow step is a single line, in both files:

```python
tan_half_q4 = (a1 + ca.sqrt(a1**2 - 4 * a0 * a2)) / (2 * a2)     # original/ik_ca.py, line 104
```

That is the `+` root of the quadratic in `tan(θ₄/2)` derived in `STEP2`
(`docs/method.md` §4). The `−` root is never evaluated, so half of the elbow configurations are
unreachable for any pose.

The measurement says exactly that, with no residue. For each of the same 300 targets, ask which
elbow root reproduces the configuration that generated the pose:

| root of the `STEP2` quadratic the target sits on | targets | recovered by the published subset |
|---|---|---|
| `+` root (`q4_root = +1`) | **265** | yes — 265 |
| `−` root (`q4_root = −1`) | **35** | no — 0 |
| neither | 0 | — |

265 and 35. The published subset's 265 successes are precisely the targets on the `+` root, and
its 35 failures are precisely the targets on the `−` root. The two roots are the analogue of
`θ₄ = ±arccos(·)` in the pure S-R-S case: they are the elbow on either side of the
shoulder–wrist line. With the Panda's bias they are not symmetric — one sampled pose gave
−70.38° and +16.87° where the bias-free analogue would give ±34.79° — but the role is the same.

### 3.1 What a missing branch costs, in one pose

Missing half of the elbow configurations is not only a lost choice of redundancy resolution.
It can make the solver report a reachable pose as unreachable.

Over 150 random in-limit configurations (seed 0) the eight-branch solver returns **483**
in-limit solutions in total, and the published subset returns **393** — 90 solutions, **18.6 %**
of the set, simply absent. And for one of those 150 poses the published subset returns
**nothing at all**, while the arm reaches that pose in four different ways:

```python
q = np.radians([-161.01, -3.40, -104.70, -9.82, 131.24, 205.58, 34.28])
pose, q7 = fk.fk_flange(q), q[6]

fk.solve(pose, q7, within_limits_only=True)                          # 4 solutions
[s for s in fk.solve(pose, q7, within_limits_only=True)
 if s.q4_root == analysis.PUBLISHED_Q4_ROOT]                         # []  -- empty
```

The four solutions are all on the `−` root (`q4- phi+ plain`, `q4- phi+ flip`, `q4- phi- plain`,
`q4- phi- flip`), with pose residuals 3.3 × 10⁻¹⁶, 3.3 × 10⁻¹⁶, 1.8 × 10⁻¹⁵ and 1.8 × 10⁻¹⁵, and
one of them *is* the configuration the pose came from. A controller using only the published
branches would classify this pose as out of reach.

One pose in 150 is about 0.7 %, which is the same order as the 35-in-300 elbow-root statistic
above: the pose is reachable only through the discarded half. The useful lesson is that a
solver's *failure* signal ("no solution") is only as trustworthy as its branch enumeration.

## 4. The full branch set, and what it buys

`franka_ik.solver` enumerates `2 (q4_root) × 2 (phi_root) × 2 (shoulder_flip) = 8` candidates
(`NUM_BRANCHES = 8`); the wrist posture is decided per branch rather than enumerated (see
`docs/method.md` §6). Keeping both roots of the quadratic is the entire difference between the
published solver and this one.

| measurement | published four | eight branches |
|---|---|---|
| target configuration recovered (300 samples, seed 0) | 265/300 = **88.3 %** | 300/300 = **100 %** |
| pose residual of the solutions returned | ≤ 2.0 × 10⁻¹⁴ | ≤ 2.0 × 10⁻¹⁴ |
| branches compared against the published implementation | 1200/1200 matched, worst deviation 1.30 × 10⁻¹³ rad | — |
| runtime for all candidates, pure NumPy | — | ≈ 1.3 ms per pose |

The number of distinct in-limit solution sets is itself worth recording, because it is a
property of the arm and not of any solver (`analysis.solution_count_study`, 300 poses,
seed 0):

```
distinct in-limit solutions for 300 random poses (mean 3.15):
  1 solution(s):    29  ######
  2 solution(s):   108  ######################
  3 solution(s):    37  #######
  4 solution(s):    92  ##################
  5 solution(s):     7  #
  6 solution(s):     9  ##
  7 solution(s):    12  ##
  8 solution(s):     6  #
```

A typical pose has about three reachable solutions at a given `q₇`; the range is 1 to 8, and the
maximum of 8 matches the size of the branch set exactly — no pose in the sample produced more
distinct in-limit configurations than the solver can enumerate. That is the empirical evidence
for calling the eight-enumeration *complete*; it is not a proof, and §6 says so.

## 5. What the published branches are worth

Nothing in this analysis says the published branches are wrong. Measured against the published
CasADi implementation, comparison by comparison, over 300 poses × 4 branches:

* **1200/1200 comparisons matched**, with a worst joint-wise deviation of **1.30 × 10⁻¹³ rad**
  — 684 comparisons with `cos q₇ > 0` and 516 with `cos q₇ < 0`, so the correspondence holds
  over the whole joint-7 range and not only in the middle of it;
* no comparison was skipped — the library produced a solution everywhere the originals did;
* the pose residual of the returned branches is machine precision (median 4.4 × 10⁻¹⁶, worst
  2.0 × 10⁻¹⁴ over 1104 branch evaluations).

They are four correct branches out of eight. That is why the fix is an *addition* — `q4_roots`
returns both roots and `analysis.PUBLISHED_Q4_ROOT = 1` selects the published half, so the
original behaviour is still available and is still tested.

## 6. Reproducing these numbers

```bash
python3 - <<'PY'
from franka_ik import analysis
print(analysis.coverage_study(samples=300, seed=0).describe())
print(analysis.solution_count_study(samples=300, seed=0).describe())
print(analysis.reachable_distance_range())
PY
```

The coverage study is self-contained: it samples inside the real joint limits, builds the pose
with `franka_ik.model.fk_flange`, solves, and compares configurations modulo whole turns. The
comparison against `original/` needs CasADi and the `original/` directory on `sys.path`, which
is what the repository's `conftest.py` arranges for the test suite.

The gap is not an artefact of one sample. Repeating the study at another sample size and with
other seeds, all with targets drawn uniformly inside the real joint limits:

| samples | seed | published four | full eight | mean distinct solutions |
|---|---|---|---|---|
| 300 | 0 | 265/300 = **88.3 %** | 300/300 = 100 % | 3.15 |
| 300 | 1 | 261/300 = 87.0 % | 300/300 = 100 % | 3.11 |
| 300 | 2 | 258/300 = 86.0 % | 300/300 = 100 % | 3.28 |
| 600 | 0 | 531/600 = **88.5 %** | 600/600 = 100 % | 3.13 |
| 600 | 1 | 528/600 = 88.0 % | 600/600 = 100 % | 3.11 |
| 600 | 2 | 519/600 = 86.5 % | 600/600 = 100 % | 3.23 |

The published subset loses roughly one configuration in eight, consistently, and the full
eight-branch solver loses none at any seed.

Two caveats on the numbers:

* An earlier revision of `solver.py`'s module docstring quoted 375/600 for the published
  subset at `samples=600`. That figure was written before it was measured and does not
  reproduce — the measurement is **531/600**, as in the table above — and the docstring now
  carries the measured values. It is mentioned here only because a reader who saw the old
  text in the git history should not go looking for a different experiment. (The alternative
  normal-and-wrap sampling, `use_real_limits=False`, *is* a different experiment: it
  concentrates targets near the middle of the range and gives 230/600 for the published
  subset, so it is not a substitute for the uniform draw.)
* 100 % means "100 % of the targets in this sample", not a theorem. A proof of completeness
  would have to argue that the four two-way choices of §7 of `docs/method.md` span the solution
  set; the maximum of eight distinct in-limit solutions per pose in §4 is consistent with that
  but does not establish it.

## 7. A solver that shares nothing with it agrees

The completeness claim would be weak if it rested only on a study that starts
from configurations the analytical solver is asked to reproduce.  A solver that
missed an entire branch would still return correct poses, and a coverage study
would never notice.

`franka_ik/numerical.py` therefore re-solves the same problem with machinery that
shares no code and no assumption with the analytical one: the kinematics rebuilt
symbolically in CasADi, "reach this pose with joint 7 at this value" stated as a
constrained least-squares program, and IPOPT run from many random starting points
inside the joint limits.  Anything it finds that the eight branches do not
produce would be a counter-example.

```python
from franka_ik import model, completeness_check
q = ...                                    # any in-limit configuration
report = completeness_check(model.fk_flange(q), float(q[6]), starts=300)
print(report.describe())                   # optimiser n, analytical m, matched n
print(report.worst_distance_deg)           # how far apart they ever were
```

Measured over 15 random poses: the optimiser returned **56** configurations, the
analytical solver **56**, and there were **0 counter-examples**.  The largest
distance from a numerical solution to its nearest analytical branch was
**0.0854°**, and it is worth being precise about what that residual is: IPOPT
drives the *pose* error to zero, so near a kinematic singularity — where a joint
can move a little without moving the tool — it stops a fraction of a degree away
from the configuration it is standing on.  It is a property of the optimiser, not
a missing branch, which is why the module reports the whole distance list instead
of hiding the margin in one tolerance.

## 8. The other defects found on the way

The missing elbow root is the headline, but the same cross-check turned up three more defects in
the published files. All are documented in `docs/limitations.md`, and the short version is:

* **`limit_joints` hangs on a `nan`.** `original/ik_ca.py`'s `while True` compares `nan` against
  its bounds, every comparison is `False`, and it subtracts 360 forever — a controller that
  hands it an unreachable target waits instead of getting a failure. Measured with a 5-second
  timeout, the call is still spinning when it is killed. `franka_ik.solver.wrap_to_limits`
  replaces it and raises `ValueError` on non-finite input.
* **The joint-limit windows are not the robot's.** Joints 4 and 6 get windows 362° wide
  (`[-271°, 91°]` and `[-74°, 288°]`), i.e. wider than a full turn, and neither matches the real
  `[-175°, -5°]` and `[0°, 214°]`.
* **`Panda.fk` cannot be evaluated numerically**: it returns a CasADi `SX` even for a NumPy
  input, because `Panda.rot_z` starts from `ca.SX_eye(4)`. Wrapped in a `casadi.Function` it
  agrees with `franka_ik.model.fk_tool` to 3.3 × 10⁻¹⁶.

And a fourth, in the wrist-flip test, is the one that was *repaired* rather than merely
documented: the published sign test vanishes at `q₇ = ±90°` and returns a wrong pose there,
whereas the library decides the wrist by which candidate reaches the pose
(`docs/limitations.md` §7). These four defects share a property worth noting: they are invisible
from the outside. The `nan` hang does not raise, the symbolic `fk` fails with a message about
NumPy, and the wrong wrist is a configuration that reaches a *different* pose.
