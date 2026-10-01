# examples

Six numbered, self-contained examples that walk the derivation in the order it is
derived. Each one runs with no arguments, prints every number it demonstrates,
and writes a figure when `--save-dir` is given.

```bash
cd /home/yang/workspace/research/franka-emika-inverse-kinematics
MPLBACKEND=Agg python3 examples/01_forward_kinematics.py --save-dir /tmp/fk
```

Each example inserts the repository root into `sys.path` itself, so it also runs
from any other working directory without installing the package. With
`--save-dir DIR` the figure is saved with the `Agg` backend and nothing is shown;
without it, `pyplot` is imported lazily and the figure is displayed. Every one
takes `--save-dir`, `--seed` and `--log-level`, and the diagnostics go through
`logging` while the deliberate tables go to stdout.

| example | derivation step | what to look for |
|---|---|---|
| `01_forward_kinematics.py` | STEP 0 — the modified-DH model | the nine DH rows (seven joints plus the flange row `d = 0.107` and the tool row `d = 0.1034`); `fk_tool` differs from `fk_flange` by exactly `0.1034 m` along the flange `z` and the factory `RotZ(-45°)`; the two one-sided joint ranges; manipulability is *invariant* to joint 1 (the base rotation is a symmetry) and peaks at `0.0929` on joint 4 at `-112.7°` |
| `02_srs_reduction.py` | STEP 1 and STEP 2 — the equivalent S-R-S arm | `beta = 39.4349°` and the effective wrist length `0.1385388 m` (not `d_wt = 0.107`); on the **corrected** frame the elbow quadratic reproduces the true joint 4 to `1e-14` deg, on the raw frame it is off by `37.6°`; both roots of `a2 x² − a1 x + a0`, with `x = tan(θ4/2)`; with `bias = 0` the coefficient `a1` vanishes and the quadratic collapses onto `cos θ4 = (d² − d_se² − d_ew²)/(2 d_se d_ew)` to `3.3e-16` |
| `03_branches.py` | the eight branches — the headline finding | one pose with **eight distinct in-limit solutions**, printed through `IkSolution.describe()`; the published four-branch subset (`q4_root = +1`) returns four, all correct, and **misses the configuration the pose came from**, which a `q4−` branch recovers; `solve_closest` then picks the branch nearest a reference instead of the first one |
| `04_joint_limits.py` | joint limits and the wrap | joint 4 `[-175, -5]°` and joint 6 `[0, 214]°` are one-sided, so `+90°` on joint 4 becomes `-270°` — still outside; `ValueError` on a `nan` (the published `limit_joints` instead spins on `nan` forever, which `--check-original` reproduces in a subprocess with a timeout) |
| `05_coverage_study.py` | the solution set — coverage and count | `coverage_study(300, seed=0)`: the published subset recovers the target in **265/300 (88.3 %)**, the full eight branches in **300/300**; histogram of solutions per pose (`1: 29, 2: 108, 3: 37, 4: 92, 5: 7, 6: 9, 7: 12, 8: 6`); the reachable shell of `‖x_sw‖`, `0.0662–0.7194 m` with the bias and `0.068–0.700 m` without it |
| `06_tracking.py` | choosing a branch along a path | a 61-step Cartesian path with joint 7 prescribed, followed twice: the naive "first solution" rule jumps **179.4°** in a single step when the branch changes, `solve_closest` moves at most **1.0°** per step — and both are correct at every step (residual ≈ `1e-14`) |

The figures are

| example | file written into `--save-dir` | content |
|---|---|---|
| 01 | `01_forward_kinematics.png` | the arm at four configurations, drawn in a `projection="3d"` axes through `franka_ik.viz.plot_arm_3d`, and manipulability against one joint |
| 02 | `02_srs_reduction.png` | the elbow quadratic for four wrist distances with **both** roots marked, and the `bias = 0` collapse |
| 03 | `03_branches.png` | the eight branch configurations plus the target as a heat map of joint angles |
| 04 | `04_joint_limits.png` | the allowed ranges of joints 1, 4 and 6 with the input and wrapped angles marked |
| 05 | `05_coverage_study.png` | the stacked solution-count histogram, full versus published, and the recovery rates |
| 06 | `06_tracking.png` | joint trajectories of both control rules, the step sizes and the pose residuals |

One note on the environment, and one on the numbers. Drawing in 3D needs
`franka_ik.viz.import_mplot3d()` on a machine with two matplotlib installations:
a distribution `mpl_toolkits` under `/usr/lib/python3/dist-packages` shadows the
pip one and `import mpl_toolkits.mplot3d` fails with `cannot import name
'docstring' from 'matplotlib'`. That helper loads `mpl_toolkits.mplot3d` from the
directory belonging to the installed matplotlib, so example 01 calls it before
opening a `projection="3d"` axes and gets the ordinary 3D projection. And the
numbers above are the defaults at the seeds each example ships; every one of them
is printed by the example itself, so a change in the library shows up as a changed
number rather than as a silent drift.
