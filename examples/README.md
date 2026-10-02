# examples

Eleven numbered, self-contained examples. `01`-`06` walk the derivation in the
order it is derived, in matplotlib; `07`-`11` are the interactive 3D viewers that
need the optional `viz` extra. Each one runs with no arguments, prints every number
it demonstrates, and writes a figure when `--save-dir` is given.

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

## The Swift viewers, `07`-`11`

These need `uv sync --extra viz` (or `pip install -e ".[viz]"`), which brings in
`roboticstoolbox`, `swift-sim`, `spatialmath` and `spatialgeometry`. They open a
browser window and animate, so they are not part of the pytest suite; `--headless`
runs one to completion without a browser and prints everything it measured, which is
what CI (or a terminal over ssh) wants. They share their command-line flags and
their sample poses through `franka_ik.swift_app`.

| example | what it shows |
|---|---|
| `07_swift_branches.py` | every in-limit solution for one pose at once, coloured by elbow root, with the generating configuration solid. A **joint-7 slider** re-solves and rebuilds the fan as you drag it, a **play** button sweeps it, and the readout names the selected solution. For `--pose second_root` the published four-branch subset cannot return it, and the readout says so |
| `08_swift_workspace.py` | the reachable shell as two point clouds (the outer boundary and the hole around the shoulder), the arm inside it, and `--scan`, which walks the shell in bands and reports how often the published subset recovers the target per band. The slider is the **elbow**, which is a measurement rather than a choice: it is the only single joint that moves `‖x_sw‖` at all, from 0.20685 m folded to 0.71935 m fully stretched — the outer radius itself, at joint 4 = −26.7573°, an angle that depends on the arm and not on the pose. An `inner` checkbox adds the second surface to the scene |
| `09_swift_elbow_roots.py` | one configuration per elbow root side by side, or `--show all` for the whole set, or `--sweep` to walk joint 7 across its in-limit window at a fixed pose. A **joint-7 slider** re-solves and a **roots/all radio** switches the view, with `θ₄` printed for every configuration drawn — the pose never moves, and the number of solutions still changes |
| `10_swift_singularities.py` | `q₂ = 0`, where the solver returns nothing: all eight branches are listed with their residuals and the joint that disqualifies each. The four printed offsets bracket the failure window, which is `1e-4` degrees wide, and the **offset slider** (`±0.01` degrees, step `1e-5`, so one arrow-key press is a tenth of the window) is fine enough to find it by hand; a **preset radio** jumps to each measured offset |
| `11_swift_tracking.py` | a straight Cartesian line followed with `solve_closest`, with the manipulability ellipsoid as the readout. The plan is computed once and then **scrubbed**: a step slider, play/pause, and a speed slider in trajectory steps per second. The default line sweeps joint 2 through zero and prints how close it came and that it survived |

```bash
uv sync --extra viz
python3 examples/07_swift_branches.py --pose second_root
python3 examples/08_swift_workspace.py                    # drag the elbow, watch |x_sw|
python3 examples/10_swift_singularities.py                # drag the offset out to 1e-4
python3 examples/10_swift_singularities.py --headless
python3 examples/11_swift_tracking.py --headless --plan-steps 200 --steps 200
```

`--steps` is how many frames to run in every example; `11` also takes `--plan-steps`, which is
how finely the Cartesian line is sampled. The interactive paths need a browser, and `--headless`
prints everything while drawing nothing. To check one of them without a person watching,
`scripts/browser_drive.py` runs an example with Playwright as its client and reports on what the
readout says after each interaction:

```bash
pip install playwright && playwright install chromium   # once, ad hoc: not in the extras
python3 scripts/browser_drive.py 10 --model skeleton
```

Three arms are available, and `--model` picks between them. All are offline.

| `--model` | what it draws | where the geometry comes from | can be recoloured |
|---|---|---|---|
| `mesh` (default) | the Panda's own visual meshes — the real robot | `third_party/franka_description`, expanded from xacro on first use | no |
| `collision` | 30 cylinders and spheres, a decent likeness | the `rtb-data` description inside `roboticstoolbox` | no |
| `skeleton` | a stick figure | `franka_ik.model.joint_frames` | yes, per elbow root |

The meshes and the skeleton are both placed by `franka_ik.model.forward_kinematics`,
so neither can drift from the solver; the collision capsules come from
`arm.fkine_all`, the same kinematics the collision checker uses, on a model
`load_arm` has already verified against `fk_tool`.

```bash
python3 examples/07_swift_branches.py --pose second_root --browser auto
python3 examples/07_swift_branches.py --pose second_root --model collision
python3 examples/07_swift_branches.py --pose second_root --model skeleton   # recoloured per branch
```

`07` and `09` are the two examples where the branch colouring matters, so they are
the two worth running with `--model skeleton` as well. A URDF, if one is found, is used for the *robot* and
is verified against `fk_tool` before it is drawn: `franka_description` ships the
Panda and the FR3, whose wrists differ by 57 mm, and driving the wrong one with
Panda joint vectors would render a plausible, wrong arm. Point `FRANKA_IK_URDF` at
a Panda URDF to draw real bodies instead of the capsule skeleton.

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
