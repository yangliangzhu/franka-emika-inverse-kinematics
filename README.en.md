# franka-ik — analytical inverse kinematics for the Franka Emika Panda

[中文](README.md) | **English**

An analytical (closed-form) inverse kinematics for the Franka Emika Panda, derived in 2020 by
reducing the arm to an *equivalent* S-R-S arm and solving that with the closed form of
[Shimizu et al. (2008)](https://doi.org/10.1109/TRO.2008.2003266). The redundancy is
parameterised by **joint 7**: you say which pose to reach and what joint 7 should be, and you
get back every configuration that does it.

The Panda is not an S-R-S arm — its shoulder is offset by 0.0825 m and its wrist by 0.088 m —
so the reduction has real work to do. `docs/method.md` follows it step by step against the
original hand-written derivation (`franka解析反解方法.pdf`, `STEP1`–`STEP4`).

```python
import numpy as np
import franka_ik as fk

q = np.array([0.3, -0.4, 0.2, -1.2, 0.1, 1.0, 0.5])   # any valid Panda configuration
pose = fk.fk_flange(q)

solutions = fk.solve(pose, q7=q[6], within_limits_only=True)
print(solutions[0].describe())
# q4+ phi+ flip    q = [ 17.189, -22.918,  11.459, -68.755,   5.730,  57.296,  28.648] deg   pose error 3.75e-16   in limits
```

## The headline result

The implementation published on this branch believed the arm has **four** kinematic branches
(its readme says so, and it ships two files with two entry points each). It has **eight**. The
elbow equation of `STEP2` is a quadratic in `tan(θ₄/2)`, and the published code evaluates only
its `+` root:

```python
tan_half_q4 = (a1 + ca.sqrt(a1**2 - 4*a0*a2)) / (2*a2)     # original/ik_ca.py
```

Keeping both roots doubles the branch set, and the difference is measurable:

| measurement (300 random in-limit configurations, `seed 0`) | published four branches | this library, eight |
|---|---|---|
| target configuration recovered | 265/300 = **88.3 %** | 300/300 = **100 %** |
| pose residual of what is returned | ≤ 2.0 × 10⁻¹⁴ | ≤ 2.0 × 10⁻¹⁴ |
| distinct in-limit solutions per pose | — | mean **3.15**, range 1–8 |
| runtime, all branches, pure NumPy | — | ≈ 1.3 ms per pose |

The distinction that matters: the four published branches are all *correct*. They are simply not
all of them. A solver that misses a branch still returns four exact poses, so the only test that
detects the problem is "does the solver give me back the configuration I started from" — which
is exactly what `analysis.coverage_study` measures, and what `docs/branch_analysis.md`
documents. The missing 35 of 300 targets are precisely the configurations whose elbow lies on
the second root of the quadratic.

The gap is not only a matter of taste in redundancy resolution: over 150 random configurations
the published subset returns 18.6 % fewer in-limit solutions, and for one pose it returns none
at all — reporting a reachable pose as unreachable — while the full solver finds four.

Keeping both roots is a **completion** of the original derivation, not a correction of it.

## Verified against the published implementation

`original/` holds the 2020 files exactly as published. The library is checked against them:

| check | result |
|---|---|
| NumPy vs CasADi model (`fk_flange`, `fk_tool`) over 300 random configurations | 3.3 × 10⁻¹⁶ |
| library vs published solver, branch by branch over the whole joint-7 range | 1200/1200 matched, worst deviation **1.30 × 10⁻¹³ rad** |
| pose residual of returned solutions | median 4.4 × 10⁻¹⁶, worst 2.0 × 10⁻¹⁴ |
| shoulder-flip symmetry `(q₁,q₂,q₃) → (q₁+π, −q₂, q₃+π)` | 138/138 configurations |
| wrist-flip symmetry `(q₅+π, −q₆, q₇+π)` | 0/138 — the Panda is not a true S-R-S arm |
| independent optimiser (CasADi + IPOPT) finds only branches the solver returns | 56 solutions over 15 poses, **0 counter-examples**, worst distance 0.085° |

Three defects of the published files are recorded rather than quietly fixed, because they are
the kind that waste an afternoon: `limit_joints` never returns when it is handed a `nan`,
`Panda.fk` returns a symbolic object that cannot be converted to numbers, and the joint-limit
windows for joints 4 and 6 are wider than a full turn. A fourth, in the wrist-sign test, is
*repaired* rather than recorded: the published test vanishes at `q₇ = ±90°` and returns a wrong
pose there, while this library decides the wrist by which candidate actually reaches the pose.
See `docs/limitations.md`.

## Install

The environment is managed with [uv](https://docs.astral.sh/uv/). One command
creates the virtual environment, installs the library in editable mode and
brings in the development tools:

```bash
git clone git@github.com:yangliangzhu/franka-emika-inverse-kinematics.git
cd franka-emika-inverse-kinematics
uv sync                 # .venv + uv.lock, dev group included
uv run pytest -q        # 131 tests
uv run ruff check .
```

`uv sync` reads `uv.lock`, so a checkout reproduces the exact versions that were
tested. `uv run <command>` executes inside that environment without activating
it; `source .venv/bin/activate` works too if you prefer.

The library itself needs only **numpy** and **matplotlib**. **CasADi** is an
optional extra, pulled in by the `dev` group:

```bash
uv sync --no-dev                 # library only
uv sync --extra reference        # + CasADi, for the cross-check against original/
```

CasADi is not used by `franka_ik` -- the solver is pure NumPy and runs all eight
branches in about a millisecond -- but it is what `original/` uses, so
`tests/test_branches.py` needs it to compare the library against the published
implementation. Without it the suite skips those tests (107 passed, 18 skipped)
rather than failing.

## Using it

```python
import numpy as np
import franka_ik as fk

pose = fk.fk_flange(np.array([0.3, -0.4, 0.2, -1.2, 0.1, 1.0, 0.5]))   # you bring the 4x4 pose
q7   = 0.5                                # the redundancy parameter, in radians

fk.solve(pose, q7, within_limits_only=True)      # verified, in-limit, deduplicated
fk.branch_solutions(pose, q7)                    # all branches, unfiltered
fk.solve_closest(pose, q7, reference=previous)   # the branch nearest a reference, for tracking
```

`solve` checks every candidate against the forward kinematics and drops the ones whose residual
exceeds `tolerance` (`1e-9` by default), so what comes back is correct — but a solution is not
guaranteed to exist, and the answer depends on the `q₇` you asked for. Three things to know
before putting it in a controller:

* **`q₇` must be tracked, not fixed.** At a constant `q₇` only about a third of reachable poses
  are solvable; feeding the previous command's `q₇` solves 200/200 in the same sample.
* **The solver works on the flange frame**, not the tool frame. Convert with
  `T_flange = T_tool @ rot_z(+π/4) @ trans_z(-0.1034)`; forgetting the 0.1034 m term gives
  silently wrong answers.
* **`q₇ = ±90°` used to be the weak point.** The published wrist-sign test is scaled by
  `cos²(q₇)`, vanishes there, and returns a wrong pose (16 wrong-pose calls out of 7896 across
  the whole joint-7 range, all at `±90°`). This library does not have that failure: it evaluates
  both wrist candidates and keeps the one that reaches the pose, which recovers 100/100 targets
  at exactly `±90°`.

`docs/limitations.md` has the measurements behind all three.

## Seeing it move

Three of the findings above are easier to believe when the arm is on screen: the eight branches, the
reachable shell, and joint 2 at zero. `examples/07`-`11` are interactive Swift viewers for them,
driven by the same closed form that `solve` uses:

```bash
uv sync --extra viz
python3 examples/07_swift_branches.py --pose second_root   # all solutions for one pose, by elbow root
python3 examples/08_swift_workspace.py                      # the reachable shell, with --scan
python3 examples/10_swift_singularities.py                  # q2 = 0, where nothing is both exact and in range
python3 examples/11_swift_tracking.py                       # a straight line followed with solve_closest
```

The Swift windows are **interactive**, and every control is a measurement.  `07` has a joint-7
slider that re-solves and rebuilds the whole fan, plus a play button and a camera radio; `08` has
the **elbow** (joint 4, the only single joint that changes `‖x_sw‖`) sweeping the wrist from
0.20685 m to 0.71935 m -- the closed-form outer radius itself; `09` sweeps joint 7 with a
one-per-root / every-solution radio; `10` has a joint-2 offset slider (±0.01°, step 1e-5) that
brackets the 1e-4° failure window, with the four measured offsets on a preset radio; `11` scrubs
and plays a whole Cartesian line at a speed in steps per second.  Every one of them carries a
readout saying what is on screen. `demos/branches.html` does the same thing without Swift: a joint-7
slider and play button over a sweep that `franka_ik.report` pre-computes and embeds, with the
arms coloured **blue for the elbow root the published code keeps and orange for the one it
drops**, so scrubbing to the generating configuration shows it sitting on the orange side.

They need a browser (or `--headless`, which runs to completion and prints every number, and draws
nothing — if you see only text, that is the flag, not a missing URDF).

`--model` chooses what the arm looks like, and all three work offline:

* **`--model mesh` (default)** draws the Panda's **own visual meshes** — the real robot. The
  description is vendored under `third_party/` (Apache-2.0, ~11 MB, 8 links) and expanded from
  xacro on first use; the meshes are placed by `franka_ik.model.forward_kinematics`, so they
  cannot drift from the solver either;
* `--model collision` draws the bundled `rtb-data` description's collision geometry: 30 cylinders
  and spheres, a decent likeness that needs no files beyond the installed package;
* `--model skeleton` is a stick figure, and the only mode that can be recoloured per elbow root —
  which is how examples 07 and 09 tell the two halves of the branch set apart.

```bash
uv run --extra viz python examples/07_swift_branches.py --pose second_root --browser auto
uv run --extra viz python examples/07_swift_branches.py --pose second_root --model collision
```

`FRANKA_IK_URDF` overrides the description with a URDF of your own, and it is **verified against
`fk_tool` before it is drawn** — `franka_description` ships the Panda and the FR3, whose wrists
differ by 57 mm, and driving the wrong one with Panda joint vectors would draw a plausible, wrong
arm. `third_party/README.md` has the provenance; `examples/README.md` lists all eleven examples.

## Repository layout

| path | what it is |
|---|---|
| `franka_ik/` | the library: `model` (modified-DH FK, Jacobian), `geometry` (the S-R-S reduction), `solver` (eight-branch IK), `analysis` (studies of the solution set), plus `viz` and `report` for the figures and the demo pages |
| `original/` | the 2020 implementation, kept verbatim: `panda.py`, `ik_ca.py`, `ik_ca2.py`, `test.ipynb` |
| `docs/` | `method.md`, `branch_analysis.md`, `limitations.md`, `api.md`, and the author's `original_notes_zh.md` |
| `franka解析反解方法.pdf` | the hand-written derivation, `STEP1`–`STEP4` |
| `tests/` | the test suite, including the branch-by-branch cross-check against `original/` |
| `examples/` | eleven walkthroughs: `01`-`06` are matplotlib studies (the model, the reduction, the eight branches, the joint limits, the coverage study, tracking), `07`-`11` are the interactive Swift viewers |
| `demos/` | self-contained interactive HTML pages — open `demos/index.html` in a browser, no server or build step |
| `third_party/` | the vendored Panda description: an Apache-2.0 subset of `franka_description` (about 11 MB, 8 visual meshes and 30 collision primitives) that `--model mesh` draws; provenance in `third_party/README.md` |
| `scripts/` | the study scripts (`study_branches.py`, `study_wrist_offset_ik.py`), the demo checker `check_demo.js`, and `browser_drive.py`, which drives a viewer in a real browser with Playwright |

## Documentation

| document | contents |
|---|---|
| [docs/method.md](docs/method.md) | the derivation, `STEP1`–`STEP4`, equation by equation, with the S-R-S paper's equation numbers |
| [docs/branch_analysis.md](docs/branch_analysis.md) | four branches or eight: the measurement, and why the obvious test cannot see the difference |
| [docs/browser_debugging.md](docs/browser_debugging.md) | nothing on screen, nothing moving: the Swift API traps this repository walked into, and how to see the page with Playwright |
| [docs/limitations.md](docs/limitations.md) | joint limits, `q₇ = ±90°`, symmetries that do not exist, the published code's defects |
| [docs/provenance.md](docs/provenance.md) | what is original here and what is not: the dates, the closest published work, and what it says about the same second root |
| [docs/api.md](docs/api.md) | every exported symbol, with signature and example |
| [docs/original_notes_zh.md](docs/original_notes_zh.md) | the author's own notes from 2020 |
| [AGENTS.md](AGENTS.md) | contributor guide: environment, tests, style |

## Provenance

An original, self-derived method — first pushed in **February 2021**, when the author worked with
a Franka arm and could not find a published analytical inverse kinematics for it. It reduces the
arm to an *equivalent* S-R-S family by rotating the wrist offset into a lengthened link, and
solves that with the closed form for KUKA-style S-R-S arms, parameterised by joint 7.

It is **not novel**, and the repository should not pretend otherwise. He and Liu published the
same reduction, with the same redundancy parameter and the same eight branches, at ICRA 2022
([IEEE Xplore 9646185](https://ieeexplore.ieee.org/abstract/document/9646185)); their preprint
postdates the first push here by about eight months, but a repository nobody reads is not a
scientific claim. What this repository does that the published work does not is *measure* its own
completeness — against the 2020/2023 files, against an independent IPOPT enumeration, and against
the branch count — and record where the method is wrong.

Three findings from that measurement are worth stating plainly, and all three are in
[docs/provenance.md](docs/provenance.md) with their numbers:

* the 2021 prototype exposed the second root of the elbow quadratic as a parameter
  (`solution_theta_4(lamda, choice)`) and then pinned it at the single call site, leaving the
  alternative commented out on the next line — so a root was dropped on purpose, not overlooked,
  and the 2023 rewrite that `original/` preserves makes a different choice again;
* the joint limits are **not** the reason, and the published `limit_joints` never checks one: it
  wraps into hand-written windows that are wider than the limits `Panda` declares;
* every one of the 35 configurations that sit on the discarded root has a valid in-limit solution
  there, so the second root is a genuine engineering trade, not a dead branch.

The 2020/2023 code is preserved unmodified in `original/`; `docs/original_notes_zh.md` is the
author's own summary, and the derivation itself is the hand-written PDF at the repository root.

If you use the method, cite the published derivations it stands on:

> M. Shimizu, H. Kakuya, W.-K. Yoon, K. Kitagaki, K. Kosuge, "Analytical Inverse Kinematic
> Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to
> Redundancy Resolution", *IEEE Transactions on Robotics*, 24(5):1131–1142, 2008.
> [doi:10.1109/TRO.2008.2003266](https://doi.org/10.1109/TRO.2008.2003266)

> Y. He, S. Liu, "Analytical Inverse Kinematics for Franka Emika Panda — a Geometrical Solver for
> 7-DOF Manipulators with Unconventional Design", *IEEE International Conference on Robotics and
> Automation (ICRA)*, 2022.

## License

MIT — see [LICENSE](LICENSE).
