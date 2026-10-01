# franka-ik — analytical inverse kinematics for the Franka Emika Panda

**English** | [中文](README_zh-CN.md)

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
git clone git@gitee.com:yangliangzhu_rob/franka-emika-inverse-kinematics.git
cd franka-emika-inverse-kinematics
uv sync                 # .venv + uv.lock, dev group included
uv run pytest -q        # 83 tests
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
implementation. Without it the suite skips those tests (66 passed, 17 skipped)
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

## Repository layout

| path | what it is |
|---|---|
| `franka_ik/` | the library: `model` (modified-DH FK, Jacobian), `geometry` (the S-R-S reduction), `solver` (eight-branch IK), `analysis` (studies of the solution set), plus `viz` and `report` for the figures and the demo pages |
| `original/` | the 2020 implementation, kept verbatim: `panda.py`, `ik_ca.py`, `ik_ca2.py`, `test.ipynb` |
| `docs/` | `method.md`, `branch_analysis.md`, `limitations.md`, `api.md`, and the author's `original_notes_zh.md` |
| `franka解析反解方法.pdf` | the hand-written derivation, `STEP1`–`STEP4` |
| `tests/` | the test suite, including the branch-by-branch cross-check against `original/` |
| `examples/` | three runnable walkthroughs: the model, the reduction, the eight branches |
| `demos/` | self-contained interactive HTML pages — open `demos/index.html` in a browser, no server or build step |
| `scripts/` | study scripts and the demo checker |

## Documentation

| document | contents |
|---|---|
| [docs/method.md](docs/method.md) | the derivation, `STEP1`–`STEP4`, equation by equation, with the S-R-S paper's equation numbers |
| [docs/branch_analysis.md](docs/branch_analysis.md) | four branches or eight: the measurement, and why the obvious test cannot see the difference |
| [docs/limitations.md](docs/limitations.md) | joint limits, `q₇ = ±90°`, symmetries that do not exist, the published code's defects |
| [docs/api.md](docs/api.md) | every exported symbol, with signature and example |
| [docs/original_notes_zh.md](docs/original_notes_zh.md) | the author's own notes from 2020 |
| [AGENTS.md](AGENTS.md) | contributor guide: environment, tests, style |

## Provenance

This is an original, self-derived method, written at the end of 2020 when the author first
worked with a Franka arm and could not find a published analytical inverse kinematics for it.
It was obtained by applying a change-of-variables idea to a closed form for KUKA-style S-R-S
arms — it is not claimed to be the optimal method, and several better ones have appeared since.
The 2020 code is preserved unmodified in `original/`; `docs/original_notes_zh.md` is the
author's own summary, and the derivation itself is the hand-written PDF at the repository root.

If you use it, cite the paper the closed form comes from:

> M. Shimizu, H. Kakuya, W.-K. Yoon, K. Kitagaki, K. Kosuge, "Analytical Inverse Kinematic
> Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to
> Redundancy Resolution", *IEEE Transactions on Robotics*, 24(5):1131–1142, 2008.
> [doi:10.1109/TRO.2008.2003266](https://doi.org/10.1109/TRO.2008.2003266)

## License

MIT — see [LICENSE](LICENSE).
