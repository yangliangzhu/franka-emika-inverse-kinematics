# AGENTS.md — working on `franka-ik`

## What this repository is

A closed-form inverse kinematics for the Franka Emika Panda, obtained by reducing the arm to an
*equivalent* S-R-S arm and solving that with the method of Shimizu et al. (2008). The
redundancy is parameterised by joint 7.

Two things are unusual about it, and both shape how it should be edited:

* **It carries a scientific claim, not just an implementation.** The published solver has four
  branches; the arm has eight. `docs/branch_analysis.md` is the measurement, `tests/test_branches.py`
  is the check, and `examples/03_branches.py` and `demos/branches.html` are the demonstration.
  Changing the branch structure changes the claim, so it needs a measurement, not an opinion.
* **`original/` is a historical record.** It holds the 2020 code exactly as published, defects
  included, because the library is cross-checked against it and the defects are documented
  rather than erased (`docs/limitations.md`). Never reformat, "fix" or re-derive it.

Read `README.md` first, then `docs/method.md`. `docs/limitations.md` is the list of things that
are known to be imperfect, and is the first place to look when a change seems to break
something. `docs/provenance.md` is what keeps the claims honest: what here is original, what is
not, and which published work got there independently.

## The provenance rule

The method here is **not** novel. He and Liu published the same reduction — joint 7 as the
redundancy parameter, the elbow solved in two variants, eight branches per pose — at ICRA 2022
([IEEE Xplore 9646185](https://ieeexplore.ieee.org/abstract/document/9646185),
[preprint](https://github.com/ffall007/franka_analytical_ik/blob/main/paper_preprint.pdf)). The
first push here predates their preprint by about eight months (`git show --stat 295c6e0`,
2021-02-02), but a repository nobody read is not a scientific claim.

Three consequences for anyone editing this repository:

* **Never write a novelty claim.** Not in the README, not in a docstring, not in a commit
  message. The contribution is the *measurement* — the coverage study, the cross-check against
  `original/`, the IPOPT completeness test — and that is what the documents should claim.
* **Keep `docs/provenance.md` accurate.** Its numbers come from
  `python3 scripts/study_wrist_offset_ik.py`; if a change moves one of them, the document moves
  with it, and the change needs to say why.
* **Do not describe the 2023 published rewrite as the original derivation.** The 2021 prototype
  parameterised the elbow root and pinned it at the call sites
  (`git show 295c6e0:INVERSE_FRANKA.py`, line 97, `solution_theta_4(kesai, -1)` with
  `#/此处有分支` beside it and the alternative commented out on the next line). The 2023 rewrite
  picks the `+` root by writing `(a1 + sqrt(...)) / (2*a2)`. Those are not the same choice, and
  `docs/provenance.md` §1 is careful about exactly how much the commit history does and does not
  establish.

## Environment

| requirement | why |
|---|---|
| Python ≥ 3.10 | `pyproject.toml`'s `requires-python`, forced by the `viz` extra: `roboticstoolbox-python>=1.4` needs 3.10. The code keeps `typing.Optional`/`List` rather than `X | None`, matching the sibling repositories |
| NumPy ≥ 1.21 | the whole library |
| matplotlib ≥ 3.5 | `franka_ik.viz`, the examples and the demo pages only |
| CasADi ≥ 3.6 | **only** to import `original/` and the tests that cross-check against it; tests skip cleanly without it |
| roboticstoolbox, swift-sim, spatialmath, spatialgeometry | the `viz` extra: `examples/07`-`11` and `tests/test_swift_viz.py`. Both skip cleanly without it |
| Node.js | optional, for `scripts/check_demo.js` |

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # .venv + uv.lock, dev group included (pytest, casadi, ruff)
uv run pytest -q           # run anything inside it without activating
uv sync --no-dev           # library only, no pytest/casadi/ruff
uv sync --extra reference  # + CasADi, if you only want the cross-check tests
uv sync --extra viz        # + the 3D viewers (roboticstoolbox, swift-sim, ...)
```

`uv.lock` is committed, so `uv sync` reproduces the exact versions the suite was
last run against. `requirements.txt` is kept for callers who are not using uv:

```bash
pip install -e .                    # numpy + matplotlib
pip install -r requirements.txt     # adds casadi and pytest
```

`import franka_ik` must never pull in CasADi. `franka_ik/model.py`, `geometry.py`, `solver.py`,
`analysis.py` and `numerical.py` are pure NumPy; the presentation layers -- `report.py`,
`viz.py`, `swift_viz.py`, `swift_app.py` -- are not re-exported from `__init__.py`, and only the
last two need the `viz` extra, which they import inside their functions rather than at module
level.

## Running things

Prefix with `uv run` (or activate `.venv` first) if you are using the uv environment:

```bash
uv run pytest                                      # the whole suite (testpaths = tests)
uv run pytest tests/test_branches.py -q            # the cross-check against original/
uv run pytest tests/test_geometry.py::test_q4_collapses_to_the_srs_elbow_law -q

MPLBACKEND=Agg uv run python examples/01_forward_kinematics.py --save-dir /tmp/fk
MPLBACKEND=Agg uv run python examples/02_srs_reduction.py --save-dir /tmp/srs
MPLBACKEND=Agg uv run python examples/03_branches.py --save-dir /tmp/branches

uv run --extra viz python examples/07_swift_branches.py --headless --steps 1
uv run --extra viz python examples/10_swift_singularities.py --headless --steps 1
uv run --extra viz python examples/07_swift_branches.py --headless --model collision --steps 1
uv run --extra viz python examples/07_swift_branches.py --headless --model mesh --steps 1

uv run python -c "from franka_ik.report import build_demo_site; build_demo_site('demos')"
for f in demos/*.html; do node scripts/check_demo.js "$f" || exit 1; done

uv run ruff check .                                # line-length 100, target py310

# the browser side, which has no other witness (needs Playwright, ad hoc)
pip install playwright && playwright install chromium
python3 scripts/browser_drive.py 10 --model skeleton
python3 scripts/browser_drive.py 08 --model skeleton --markers 150   # marker upload dominates
```

Without CasADi installed the suite reports 107 passed and 18 skipped rather than failing, and
without the `viz` extra it reports 107 passed and 1 skipped -- that single entry is the whole
`tests/test_swift_viz.py` module, which a module-level `importorskip` skips as a unit rather than
23 times. The cross-check tests and the Swift tests skip, and everything else runs. CasADi is only ever needed to import the published modules
in `original/`; the visualisation stack is only ever needed by `franka_ik.swift_viz` and the
viewers.

`franka_ik/model.py`, `geometry.py`, `solver.py`, `analysis.py` and `numerical.py` stay free of
both. `swift_viz.py` and `swift_app.py` import `roboticstoolbox`/`swift`/`spatialgeometry` inside
their functions, never at module level, and are not re-exported from `__init__.py`.

### The four rules about the Swift scene

All four are API traps that produce a scene which *looks* like a geometry bug, and all four cost
real time here. `docs/browser_debugging.md` is the full playbook, taken from the sibling S-R-S
study and extended with what was hit here.

* **`env.add_shape(shape)`, one shape at a time, via `swift_viz.add_shapes`.** ``Swift.add``
  dispatches on *one* shape, robot or UI element and **returns ``None`` for anything else -- a
  list included -- without raising**. Handing it ``skeleton.shapes`` produces an empty window
  and no error: every viewer opened blank until this was found.
* **`env.step(dt)` renders; `time.sleep` does not.** A loop that only sleeps leaves the first
  frame on screen, which reads as "the viewer is broken". `swift_app.hold` is the shared loop,
  and `swift_app.interaction_loop` is that plus a slider.
* **`--model mesh` needs `y_up=True`.** The Franka DAE files are Z-up and three.js's
  ColladaLoader re-orients them, so Swift has to undo it; without the flag the whole robot is
  drawn tipped 90 degrees about X, and a tipped robot is still a robot, so nothing else catches
  it.
* **A slider is driven by `slider.value`, never by its callback, and its value has to be written
  back after the element is added.** Swift's slider JavaScript assigns ``value`` before
  ``min``/``max`` while its markup starts at ``0..100``, so a range that does not contain 0 has
  its initial value clamped to the nearer end -- measured, a joint-7 slider built with ``-58.78``
  over ``[-77.92, -50.42]`` arrived as ``-50.42`` -- and the element is reported once, on attach,
  as changed. Build sliders with `swift_app.add_slider`, which writes the value back once the
  browser has the real range; `swift_app.interaction_loop` reads the live value; and
  `tests/test_swift_app.py` pins both halves without a browser. Its ``tolerance`` must be below
  the slider's own ``step``, or a feature as small as `examples/10`'s ``1e-4`` degree failure
  window never triggers a redraw.

One cost and one helper: **`add_shape` blocks until the browser confirms the shape is mounted**,
so a group added that way is one round trip per shape -- measured, 150 markers took 87 s to appear,
and a branch fan rebuilt per slider step never appeared at all. **`swift_viz.add_cloud(env,
shapes)`** is the fix: it adds a group as a single assembly, one message and one wait, and
`env.remove(handle)` takes it off screen in one call. The group can still be moved (its assembly
re-reads each part's pose every frame), but a part's *colour* has to be set before it goes in.
Its `fk` must return `SE3` rather than the 4x4 array `Shape.T` gives back -- an array survives
every Python-side check and then fails in Swift's serialiser on the first frame.
`docs/browser_debugging.md` §2.5 has the measurements.

The root `conftest.py` puts the repository root and `original/` on `sys.path`, which is what
lets the tests import the published modules by their bare names (`import ik_ca`). `tests/conftest.py`
holds the shared fixtures: a seeded `rng`, the in-limit configuration sample, the published
model and entry points, and the session-scoped `coverage_report`.

| test module | tests | what it pins |
|---|---|---|
| `tests/test_model.py` | 14 | the modified-DH table, the limits, `fk_flange`/`fk_tool`/`jacobian` against the published CasADi model and against finite differences |
| `tests/test_geometry.py` | 20 | `STEP1` and `STEP2`: the wrist correction, the elbow quadratic, the reachable shell, the `bias = 0` collapse |
| `tests/test_solver.py` | 20 | the eight branches, the round trip, `wrap_to_limits`, `solve_closest` continuity |
| `tests/test_branches.py` | 14 | **the scientific claim**: the library reproduces every published branch, the published subset is the `+` elbow root, it is incomplete, and the published code's defects |
| `tests/test_analysis.py` | 13 | the coverage and solution-count studies, `reachable_distance_range`, `classify_failure` |
| `tests/test_numerical.py` | 6 | the CasADi model, the IPOPT cross-check, and that `import franka_ik` stays CasADi-free |
| `tests/test_swift_viz.py` | 24 | the Swift layer: the skeleton sits on the model's own joint origins, the URDF check **rejects another arm**, and the sample poses behave as documented. Skips without the `viz` extra |
| `tests/test_swift_app.py` | 17 | the examples' shared plumbing without Swift at all: the interaction loop follows the live slider value, a play hook's write-back sticks, the tolerance can find a `1e-4` degree window, and Swift's empty first radio event is not a choice |

131 tests in total, and 107 of them pass without the `viz` extra. The per-module counts are a
reader's map to the suite rather than a contract, so a new test belongs in the module that matches
its claim and does not need this table edited in the same commit.

### The one rule about `tests/test_branches.py`

**`tests/test_branches.py` must stay green.** It is the only place where the library is compared
against `original/` branch by branch, and it is what makes "this is the same method in another
notation" a fact rather than a hope. Two consequences:

* if it fails, the default assumption is that the change is wrong, not that the test is;
* never make it pass by editing `original/` or by loosening the comparison tolerance. The
  measured agreement is 1.30 × 10⁻¹³ rad over 1200 comparisons, so a tolerance around `1e-9` is
  already generous.

If a genuine change to the *method* makes it fail, that is a finding: write down the
measurement, put it in `docs/branch_analysis.md`, and say so in the pull request.

## Code style

* **English docstrings everywhere.** Every public function says what the maths is, cites the
  derivation, and documents its arguments and return value. Module docstrings carry the
  reasoning that does not fit in a function.
* **Chinese comments are welcome where the domain terminology is clearer in Chinese** — the
  derivation's own vocabulary (`等效`, `和差化积`, `特殊方程`) comes from the PDF, and a comment
  that names the same object as the source document is easier to check than a translation.
  Chinese is not a substitute for an English docstring.
* **Type hints on every public function**, with `from __future__ import annotations`. Return
  types are written as strings (`-> "np.ndarray"`), which is the house style rather than a
  version requirement now that the floor is 3.10.
* **No wildcard imports.** Import names explicitly and list the module's public surface in
  `__all__`. The package re-exports a curated set from `franka_ik/__init__.py`; keep that list
  and `__all__` in sync, and add anything new to `docs/api.md` in the same change.
* **Cite the derivation for anything non-obvious.** Either the PDF's step (`STEP1`…`STEP4`, and
  the appendix *特殊方程的求解* for the quadratic) or the paper's equation number (§3 of the
  paper for the standard-DH rotation, eq. (12) for the S-R-S elbow law, eqs. (15) and (19) for
  the `A`/`B`/`C` matrices). A reader who cannot tell which convention a formula is written in
  will get it wrong; saying so is part of the code, not a nicety.
* **Named constants, not inline literals.** Link lengths, limits and the tolerance defaults live
  in `DH_PARAMETERS`, `PAPER_GEOMETRY`, `LOWER_LIMITS_DEG`/`UPPER_LIMITS_DEG` and the function
  defaults, each with a comment saying where the number comes from.
* **Comment the *why*, especially for deviations.** The existing code flags its own hazards
  (`rot_y`'s sign convention, the `nan` loop in the published `limit_joints`, the one-sided
  joint ranges). A rule that is only obvious to someone who has already been bitten deserves a
  sentence.
* **Never conflate the two DH conventions.** `franka_ik/model.py` is modified (Craig) DH;
  `franka_ik/geometry.py` is the standard DH rotation of the paper. They are reconciled
  numerically and must stay separate.

## Adding a test

1. Put it in the module that matches the claim: model → `test_model.py`, the reduction →
   `test_geometry.py`, the solver → `test_solver.py`, `original/` → `test_branches.py`.
2. Use the fixtures from `tests/conftest.py` rather than sampling again, so different tests
   assert on the same data. Never touch the global NumPy RNG: ask for the `rng` fixture, or
   build your own `np.random.default_rng(seed)`.
3. Sample **inside the real joint limits** (`fk.lower_limits()`, `fk.upper_limits()`), not on
   `[-π, π]`. Joints 4 and 6 are one-sided, so a uniform draw on a symmetric interval produces
   configurations the robot cannot reach.
4. Assert on numbers you measured, with `pytest.approx` or an explicit tolerance; state the
   measured value in the docstring, as the existing tests do. A test whose comment says
   "measured 265/300 at seed 0" is a measurement anyone can re-run.
5. Keep the suite fast — the whole thing runs in seconds, and a test that takes a minute will be
   skipped by whoever is in a hurry. Long studies belong in `scripts/` and are quoted in the
   docs.

```python
"""Template: does the solver return the configuration the pose came from?"""
import numpy as np
import pytest

import franka_ik as fk


def test_recovers_the_generating_configuration(rng):
    lower, upper = fk.lower_limits(), fk.upper_limits()
    for _ in range(25):
        target = rng.uniform(lower, upper)
        pose = fk.fk_flange(target)
        solutions = fk.solve(pose, float(target[6]), within_limits_only=True)
        delta = np.array([s.q for s in solutions]) - target
        delta = np.arctan2(np.sin(delta), np.cos(delta))       # modulo whole turns
        assert np.any(np.all(np.abs(delta) < 1e-6, axis=1))


def test_out_of_reach_returns_nothing():
    pose = np.eye(4)
    pose[:3, 3] = [2.0, 0.0, 0.0]                              # 2 m away: outside the shell
    assert fk.solve(pose, 0.0) == []
    assert fk.classify_failure(pose, 0.0) == "outside_reachable_shell"
```

For anything that needs a hard timeout — the published `limit_joints` hang is the example —
follow `test_branches.py`: run the work in a forked subprocess and fail if it does not return,
so the suite itself can never hang.

## Changing the documentation

The documents are part of the deliverable, not a summary of it.

* **Every published number must come from a command.** If a figure cannot be reproduced by one
  of the studies in `franka_ik.analysis`, a `scripts/` run, or a short snippet in the document,
  it does not belong there. This is why `docs/branch_analysis.md` quotes its measurements and
  says which sample and seed produced them.
* **Say what is measured and what is claimed.** "300 of 300 targets at seed 0" is a measurement;
  "the enumeration is complete" is a claim resting on it. Keep the two apart — the repository
  was built on exactly that distinction.
* **`franka_ik/` is the source of truth for signatures.** `docs/api.md` lists every symbol in
  `__all__`; check a name exists before you write it down.
* **Do not overstate the finding.** The four published branches are all correct; they are
  incomplete. The two roots of the elbow quadratic are a *completion* of the 2020 derivation,
  not a correction of it.
