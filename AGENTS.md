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
something.

## Environment

| requirement | why |
|---|---|
| Python ≥ 3.9 | `pyproject.toml`'s `requires-python`; the code keeps `typing.Optional`/`List` for this reason |
| NumPy ≥ 1.21 | the whole library |
| matplotlib ≥ 3.5 | `franka_ik.viz`, the examples and the demo pages only |
| CasADi ≥ 3.6 | **only** to import `original/` and the tests that cross-check against it; tests skip cleanly without it |
| Node.js | optional, for `scripts/check_demo.js` |

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # .venv + uv.lock, dev group included (pytest, casadi, ruff)
uv run pytest -q           # run anything inside it without activating
uv sync --no-dev           # library only, no pytest/casadi/ruff
uv sync --extra reference  # + CasADi, if you only want the cross-check tests
```

`uv.lock` is committed, so `uv sync` reproduces the exact versions the suite was
last run against. `requirements.txt` is kept for callers who are not using uv:

```bash
pip install -e .                    # numpy + matplotlib
pip install -r requirements.txt     # adds casadi and pytest
```

`import franka_ik` must never pull in CasADi or matplotlib. `franka_ik/model.py`, `geometry.py`,
`solver.py` and `analysis.py` are pure NumPy; the presentation layer is `report.py` and
`viz.py`, and they are not re-exported from `__init__.py`.

## Running things

Prefix with `uv run` (or activate `.venv` first) if you are using the uv environment:

```bash
uv run pytest                                      # the whole suite (testpaths = tests)
uv run pytest tests/test_branches.py -q            # the cross-check against original/
uv run pytest tests/test_geometry.py::test_q4_collapses_to_the_srs_elbow_law -q

MPLBACKEND=Agg uv run python examples/01_forward_kinematics.py --save-dir /tmp/fk
MPLBACKEND=Agg uv run python examples/02_srs_reduction.py --save-dir /tmp/srs
MPLBACKEND=Agg uv run python examples/03_branches.py --save-dir /tmp/branches

uv run python -c "from franka_ik.report import build_demo_site; build_demo_site('demos')"
for f in demos/*.html; do node scripts/check_demo.js "$f" || exit 1; done

uv run ruff check .                                # line-length 100, target py39
```

Without CasADi installed the suite reports 66 passed and 17 skipped rather than failing: the
cross-check tests skip, and everything else runs. CasADi is only ever needed to import the
published modules in `original/`.

The root `conftest.py` puts the repository root and `original/` on `sys.path`, which is what
lets the tests import the published modules by their bare names (`import ik_ca`). `tests/conftest.py`
holds the shared fixtures: a seeded `rng`, the in-limit configuration sample, the published
model and entry points, and the session-scoped `coverage_report`.

| test module | what it pins |
|---|---|
| `tests/test_model.py` | the modified-DH table, the limits, `fk_flange`/`fk_tool`/`jacobian` against the published CasADi model and against finite differences |
| `tests/test_geometry.py` | `STEP1` and `STEP2`: the wrist correction, the elbow quadratic, the reachable shell, the `bias = 0` collapse |
| `tests/test_solver.py` | the eight branches, the round trip, `wrap_to_limits`, `solve_closest` continuity |
| `tests/test_branches.py` | **the scientific claim**: the library reproduces every published branch, the published subset is the `+` elbow root, it is incomplete, and the published code's defects |

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
  types are written as strings (`-> "np.ndarray"`) so the module still imports on 3.9.
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
