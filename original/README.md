# original

The files exactly as they were published on the `ik` branch, kept verbatim.

| file | what it is |
|---|---|
| `panda.py` | the CasADi model: modified-DH forward kinematics, geometric Jacobian, joint limits |
| `ik_ca.py` | the analytical solver, first equivalent-arm-angle branch, both signs of joint 2 |
| `ik_ca2.py` | the same for the second equivalent-arm-angle branch |
| `test.ipynb` | the original verification notebook |

They are kept for three reasons.

**Provenance.** This is the 2020 derivation; `docs/method.md` explains it
equation by equation and this directory is the primary source it explains.

**Cross-checking.** `franka_ik/` is a reimplementation, and a reimplementation is
only trustworthy if it can be compared against something. `tests/test_branches.py`
imports these modules and asserts, branch by branch, that the library reproduces
them -- measured agreement is 1.6e-13 rad over 800 comparisons, so the two
implementations are the same method in two notations.

**The record of what was wrong.** The published solver takes one root of the
elbow quadratic and therefore finds four of the eight branches; `limit_joints`
loops forever on a `nan`. Both are visible in these files, and both are fixed and
tested in `franka_ik/`. See `docs/branch_analysis.md` and `docs/limitations.md`.

## Running the originals

They import each other by bare module name, so put this directory on the path:

```bash
cd original && MPLBACKEND=Agg python3 -c "import ik_ca"
```

or, for the notebook:

```python
import sys; sys.path.insert(0, "original")
```

The originals need CasADi. The library in `franka_ik/` does not.
