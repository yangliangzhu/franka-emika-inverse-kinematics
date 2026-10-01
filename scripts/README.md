# scripts

Runnable studies and checks that are too slow, or need another runtime, to live in
the test suite.

| script | runtime | what it does |
|---|---|---|
| `study_branches.py` | Python 3, NumPy | runs the coverage study and the solution-count study of `franka_ik.analysis` over the same seeded sample of poses, prints them with the numbers labelled, and can dump them as JSON. This is where the numbers quoted in `docs/branch_analysis.md` come from |
| `check_demo.js` | Node.js | headless smoke test for the generated pages under `demos/`. It loads a page, extracts the embedded `window.FRANKA_IK_DATA` payload and the renderer, runs the renderer against a minimal DOM/canvas stub, and checks that nothing threw, that every arm polyline is a full chain of finite 3D points, and that the 2D chart stayed inside its canvas |

## `study_branches.py`

```bash
cd /home/yang/workspace/research/franka-emika-inverse-kinematics
python3 scripts/study_branches.py                                     # 300 poses, seed 0
python3 scripts/study_branches.py --samples 300 --seed 0 --json out.json
python3 scripts/study_branches.py --samples 120 --quiet                # exit status only
```

| flag | default | meaning |
|---|---|---|
| `--samples` | `300` | number of random poses sampled uniformly inside the real joint limits |
| `--seed` | `0` | RNG seed; the same seed and sample count always print the same numbers |
| `--json PATH` | – | write the numbers (rates, histograms, totals, reachable shell) to `PATH` as JSON, rounded to 6 decimals so the file is stable |
| `--quiet` | off | suppress the report; combine with `--json` for machine-readable output only |
| `--log-level` | `WARNING` | logging level, e.g. `INFO` to see where the JSON was written |

The output is deterministic by construction. Anything that varies between
machines — timings in particular — is deliberately *not* part of it, so the same
command can be quoted in the documentation.

Sections printed: (1) coverage, i.e. how often the solver returns the very
configuration the pose was generated from, for the published four-branch subset
and for the full eight; (2) the distribution of the number of distinct in-limit
solutions per pose, full versus published; (3) the reachable shell of `‖x_sw‖`,
with and without the shoulder bias.

## `check_demo.js`

```bash
node scripts/check_demo.js demos/branches.html
for f in demos/*.html; do node scripts/check_demo.js "$f" || exit 1; done
```

Takes one path to a generated HTML page and exits `0` when it rendered without
throwing and stayed in bounds, `1` otherwise. It needs Node.js rather than Python,
which is why it is not in the pytest suite; the Python side is covered by
`pytest`, the pages by this script.
