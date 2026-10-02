# scripts

Runnable studies and checks that are too slow, or need another runtime, to live in
the test suite.

| script | runtime | what it does |
|---|---|---|
| `study_branches.py` | Python 3, NumPy | runs the coverage study and the solution-count study of `franka_ik.analysis` over the same seeded sample of poses, prints them with the numbers labelled, and can dump them as JSON. This is where the numbers quoted in `docs/branch_analysis.md` come from |
| `study_wrist_offset_ik.py` | Python 3, NumPy (CasADi only with `--optimiser`) | measures this method against the 2020/2023 code it grew out of: which elbow root that code keeps, whether the joint limits explain the root it drops, where joint 4 lands on that root, and — with `--optimiser` — whether an independent IPOPT enumeration finds anything the eight branches do not. This is where the numbers quoted in `docs/provenance.md` come from |
| `check_demo.js` | Node.js | headless smoke test for the generated pages under `demos/`. It loads a page, extracts the embedded `window.FRANKA_IK_DATA` payload and the renderer, runs the renderer against a minimal DOM/canvas stub, and checks that nothing threw, that every arm polyline is a full chain of finite 3D points, and that the 2D chart stayed inside its canvas |
| `browser_drive.py` | Python 3 + Playwright | drives one of the Swift viewers (`examples/07`-`11`) in a real Chromium, performs that example's own interactions — slider drags, radios, buttons — and asserts on what its readout says. This is the only check of the browser side: `pytest` covers the maths and the geometry, `--headless` covers the Python side of the viewers, and neither can see whether a drag redraws the arm |

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

## `study_wrist_offset_ik.py`

```bash
python3 scripts/study_wrist_offset_ik.py                        # the three fast studies
python3 scripts/study_wrist_offset_ik.py --show-limit-joints    # + what limit_joints enforces
python3 scripts/study_wrist_offset_ik.py --optimiser --optimiser-poses 10
python3 scripts/study_wrist_offset_ik.py --quiet --json /tmp/provenance.json
```

| flag | default | meaning |
|---|---|---|
| `--samples` | `300` | in-limit configurations drawn for the three fast studies |
| `--seed` | `0` | RNG seed; the same seed always prints the same numbers |
| `--json PATH` | – | write every number to `PATH` as JSON |
| `--quiet` | off | suppress the report; combine with `--json` for output only |
| `--show-limit-joints` | off | also run the published `limit_joints` on four probes and report whether the results are inside the Panda limits `Panda` declares. Needs CasADi and `original/` |
| `--optimiser` | off | also run the independent CasADi + IPOPT cross-check. Slow: about 4 s per pose at the default 400 starts |
| `--optimiser-poses` | `10` | poses for `--optimiser` |
| `--optimiser-starts` | `400` | IPOPT starting points per pose |

Sections printed: (1) which root of the elbow quadratic the generating
configuration sits on, and how often the published subset recovers it; (2)
whether the joint limits explain the root the published subset drops — the
answer is no, and the section prints the counts that show it; (3) where joint 4
lands on that root, split by in-limits and out-of-limits.

Two caveats worth keeping in mind when reading the numbers:

* the sample is the same seeded one `study_branches.py` uses, so the 265/300 and
  35/300 here are the same 265 and 35 as in `docs/branch_analysis.md`, not an
  independent confirmation of them;
* with `--optimiser`, a non-zero `stopping_short_of_a_branch` is not a missing
  branch. The section prints the pose gap for those cases: at a kinematic
  singularity a configuration can move by a fraction of a degree and leave the
  pose where it was, so IPOPT stops a fraction of a degree from the branch it is
  standing on. A counter-example would be a configuration that *reaches the pose*
  and is far from every branch; the counters are kept apart for that reason.

## `browser_drive.py`

```bash
pip install playwright && playwright install chromium     # once, ad hoc: not in the extras
python3 scripts/browser_drive.py 10 --model skeleton      # one example, checked
python3 scripts/browser_drive.py 08 --model skeleton --markers 60
```

| flag | default | meaning |
|---|---|---|
| `example` | – | which example to drive: `07`, `08`, `09`, `10` or `11` |
| `--timeout` | `120` | seconds to wait for the side panel to fill |
| `--shots` | `/tmp/franka-ik-browser` | where the screenshots are written, which is what to look at when a check fails |
| *(remainder)* | – | arguments for the example itself; put this script's own flags before the example number |

Exits `0` when every check passed, `1` otherwise, and prints one `PASS`/`FAIL` line
per check together with the readout text it read. Each example's scenario is the one
in `docs/browser_debugging.md` section 4: it dispatches DOM events rather than using
Playwright's actionability-checked `click()`, because the page re-renders every 50 ms
and a stalled click can outlive Swift's 15 s reply timeout. `--model skeleton` is the
quick mode; `--model mesh` exercises the vendored Panda meshes. Example `08` uploads
its marker cloud one shape at a time unless `--markers` is kept small.

## `check_demo.js`

```bash
node scripts/check_demo.js demos/branches.html
for f in demos/*.html; do node scripts/check_demo.js "$f" || exit 1; done
```

Takes one path to a generated HTML page and exits `0` when it rendered without
throwing and stayed in bounds, `1` otherwise. It needs Node.js rather than Python,
which is why it is not in the pytest suite; the Python side is covered by
`pytest`, the pages by this script.
