# Provenance: which parts of this are new, and which are not

This repository says two things about itself that need separating, because they have very
different standing.

* **The derivation is not new.** A closed-form inverse kinematics for the Franka Panda, with
  joint 7 as the redundancy parameter and eight elbow/wrist branches, is not this repository's to
  claim; no priority is claimed for it. §1 has the dates.
* **The measurement is this repository's own.** It *checks* that the branch enumeration is
  complete, against an independent optimiser, with the numbers recorded — and it caught, in this
  repository's own predecessor, a branch that the 2023 rewrite had dropped.

Both are worth writing down, and writing down only the first would be as misleading as writing
down only the second. Everything numeric below is printed by

```bash
python3 scripts/study_wrist_offset_ik.py --show-limit-joints
python3 scripts/study_wrist_offset_ik.py --optimiser --optimiser-poses 10
```

## 1. The dates, from the commit history

| date | what happened | how to check |
|---|---|---|
| 2021-02-02 | first public push: the S-R-S reduction and the offset-elbow quadratic, in NumPy (`SRT_inverse_FRANKA.py` and five siblings) | `git show --stat 295c6e0` |
| 2021-03-03 | last commit of that period, "consider sign of joint 2" | `git log --format="%h %ad %s" --date=short origin/inverse` |
| 2023-02-10 | the CasADi rewrite: `ik_ca.py`, `ik_ca2.py`, `panda.py`, `test.ipynb` — the files `original/` preserves, and the last substantive change to the method | `git show 5c2d028` |
| 2026-10 | this library, its tests and its measurements | `git log --oneline` |

The 2021 prototype is the interesting one for priority, because it is the earliest public
evidence of the method, and because its file header states the problem it is solving:

```python
#/ SRT构型指 SRS + 3,4轴一正一负的a参数偏移
#!  此构型分支数待定, 至少为2
```

("SRT means S-R-S with opposite-sign `a` offsets on axes 3 and 4. The branch count of this
configuration is undetermined, at least 2.") The author knew in February 2021 that the branch
count was the open question. That is the same question this repository eventually answered —
eight — five years later.

**The 2021 code parameterised the elbow root and then pinned it.** Its `solution_theta_4(phi,
choice)` takes the sign of the square-root term as an argument (line 33 of
`SRT_inverse_FRANKA.py`, `git show 295c6e0:SRT_inverse_FRANKA.py`), and every call site passes a
constant. In `SRT_inverse_FRANKA.py` that is line 82, bare; the sibling `INVERSE_FRANKA.py`
writes the same call and then leaves the alternative commented out underneath it:

```python
theta_4 = solution_theta_4(kesai, -1)                     #/此处有分支     ("there is a branch here")
# theta_4 = solution_theta_4(kesai, -1) + pi              #/此处又有分支   ("and another one here")
```

That is a deliberate hook with a hard-wired value, not an oversight in the algebra: the author
knew in February 2021 that the sign was a branch, wrote it down, and used one side. The exact
numeric mapping from that prototype's convention to the 2023 rewrite's `q4_root` label is **not**
established here — the prototype's `delta_d` and pose conventions differ, and it does not run
against `franka_ik` as it stands, so this document does not claim one. What is established is the
shape: one root was computed, the other was available and unused.

## 2. What the published code does with the second root

Two separate questions, and they have different answers.

**Is the second root excluded because it violates the Panda's joint limits?** No — and the
published code never checks a limit at all. `limit_joints` wraps angles into hand-written windows
(`[-181, 181]` for joints 1, 2, 3, 5, 7; `[-271, 91]` for joint 4; `[-74, 288]` for joint 6)
while `Panda.upper_bounds`/`lower_bounds` in the same repository say `[-165, 165]`, `[-175, -5]`
and `[0, 214]`. The two do not agree, and the windows are wider:

```
  Panda model limits, lower: [-165.0, -100.0, -165.0, -175.0, -165.0, 0.0, -165.0]
  Panda model limits, upper: [165.0, 100.0, 165.0, -5.0, 165.0, 214.0, 165.0]
     joint 4, 0 deg ->     0.00 deg   inside those limits: False
   joint 6, -30 deg ->   -30.00 deg   inside those limits: False
   joint 7, +170 deg ->   170.00 deg   inside those limits: False
```

So `limit_joints` cannot have filtered anything by limit. What it is for is the *testing* path:
the 2020 notebook draws a random configuration, calls `limit_joints` to bring it into range, and
uses that as the target. It normalises a target; it never validates a solution.

**Does the second root produce anything useful?** Yes, measured. Over 300 random in-limit
configurations (`seed 0`):

| measurement | value |
|---|---|
| targets sitting on the `−` root (unreachable for the published four-branch subset) | 35 of 300 |
| of those, poses where the `−` root has at least one **in-limit** solution | **35** |
| of those, poses where it has none | **0** |
| of those, poses the `+` root also reaches | 33 |
| of those, reachable *only* through the `−` root | 2 |
| in-limit solutions at the target joint 7 | `+` root 792, `−` root 153 |

Every one of the 35 has a valid, in-limit configuration on the second root. Nothing about the
Franka's limits makes that half unusable.

## 3. Where joint 4 lands on the second root

The second root's solutions are not scattered: the ones inside the joint limits occupy one narrow
band of joint 4, and everything outside the limits lies well away from it. Measured over 300 random
in-limit configurations (`seed 0`), on the `−` root:

| joint 4 on the `−` root | value |
|---|---|
| in-limit solutions | 153, range **[−26.00°, −5.86°]**, median −16.63° |
| out-of-limit solutions on the same root | 275, range [−267.65°, +89.00°] |

One number about the arm explains why the band stops where it does. **−26.7573° is the joint-4 angle
at which the arm is exactly straight**: measured by
`python3 examples/08_swift_workspace.py --headless --steps 1`, whose ternary search over joint 4 puts
`‖x_sw‖` at `0.719354203404` m — the closed-form outer radius to the last digit (the example prints
the difference, `0.0e+00`). It is the same angle at every pose tried, which is what "full extension
is where the elbow stops contributing" means. The band is narrow, but it is real: 153
configurations over 300 poses, and 2 poses in 300 that no other branch reaches. A solver that
chooses not to return it should say so rather than report the pose as unreachable;
`franka_ik.solve` returns it and lets the caller weight it.

One caveat on the labels, because it is easy to get backwards. The `+`/`−` names in this repository
are the 2023 rewrite's, and §1 records that the 2021 prototype pinned the *other* side of the same
quadratic. What is established here is the shape: one root was computed, the other was available and
unused. Pinning down which label the 2021 prototype selected would need its conventions reconciled
with the 2023 rewrite's, and it does not run against `franka_ik` as it stands; §1 says so rather
than guessing.

## 4. What this repository actually contributes

Ranked by how much it is worth, most to least:

1. **A completeness measurement, not a completeness claim.** `coverage_study`,
   `solution_count_study`, and `completeness_check` with an independent CasADi + IPOPT
   enumeration. `docs/limitations.md` and the tests above have the rest.
   Measured over **35** random in-limit poses (400 to 600 IPOPT starts each, two seeded samples):
   109 IPOPT solutions, every one of them reaching the pose, **0 counter-examples** — no
   configuration that reaches the pose and sits more than 0.05° from every analytic branch. 105
   land within tolerance of a branch; the 4 that do not all come from one near-singular pose, at
   most 0.0854° away, where a configuration offset of that size moves the pose by 7.8 × 10⁻⁹.
   That is the optimiser stopping short, not a branch the derivation misses. The script prints the
   10-pose subset of this run; `docs/limitations.md` §12 has the singularity.
2. **The self-correction.** The 2023 rewrite dropped a branch relative to the 2021 prototype
   (§1), and nothing in the repository noticed for two years because a solver that misses a
   branch still returns exact poses. The finding in `docs/branch_analysis.md` is, first of all,
   a finding about this repository's own history. The correction is visible in the numbers: with
   only the one root, the target recovery rate is 265 of 300; with both, 300 of 300.
3. **The cross-check against the published files.** 1200 of 1200 label-level comparisons over
   the whole joint-7 range, worst deviation 1.3 × 10⁻¹³ rad (`tests/test_branches.py`) — the
   evidence that `franka_ik` is the same method as `original/` and not a different one wearing
   its name.
4. **The documented defects of the published files**: the `nan` hang in `limit_joints`, the
   symbolic `Panda.fk`, the over-wide limit windows, the vanishing wrist-sign test at
   `q₇ = ±90°`, and the `q₂ = 0` singularity. See `docs/limitations.md`.
5. **The method itself**, which is a clean reduction but not a new one: reduce to an *equivalent*
   S-R-S arm by rotating the wrist offset into a lengthened link, solve that with
   Shimizu et al. (2008), parameterise by joint 7. `docs/method.md` has it equation by equation.

## 5. If you are choosing a solver in 2026

Use someone else's, and use this repository for the numbers and the tests.

* [`juelg/frankik`](https://github.com/juelg/frankik) — analytical IK for Panda and FR3, C++ with
  Python bindings, tested against `franka_ros` and `pinocchio`, maintained.
* For anything that does not need a closed form, a numerical solver with a good seed is faster to
  trust than a closed form with an undocumented branch set.

What this repository is for is the question "how do you know your closed form returns *all* the
configurations" — and the answer, here, is a measurement anyone can re-run.

## 6. Citing

Cite Shimizu et al. (2008) for the closed form this method reduces to. Cite this repository, if at
all, for the completeness measurement and the tests.
