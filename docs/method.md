# The method

How the Franka Emika Panda — which is *not* an S-R-S arm — is reduced to one and solved in
closed form, exactly in the four steps of `franka解析反解方法.pdf`.

The same derivation, typeset as a single self-contained HTML page — the formulas as real math
instead of images, the hand sketches redrawn as vector figures, and rendered screenshots of the
Panda and of an S-R-S arm (the KUKA iiwa 14) alongside — is
[`method_illustrated.html`](method_illustrated.html). It needs no network and opens straight from
the file system.

This document follows the derivation of the handwritten PDF at the repository root
(`franka解析反解方法.pdf`, four pages: `STEP1`–`STEP4` on pages 1–3, the special-equation
appendix at the end of page 3, references on page 4). The formulas in that PDF are images, so
the same content is carried in three places that can be read side by side:

| source | what it carries |
|---|---|
| `docs/original_notes_zh.md` | the author's own summary of the derivation, in Chinese |
| `franka_ik/geometry.py`, `franka_ik/solver.py` | the derivation in English, module by module, with the measurements |
| `original/ik_ca.py`, `original/ik_ca2.py` | the 2020 implementation, kept verbatim |

The closed form that is being reduced *to* is [Shimizu, Kakuya, Yoon, Kitagaki, Kosuge,
"Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits
and Its Application to Redundancy Resolution", IEEE T-RO 24(5):1131–1142,
2008](https://doi.org/10.1109/TRO.2008.2003266). Equation numbers below (`(3)`, `(12)`,
`(15)`, `(19)`) are that paper's.

---

## 1. The obstruction

A seven-axis S-R-S arm — shoulder, elbow, wrist, with three consecutive axes intersecting at
each of the shoulder and the wrist — has a closed-form inverse kinematics whose redundancy is
one-parameter. The Panda satisfies neither condition:

| what | where | value |
|---|---|---|
| shoulder bias `b` | DH rows 4 and 5, `a = ±0.0825` | 0.0825 m |
| wrist offset | DH row 7, `a = 0.088` | 0.088 m |

The shoulder bias is the important one: the three shoulder axes do not intersect, so joint 4
does **not** obey the S-R-S elbow law. The wrist offset is the cheap one: it is a fixed
rotation of the last link, and it can be absorbed.

The reduction removes the wrist offset first (`STEP1`), then absorbs the shoulder bias into the
link lengths (`STEP2`). The result is not one S-R-S arm but a one-parameter **family** of them,
and `STEP3` picks the member that reproduces the requested joint 7. `STEP4` finishes the S-R-S
solution and undoes `STEP1`.

## 2. Notation

All lengths in metres, all angles in radians unless stated.

| symbol | meaning | value | code |
|---|---|---|---|
| `d_bs` | base → shoulder | 0.333 | `PAPER_GEOMETRY.d_bs` |
| `d_se` | shoulder → elbow (DH row 3) | 0.316 | `PAPER_GEOMETRY.d_se` |
| `d_ew` | elbow → wrist (DH row 5) | 0.384 | `PAPER_GEOMETRY.d_ew` |
| `d_wt` | wrist → flange (DH row 8) | 0.107 | `PAPER_GEOMETRY.d_wt` |
| `offset` | wrist offset, DH row 7 `a` | 0.088 | `PAPER_GEOMETRY.offset` |
| `b` | shoulder bias, DH rows 4–5 `a` | 0.0825 | `PAPER_GEOMETRY.bias` |
| `β` | extra flange rotation caused by the offset | 39.4349° | `wrist_offset_angle` |
| `ℓ_wt` | last link length after `STEP1` | 0.13853880 | `effective_wrist_length` |
| `x_sw` | shoulder → wrist vector of the equivalent arm | — | `shoulder_to_wrist` |
| `u_sw` | `x_sw / ‖x_sw‖` | — | `solver._skew` input |
| `φ` | equivalent arm angle (the free parameter of `STEP3`) | — | `solver.solve_branch` |
| `δ` | link growth of the equivalent arm | — | `link_offset` |
| `q₄` | joint 4 of the *reduced* arm, i.e. the elbow angle | — | `q4_roots` |

`PAPER_GEOMETRY` is the parameter set the 2020 implementation uses, and the one all numbers in
these documents were measured with:

```python
from franka_ik import PAPER_GEOMETRY
# EquivalentGeometry(d_bs=0.333, d_se=0.316, d_ew=0.384, d_wt=0.107, offset=0.088, bias=0.0825)
```

### 2.1 Two DH conventions meet here

This is the single most confusing thing in the repository, so it is worth stating plainly.

* `franka_ik/model.py` uses the **modified** (Craig) DH transform
  `RotX(αᵢ)·TransX(aᵢ)·TransZ(dᵢ)·RotZ(θᵢ)`, chained by right multiplication. This is the
  convention of `panda.py`, and it is the one that evaluates the forward kinematics.
* The reduction works in the **standard** DH rotation of the paper's eq. (3),
  `Rz(θ)·Rx(α)`, with the seven `α` values carried in
  `franka_ik.geometry.DEFAULT_ALPHAS` = `[-1, 1, 1, -1, 1, 1, 0]·π/2`.
  `franka_ik.geometry.equivalent_joint_rotation(θ, i)` returns exactly that.

The two are never converted into each other symbolically; they are reconciled **numerically**,
which is why the port is verified against the published implementation rather than by
inspection. See `docs/limitations.md`.

## 3. `STEP1` — make the flange frame look like an S-R-S flange

The last link of an S-R-S wrist meets the flange along the joint-6 axis. On the Panda the
`offset = 0.088` of DH row 7 tilts it: the hand-written sketch on page 1 of the PDF shows the
joint-6 axis and the `A₇` frames of the Franka and of the S-R-S arm differing by one fixed
angle `β`. That angle, and the length it costs, are

```math
\beta = \operatorname{atan2}(\text{offset},\; d_{wt}) = 39.4349^\circ,
\qquad
\ell_{wt} = \sqrt{\text{offset}^2 + d_{wt}^2} = 0.13853880\ \text{m}.
```

Rotating the target orientation by

```math
R_{\text{srs}} = R_{\text{franka}}\; R_z(-q_7)\, R_y(-\beta)\, R_z(q_7)
```

makes the flange frame behave exactly like an S-R-S flange. The correction is applied *on the
right*, i.e. in the flange's own frame, and it is a rotation by `β` about the flange `y` axis
as seen after a rotation of `-q₇` about the flange `z` axis.

The price is the second half of the pair above: the distance from the wrist point to the
flange, which was `d_wt`, is now `ℓ_wt`. Everything downstream uses `ℓ_wt`, which is why the
shoulder-to-wrist vector is

```math
{}^{0}x_{sw} = x - [0,0,d_{bs}] - R_{\text{srs}}\,[0,0,\ell_{wt}].
```

| code | role |
|---|---|
| `geometry.wrist_offset_angle` | `β` |
| `geometry.effective_wrist_length` | `ℓ_wt` |
| `geometry.wrist_correction(q7)` | the matrix `Rz(-q₇) Ry(-β) Rz(q₇)` |
| `geometry.shoulder_to_wrist(x, R_srs)` | `x_sw` |

Note the sign convention of `geometry.rot_y`: it is `[[c, 0, -s], [0, 1, 0], [s, 0, c]]`,
matching the published implementation. Flipping it mirrors the correction and every branch
returns a wrong pose — see `docs/limitations.md` for the measured residual.

After `STEP1` the arm is what the PDF calls **S-R-S+**: identical to S-R-S except for the
offset at joint 4.

## 4. `STEP2` — absorb the shoulder bias into the links

Page 2 of the PDF draws the two triangles side by side: the real one, with the elbow displaced
by `b` at joint 4, and the equivalent one with adjusted link lengths.

Because joint 4 is displaced by `b`, the plain S-R-S elbow law
`cos θ₄ = (‖x_sw‖² - d_se² - d_ew²)/(2 d_se d_ew)` — eq. (12) of the paper — no longer holds.
Solving the same triangle *with* the bias gives, instead, a **quadratic in `tan(θ₄/2)`**:

```math
a_2 x^2 - a_1 x + a_0 = 0, \qquad x = \tan\frac{\theta_4}{2},
```

```math
a_2 = 4b^2 + (d_{se}-d_{ew})^2 - \|x_{sw}\|^2, \qquad
a_1 = 4b\,(d_{se}+d_{ew}), \qquad
a_0 = (d_{se}+d_{ew})^2 - \|x_{sw}\|^2 .
```

The appendix of the PDF (*特殊方程的求解*, "solving the special equation") is the solution of
this quadratic. The roots are

```math
\tan\frac{\theta_4}{2} = \frac{a_1 \pm \sqrt{a_1^2 - 4a_0a_2}}{2a_2},
\qquad
\theta_4 = 2\arctan\!\left(\frac{a_1 \pm \sqrt{\cdot}}{2a_2}\right).
```

Setting `b = 0` collapses this back to eq. (12), whose two solutions are
`θ₄ = ±arccos(·)`; the two roots above play exactly that role. With `b ≠ 0` they are no longer
symmetric about zero — for one sampled pose they came out as **−70.38°** and **+16.87°**, where
the bias-free analogue would give ±34.79°.

Once `θ₄` is fixed, the equivalent links are obtained by *lengthening* both of them by

```math
\delta = -\tan(\theta_4/2)\; b,
\qquad
l_{se} = [0,\; d_{se} + \delta,\; 0], \qquad
l_{ew} = [0,\; 0,\; d_{ew} + \delta].
```

`δ` is zero when the bias is zero, which is what makes the reduction exact for a true S-R-S
arm. Since `δ` depends on `θ₄`, the object `STEP2` produces is **a one-parameter family of
equivalent S-R-S arms**, indexed by `θ₄` — the PDF makes this point explicitly on page 2. The
code's `equivalent_link_vectors(q4)` returns the two links of one member.

| code | role |
|---|---|
| `geometry.q4_coefficients(D)` | `(a₂, a₁, a₀)` for `D = ‖x_sw‖²` |
| `geometry.q4_discriminant(D)` | `a₁² − 4a₀a₂`; negative means out of reach |
| `geometry.q4_roots(D)` | **both** roots as angles, `+` first |
| `geometry.link_offset(q4)` | `δ` |
| `geometry.equivalent_link_vectors(q4)` | `(l_se, l_ew)` |

### 4.1 The reachable shell

The discriminant is a quadratic in `D = ‖x_sw‖²`,

```math
\operatorname{disc}(D) = -D^2 + (A+B+C)\,D - AB,
\qquad A = (d_{se}+d_{ew})^2,\quad B = (d_{se}-d_{ew})^2,\quad C = 4b^2,
```

so the set of shoulder-to-wrist distances the equivalent arm can reach is available in closed
form rather than by search (`analysis.reachable_distance_range`):

| arm | shell of `‖x_sw‖` |
|---|---|
| Panda, `b = 0.0825` | **0.06617 m … 0.71935 m** |
| the same links with `b = 0` (a true S-R-S arm) | 0.06800 m … 0.70000 m |

The bias is therefore not only what makes the elbow law quadratic — it also *widens* the shell
at both ends, because the offset lets the elbow fold slightly further in and reach slightly
further out. An independent scan of `q4_discriminant` over 0.046–0.739 m at 4001 points gives
0.06627–0.71925 m, i.e. the same interval to the grid step of 1.73 × 10⁻⁴ m.

## 5. `STEP3` — pick the equivalent arm angle

### 5.1 The family of shoulder rotations, and what `φ` means

The shoulder rotation `R₀₃` must send the *local* elbow vector
`e = l_se + R₄(q₄) l_ew` to `x_sw`. Every rotation that does so is a rotation about `u_sw`
applied to one particular solution, so the whole family is

```math
R_{03}(\varphi) = \operatorname{Rot}(u_{sw}, \varphi)\; R_{03}^{\text{ref}},
```

and this is exactly what the code's three matrices encode:

```math
A_s = \operatorname{skew}(u_{sw}) R_{03}^{\text{ref}}, \qquad
B_s = -\operatorname{skew}(u_{sw}) A_s, \qquad
C_s = u_{sw} u_{sw}^{\top} R_{03}^{\text{ref}},
```

```math
R_{03}(\varphi) = A_s \sin\varphi + B_s \cos\varphi + C_s
               = \operatorname{Rot}(u_{sw}, \varphi) R_{03}^{\text{ref}} .
```

The last equality is Rodrigues' formula once `u uᵀ = I + skew(u)²` is used, so `φ` is literally
**the arm angle about the shoulder–wrist line**: it rotates the whole arm plane, shoulder,
elbow and all, around the line from shoulder to wrist. This is eq. (15) of the paper, and the
counterpart for the wrist, eq. (19), is

```math
A_w = R_4^{\top} A_s^{\top} R_{\text{srs}},\quad
B_w = R_4^{\top} B_s^{\top} R_{\text{srs}},\quad
C_w = R_4^{\top} C_s^{\top} R_{\text{srs}},
\qquad
R_{47}(\varphi) = A_w \sin\varphi + B_w \cos\varphi + C_w .
```

### 5.2 The reference plane

`R₀₃^ref` is the shoulder rotation of the arm plane at `q₃ = 0`. The code builds the local
elbow vector at `q₃ = 0`,

```math
x_{aux} = R_3(0)\,\bigl(l_{se} + R_4(q_4)\,l_{ew}\bigr),
\qquad y_{aux} = x_{sw},
```

and solves `R₁(q₁) R₂(q₂) r = x_sw` for the two reference shoulder angles with

```python
amplitude1 = hypot(x_aux[0], x_aux[2]);  amplitude2 = hypot(y_aux[0], y_aux[1])
q1_ref = asin(-x_aux[1] / amplitude2) + atan2(y_aux[1], y_aux[0])
q2_ref = asin(-y_aux[2] / amplitude1) + atan2(x_aux[2], x_aux[0])
r_03_ref = R1(q1_ref) @ R2(q2_ref) @ R3(0)
```

Only the principal branch of the arc-sine is taken, and that is enough: the reference plane is
only a *reference*, and every other arm plane is reached by a different `φ` in §5.3. The
rotations `R₁, R₂, R₃` are the standard-DH ones of `geometry.equivalent_joint_rotation`.

### 5.3 The equation that fixes `φ`

`q₇` has already been chosen by the caller — it parameterises the redundancy. For the answer to
be consistent, the wrist rotation must produce that `q₇`. Reading the wrist of the reduced arm,
whose last three `α` are `π/2, π/2, 0`, the relevant entries are

```math
\left(R_{47}\right)_{3,1} = \sin q_6 \cos q_7,
\qquad
\left(R_{47}\right)_{3,2} = -\sin q_6 \sin q_7,
```

so the condition `q₇ = requested` is equivalent to the linear condition

```math
\left(R_{47}\right)_{3,2} + \tan(q_7)\,\left(R_{47}\right)_{3,1} = 0 .
```

Substituting `R₄₇(φ) = A_w sinφ + B_w cosφ + C_w` turns this into a sinusoid equation in `φ`:

```math
c_0 \sin\varphi + c_1 \cos\varphi + c_2 = 0,
\qquad
c_i = (A_w)_{3,2} + \tan(q_7)\,(A_w)_{3,1}\ \text{for } A_w, B_w, C_w .
```

Writing `(c₀, c₁) = ρ (cos δ, sin δ)` with `ρ = hypot(c₀, c₁)` and `δ = atan2(c₁, c₀)`, this
is `ρ sin(φ + δ) = -c₂`, the "和差化积" (sum-to-product) step of the PDF. Hence, whenever
`|c₂| ≤ ρ`, there are exactly **two** arm angles:

```math
\boxed{\;\varphi = \pi - \delta - \arcsin\!\Bigl(\frac{-c_2}{\rho}\Bigr)
\qquad\text{or}\qquad
\varphi = -\delta + \arcsin\!\Bigl(\frac{-c_2}{\rho}\Bigr).\;}
```

`|c₂| > ρ` means no arm plane places joint 7 at the requested value, and the branch does not
exist for this pose — the code returns `None` there, and `analysis.classify_failure` reports it
as `arm_angle_singular`.

### 5.4 The form the code evaluates

`solver.solve_branch` does not build `c` with the tangent. It multiplies the whole equation by
`cos(q₇)` first — exact, because it scales all three coefficients by the same constant, so the
roots are unchanged — and evaluates

```python
sin_q7, cos_q7 = math.sin(q7), math.cos(q7)
coeff = np.array([W[2, 1] * cos_q7 + W[2, 0] * sin_q7 for W in (A_w, B_w, C_w)])
```

At `q₇ = ±90°` this is simply `W[2,0]`, which is the correct limit, whereas `tan(q₇)` is not
finite there. The published implementation uses the tangent form and carries a comment about it
(`#. tan(q7) 为无穷则需要调整`), but the tangent turns out not to be what fails at `±90°` — see
`docs/limitations.md` §7.

Multiplying by `cos(q₇)` has one consequence that has to be handled explicitly: for
`cos(q₇) < 0` the three coefficients are *negated*, which sends `δ = atan2(c₁, c₀)` to `δ + π`
and `sine = -c₂/ρ` to `-sine`, and the two roots of the sinusoid trade places. The code
therefore folds the sign in when it picks a root,

```python
take_first = (phi_root > 0) == (cos_q7 >= 0.0)
phi = math.pi - delta - arc if take_first else -delta + arc
```

so that `phi_root = +1` means "the first root of the sinusoid, as the published implementation
numbers it" at every `q₇`, not only for `|q₇| < 90°`. Without that step the branch *labels* would
silently swap in half of the joint-7 range, while the solution set stayed the same.

In code this is the block of `solver.solve_branch` that builds `coeff`, `sine`, `delta` and then
selects on `phi_root`.

## 6. `STEP4` — finish, and undo `STEP1`

With `φ` known, the two rotation matrices of eqs. (15) and (19) follow, and the joints are read
straight off them, exactly as in the S-R-S paper:

```python
r_03 = A_s * sin_phi + B_s * cos_phi + C_s
r_47 = A_w * sin_phi + B_w * cos_phi + C_w

q1 = atan2(r_03[1, 1], r_03[0, 1])
q2 = acos(clip(r_03[2, 1]))
q3 = atan2(-r_03[2, 2], -r_03[2, 0])

q5 = atan2(r_47[1, 2], r_47[0, 2])
q6 = acos(clip(-r_47[2, 2]))
# the wrist has a second configuration, (q5 + pi, -q6); keep the one that reaches the pose
candidates = [with_wrist(q5, q6), with_wrist(q5 + pi, -q6)]
q = min(candidates, key=lambda c: np.max(np.abs(fk_flange(c) - pose)))
q6 -= beta                  # undo STEP1 on joint 6
```

Two things in that block are not straight out of the paper:

* **The wrist choice.** `acos` returns `q₆ ∈ [0, π]`, but the branch has a second posture,
  `(q₅ + π, −q₆)`, and the matrix alone does not say which one reaches the pose. Reading the
  third row of `R₄₇` as `[sin q₆ cos q₇, −sin q₆ sin q₇, −cos q₆]` shows why the two are so
  close: the *published* code picks between them with

  ```math
  \texttt{criteria} = (R_{47})_{3,1}\cos q_7 = \sin(q_6)\cos^2(q_7),
  ```

  i.e. a test of the sign of `q₆`, masked by `cos²(q₇)` — which vanishes at `q₇ = ±90°` and
  takes the decision with it. `franka_ik` therefore evaluates both candidates and keeps the one
  whose forward kinematics actually reaches the target; the published criterion is still used
  when `check_pose=False`. Measured over 200 random poses, the kept candidate is the flipped one
  in 412 of 1104 branches (37 %), and flipped and unflipped branches are equally accurate (worst
  pose residual 1.97 × 10⁻¹⁴ and 1.98 × 10⁻¹⁴ respectively).
  `IkSolution.wrist_flipped` records the decision. `docs/limitations.md` §7 has the measurement
  of what the published criterion does wrong at `±90°`.
* **`q6 -= β`** is `STEP1` undone: the last link's offset was compensated on the orientation,
  and it has to be compensated on the wrist angle too, so that `q₆` is a Franka joint angle
  again.

Finally the shoulder-flip symmetry `(q₁, q₂, q₃) → (q₁+π, −q₂, q₃+π)` may be applied; it is a
genuine two-fold symmetry of this arm (138/138 random configurations, `docs/limitations.md`).

## 7. The eight branches

Four independent two-way choices appear on the way, giving the eight candidate configurations
the solver enumerates:

| choice | meaning | selected by |
|---|---|---|
| `q4_root` | which root of the `STEP2` quadratic — which side of the shoulder–wrist line the elbow lies on | `solve_branch(q4_root=±1)` |
| `phi_root` | which of the two arm angles of §5.3 reproduces `q₇` | `solve_branch(phi_root=±1)` |
| `shoulder_flip` | whether the shoulder-flip symmetry is applied | `solve_branch(shoulder_flip=…)` |
| `wrist_flipped` | which of the two wrist postures `(q₅, q₆)` / `(q₅+π, −q₆)` is kept | not enumerated; the pose test decides inside |

`NUM_BRANCHES = 8`. `branch_solutions` evaluates all eight (in a fixed order) and omits the
ones that do not exist for the pose; `solve` additionally drops the ones whose forward-kinematics
residual exceeds `tolerance` and then merges coincident configurations.

That the eight-enumeration is also **complete** — not merely sufficient — is the empirical claim
of this repository, argued and measured in `docs/branch_analysis.md`.

## 8. Verification

All numbers below were measured with the code in this repository; the commands are in
`docs/branch_analysis.md` and `docs/api.md`.

| measurement | value |
|---|---|
| NumPy model vs published CasADi model (`fk_flange`, `fk_tool`) over 300 random configurations | 3.3 × 10⁻¹⁶ |
| NumPy `jacobian` vs the published `Panda.jacobian_flange` | 5.6 × 10⁻¹⁶ |
| library vs published solver, comparison by comparison, whole joint-7 range | 1200/1200 matched, worst deviation 1.30 × 10⁻¹³ rad, none above 10⁻⁹ |
| pose residual of returned solutions | median 4.4 × 10⁻¹⁶, worst 2.0 × 10⁻¹⁴ over 1104 branch evaluations |
| target recovered, published four-branch subset (300 samples, seed 0) | 265/300 = 88.3 % |
| target recovered, full eight-branch solver (same sample) | 300/300 = 100 % |
| distinct in-limit solutions per pose | mean 3.15, range 1–8 |
| runtime, all eight branches, pure NumPy | ≈ 1.3 ms per pose |

The derivation itself is the 2020 one and is correct; what the verification adds is the
knowledge that the published *branch enumeration* was incomplete. That story is
`docs/branch_analysis.md`.
