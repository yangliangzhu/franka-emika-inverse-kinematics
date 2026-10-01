# demos

Self-contained interactive HTML reports generated from the library. Open any of
them directly in a browser — no server, no build step and no network access,
because the CSS and the JavaScript are inlined into each page.

| page | what it shows |
|---|---|
| `index.html` | what the method is, the headline measurement, and one pose drawn with all of its solutions |
| `branches.html` | **every** configuration that reaches one pose, drawn on top of each other, with the elbow quadratic that produces them |
| `coverage.html` | does the solver find the configuration the pose came from? — the four-branch subset against the full eight |

## Regenerating

```bash
python3 -c "from franka_ik.report import build_demo_site; build_demo_site('demos')"
```

or, with the study numbers:

```bash
python3 -c "
from franka_ik.report import build_demo_site
build_demo_site(out_dir='demos', samples=300, seed=0)
"
```

## Checking them without a browser

```bash
for f in demos/*.html; do node scripts/check_demo.js "$f" || exit 1; done
```

`scripts/check_demo.js` runs each page's renderer against a minimal DOM and
canvas stub, so a runtime error, a non-finite coordinate or a chart drawn outside
its canvas fails the check. It needs Node.js, not Python, which is why it is not
part of the pytest suite.

## How they work

All the geometry happens in Python. The generator embeds the computed arrays as
`window.FRANKA_IK_DATA` and inlines `franka_ik/web/demo.js`, a dependency-free
renderer that draws a wireframe 3D scene and a multi-panel chart on a `<canvas>`.
Nothing is recomputed in the browser, so what a reader sees is exactly what the
tests assert.

## Interaction

* drag the 3D view to orbit it, scroll to zoom;
* the arm that reproduces the configuration the pose was generated from is drawn
  in full colour and thick, the others faint — toggle them individually with the
  checkboxes under the 3D view;
* branch labels read `q4+` / `q4-` for the root of the elbow quadratic that was
  used, `phi+` / `phi-` for the equivalent arm angle, and `plain` / `flip` for
  the shoulder symmetry.

## The pose on `branches.html`

The generator deliberately looks for a pose with several solutions, because a
randomly chosen configuration often has only two and the drawing would be dull.
The pose currently shipped has **seven** distinct in-limit solutions, and the
configuration it was generated from is recovered by the branch labelled
`q4- phi+ flip` — a branch built on the *negative* root of the elbow quadratic,
which is precisely the half of the solution set the published four-branch solver
discards. For that pose the published solver returns four perfectly valid poses
and misses the one that was asked about. That is the finding this repository is
about, and it is visible in one glance on this page.
