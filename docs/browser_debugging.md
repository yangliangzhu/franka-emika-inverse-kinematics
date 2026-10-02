# Debugging the browser side

How to find out why a Swift viewer (`examples/07`-`11`) shows nothing, shows the wrong
thing, or does not move. Every command and every number here was run on this
repository; the playbook it follows is the sibling S-R-S study's
`../../../srs_ik_analysis/docs/browser_debugging.md`, adapted to the traps this
repository actually hit — all three of which are Swift API traps, not geometry bugs.

## 1. `--headless` draws nothing, by definition

`--headless` runs the whole example without a browser: it prints every number and
exits. If a run shows only text, that is the flag doing its job, and it says nothing
about whether the scene is right.

```bash
uv run --extra viz python examples/07_swift_branches.py --headless --steps 1
```

Use it to separate the two sides: prints and exits 0 -> the analysis is fine and any
problem is in the browser or the launch; fails -> fix the Python first.

## 2. The Swift traps this repository walked into

### 2.1 `Swift.add()` silently ignores a list

```python
env.add(skeleton.shapes)      # returns None, adds nothing, raises nothing
env.add_shape(one_shape)      # returns an object id
```

`Swift.add` dispatches on *one* `Shape`, a `Robot` or a `SwiftElement`, and
`return None` for anything else. Passing the list of shapes therefore produces an
**empty window and no error** — every viewer here opened blank until it was found.
Use `franka_ik.swift_viz.add_shapes(env, shapes)`, which calls `add_shape` per shape
and returns the count so a caller can assert on it.

Check it in one line:

```python
len(sv.add_shapes(env, shapes.shapes))     # 8 for mesh, 30 for collision, 17 for skeleton
```

### 2.2 `time.sleep()` does not render

```python
while True:
    shapes.update(q)
    time.sleep(0.05)          # scene stays on its first frame
    env.step(0.05)            # this is what advances, re-sends and renders
```

`env.step(dt)` increments the simulation clock, pushes any shape whose pose changed,
and renders; it also blocks for `dt`, so it replaces the sleep rather than joining it.
An example that sleeps is a frozen picture, which reads as "the viewer does not work".
`franka_ik.swift_app.hold(env, steps=...)` is the shared loop, and
`interaction_loop(env, slider=...)` is that plus a slider; `env.run()` is the
library's own version when a step limit is not needed.

### 2.3 A Windows browser cannot be launched by path

Under WSL the only browsers are Windows ones, and they live under
`/mnt/c/Program Files`, whose path contains a space. `webbrowser`/`xdg-open` split the
`BROWSER` command on whitespace, so the launch dies with:

```
/usr/bin/xdg-open: 880: /mnt/c/Program: not found
xdg-open: no method available for opening 'http://localhost:52000/?53000'
```

`franka_ik.swift_viz.wsl_browser_wrapper()` writes a space-free `sh` wrapper to the
cache and hands *that* to `BROWSER`; `launch_env(browser="auto")` uses it, and does so
automatically under WSL. Verified: with the wrapper, `launch_env` connects;
without it, the handshake times out.

### 2.4 A slider's initial value is clamped by the range it has not been given yet

This one cost the most time, because it looks like a stale event rather than an
ordering bug. Swift's slider JavaScript is:

```js
this.slider.value = data.value;   // the new value first ...
this.slider.step  = data.step;
this.slider.min   = data.min;     // ... against the *old* range, which the markup
this.slider.max   = data.max;     //     starts at min="0" max="100"
```

so when the range does not contain 0 the initial value is clamped to the nearer end,
`update()` calls `onInput()`, and the element is reported once — on attach — as
changed. Measured: a joint-7 slider built with `value=-58.78` over
`[-77.92, -50.42]` reached Python as `-50.42`, its maximum, and a scene driven from
that callback jumped to the end of its range on the first frame. Two consequences:

* **read `slider.value`, not the callback argument.** `franka_ik.swift_app.add_slider`
  keeps the callback as a no-op — it still has to exist, or the browser does not send
  the element at all — and `interaction_loop` compares `slider.value` with the value
  the scene was built for. Verified in a browser: a value written from Python reads
  back as written (`42.0`, then `42`), and a drag arrives as the value dragged to
  (`-70`), so one attribute is the source of truth in both directions.
* **write the value back after the element is added.** By then the browser has the
  real `min`/`max`, so the same assignment sticks. `add_slider` does it, and
  `interaction_loop` does it again for its `initial`, which is what makes frame 0 the
  scene the caller actually configured.

The tolerance matters as much as the value: `interaction_loop(tolerance=...)` has to
be below the slider's own `step`, or a feature as narrow as the `1e-4` degree failure
window of `examples/10_swift_singularities.py` never triggers a redraw.
`tests/test_swift_app.py` pins both without a browser.

### 2.5 `add_shape` waits for the browser, so a point cloud costs a round trip each

```python
id = self._send_socket("shape", [shape.to_dict()])   # blocks ...
self._wait_mounted(id, 1)                            # ... until the browser confirms
```

`add_shape` is only fire-and-forget in a headless env. With a browser attached, every
sphere in a 400-marker cloud is a request **and** a mount poll. Measured in Chromium
with `--enable-unsafe-swiftshader`: `examples/08_swift_workspace.py` printed its
model summary at 1.5 s and the side panel only appeared at **87 s** with
`--markers 150`. It is not hung — it is 150 round trips — and the same scene is
instant with `--headless`, which is what made it look like a geometry bug.

`franka_ik.swift_viz.add_cloud` is the fix: it hands the whole group to
`add_assembly`, which sends every part in **one** message and waits once, and
`env.remove(handle)` takes the group off screen again in one call. Two things to
know about it:

* the group **may still move**. An assembly is re-posed each frame through its
  `fk(q)`, and the callable `add_cloud` passes reads each part's current pose, so
  writing `shape.T` — or `ArmShapes.update`, which does — still redraws. What it will
  not re-send is a part's *colour*, so colours are set before the group goes in;
* the `fk` must return `SE3`, not the 4x4 array `Shape.T` hands back. An array
  survives every Python-side check — construction, `add_assembly`, four frames —
  and then raises inside Swift's serialiser as `AttributeError: 'numpy.ndarray'
  object has no attribute 't'`. `add_cloud` converts on every call.

Measured after the change: `examples/08` with its default 400 markers opens in
seconds instead of minutes, and `07`'s branch fan — added and removed once per slider
step — goes from never appearing to following the drag. `--markers` is still the knob
for how dense the shell looks.

## 3. Is the geometry right?

The scenes are driven by `franka_ik.model.forward_kinematics`, so a wrong *placement*
cannot come from the solver — it comes from the mapping from a URDF link to a DH frame
(`swift_viz._LINK_TO_FK_FRAME`). Two ways to check, cheapest first.

**Numeric.** At `q = 0` the mesh origins are known exactly:

```bash
uv run --extra viz python -c "
from franka_ik import swift_viz as sv
import numpy as np
m = sv.link_mesh_shapes(sv.find_panda_urdf(), None)
sv.update_link_mesh_shapes(m, sv.find_panda_urdf(), np.zeros(7))
print([np.round(np.asarray(s.T, float)[:3,3], 4).tolist() for s in m])
"
# [[0,0,0], [0,0,0.333], [0,0,0.333], [0,0,0.649], [0.0825,0,0.649],
#  [0,0,1.033], [0,0,1.033], [0.088,0,1.033]]
```

Those are the eight URDF link frames of the arm standing straight up: `link0` is the
**base plate on the floor**, not the shoulder 0.333 m up, and getting that one wrong
is a plausible-looking robot with its base buried in `link1`.
`panda_link8` has no visual (it is the flange marker), so there are **eight** meshes
and the last one is 0.107 m *from* `fk_flange`, not on it.

Three end frames, and mixing them up is a *visual* bug rather than a kinematic one — the
marker is drawn, the numbers are right, and the marker floats. Measured, at any
configuration:

| from | to | distance | direction |
|---|---|---|---|
| wrist frame (`forward_kinematics[6]`) | flange (`fk_flange`) | `0.107` m | the flange's `z` (the DH `d7`) |
| flange | tool (`fk_tool`) | `0.1034` m | the same `z` |
| tool | flange rotation | `-45` degrees | about the flange's `z` |

The meshes stop at the **flange**: `link7.dae`'s vertices span `z ∈ [+0.0520, +0.1068]` in
the link7 frame (measured with `trimesh`, an ad-hoc install), and there is no `link8`
geometry at all. The *skeleton* instead ends at the **tool** point, since its last key
point is the tool frame. So a marker at the tool point is attached to nothing in
`--model mesh` and sits at the tip of the drawn stem in `--model skeleton` — and that
10.34 cm gap is exactly what a reader notices and cannot explain. The examples therefore
draw the flange large (it is the pose the solver solves for, and the frame their pose
residual is measured in), the tool point small, and `swift_viz.tool_stem` between them,
so the distance is a drawn length rather than a surprise.

**Visual.** Draw the meshes and the skeleton together: the skeleton must run down the
middle of the bodies. `_LINK_TO_FK_FRAME` takes a *thin, translucent* skeleton, or the
capsules hide inside the meshes and the picture looks wrong when it is right.

```python
sv.add_shapes(env, mesh_shapes)
sv.add_shapes(env, sv.ArmSkeleton(radius_scale=0.45, alpha=0.85, colour=[0.9, 0.1, 0.1]).shapes)
```

## 4. Seeing the page: Playwright

`Swift.launch` blocks until the page opens its websocket, so the page has to be opened
*while* launch waits. Spy on `webbrowser.open` for the URL, launch in a thread, and
drive a Playwright page from the main thread:

```python
import threading, webbrowser
seen, cap = threading.Event(), {}
def spy(url, *a, **k):
    cap["url"] = url; seen.set(); return True
webbrowser.open = webbrowser.open_new = webbrowser.open_new_tab = spy

threading.Thread(target=lambda: env.launch(realtime=False), daemon=True).start()
seen.wait(timeout=20)

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1000, "height": 800})
    events = []
    page.on("pageerror", lambda e: events.append(f"[pageerror] {e}"))
    page.on("console", lambda m: events.append(f"[{m.type}] {m.text}"))
    page.on("requestfailed", lambda r: events.append(f"[requestfailed] {r.url}"))
    page.goto(cap["url"], wait_until="load")
    ...                                   # add shapes, step, then:
    page.screenshot(path="/tmp/scene.png")
```

Then **look at the screenshot**. A scene can be wrong with no console error at all.

Two things about driving the page:

* **Dispatch DOM events, do not `click()`.** Playwright's `click()` waits for the
  element to be stable and actionable, and a page that re-renders every 50 ms can keep
  it waiting — or, worse, stall the websocket long enough for `env.step` to raise
  `Swift browser tab stopped responding`. Set `value` and dispatch `input`/`change`
  from inside `page.evaluate`, and read the side panel with
  `document.querySelectorAll('#sidenav p, #sidenav button, #sidenav label')`.
  A readout made of `a<br>b` has **element children**, so a
  `children.length === 0` filter silently drops it.
* **Wait on the readout, not on a fixed sleep.** `#sidenav` appears first and fills
  afterwards, and the fill is slow when a cloud of markers is arriving (section 2.5).

`scripts/browser_drive.py` is this recipe, wrapped: it runs one example with Playwright
as its client and asserts on the readout after each interaction.

```bash
pip install playwright && playwright install chromium
python3 scripts/browser_drive.py 10 --model skeleton
```

Install Playwright without touching the project environment:

```bash
uv venv /tmp/pw --python 3.10
VIRTUAL_ENV=/tmp/pw uv pip install playwright
/tmp/pw/bin/playwright install chromium
```

It needs `numpy` and the project on `sys.path` to import the example's scene; the
project's own `.venv` works too if you `uv pip install playwright` into it (the
Chromium download is shared through `~/.cache/ms-playwright`, and `detect_browser()`
will then offer it as a viewer).

`--no-sandbox` is required under WSL; without `--enable-unsafe-swiftshader` the page
can abort before it opens the socket and `launch` times out instead of failing.

## 5. What the console said here

The DAE files are Z-up and three.js's `ColladaLoader` re-orients them, which is why
`link_mesh_shapes` passes `y_up=True`: Swift then applies its own `Rx(+90°)`
correction and the arm stands up. With `y_up=False` the whole robot is rendered tipped
90 degrees about X, and the console says so:

```
[warning] THREE.ColladaLoader: You are loading an asset with a Z-UP coordinate
          system. The loader just rotates the asset to transform it into Y-UP.
```

That warning is expected and correct for `--model mesh`. It is not a defect and not
something to silence.

## 6. Checklist

1. `--headless`: is the Python side fine?
2. Text but no window -> you passed `--headless`; drop it.
3. Window open and blank -> did you use `env.add(...)` on a **list**? Use
   `sv.add_shapes`.
4. Window open, arm visible, frozen -> is the loop calling `env.step(dt)`?
   `time.sleep` alone renders nothing new.
5. No window at all -> the `xdg-open`/space-in-path trap; use `--browser auto`.
6. The scene jumps to one end of a slider's range as soon as it opens -> the
   clamp in section 2.4; build sliders with `swift_app.add_slider`.
7. A slider does nothing for one pixel of travel -> the tolerance is above the step,
   or above the feature; think in the units of the thing you are looking for
   (`1e-4` degrees, in `examples/10`).
8. Slow to appear with a browser but instant headless -> it is the shape upload
   (section 2.5), not a hang. Lower `--markers`.
9. Wrong pose or a tipped-over robot -> check `_LINK_TO_FK_FRAME` and `y_up` with
   section 3 and section 5.
10. Before calling it fixed: `pytest -q`, `ruff check .`, and a run of
    `scripts/browser_drive.py` for the example you touched, in both
    `--model mesh` and `--model skeleton`.
