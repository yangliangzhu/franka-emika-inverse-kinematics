#!/usr/bin/env python3
"""Drive one of the Swift examples in a real browser, and report on what it said.

The pytest suite covers the maths and the geometry, and ``--headless`` covers the
Python side of the viewers, but the part that only exists in a browser -- sliders,
buttons, radios, and whether a drag actually redraws the arm -- has no other witness.
This script is that witness: it runs an example with Playwright's Chromium as the
Swift client, performs the example's own interactions, and asserts on the **readout**
the example prints, which is why every example's readout carries its measurement
rather than a label like "solutions: 7".

It is not part of the test suite because it needs Playwright and a WebGL-capable
Chromium, and because it takes tens of seconds per example.  Install it ad hoc::

    pip install playwright && playwright install chromium
    python3 scripts/browser_drive.py 10 --model skeleton

``--model skeleton`` is the quick one; ``--model mesh`` exercises the vendored Panda
description's meshes instead.  Note that ``examples/08`` uploads one marker at a time
and each upload waits for the browser to mount it (``docs/browser_debugging.md``
section 2.5), so give it ``--markers`` on the small side when driving it.

Two Swift behaviours are worked around here and both are documented in that file:
the side panel is written *after* the page opens, so the script polls for the readout
rather than sleeping; and the page re-renders every frame, so interaction goes through
dispatched DOM events rather than Playwright's actionability-checked ``click()``.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The text that means "the panel is ready".  A readout is added before the slider
#: beside it and is filled in last, so waiting for "some text" is a race -- the first
#: version of this script lost it, and reported a working drag as a broken slider.
EXPECT: Dict[str, str] = {
    "07": "solution 1 of",
    "08": "|x_sw|",
    "09": "in-limit:",
    "10": "failure window",
    "11": "step 1 of",
}

#: Arguments an example needs for its scenario's assertions to mean anything; the
#: caller's own arguments are appended, and argparse keeps the last of a repeated flag.
DEFAULT_EXTRA: Dict[str, List[str]] = {
    "07": ["--pose", "second_root"],
    "09": ["--pose", "second_root"],
    "11": ["--plan-steps", "40"],
}

#: The examples this script knows how to drive, by their number.
EXAMPLES: Dict[str, str] = {
    "07": "examples/07_swift_branches.py",
    "08": "examples/08_swift_workspace.py",
    "09": "examples/09_swift_elbow_roots.py",
    "10": "examples/10_swift_singularities.py",
    "11": "examples/11_swift_tracking.py",
}


class Report:
    """Collects ``check`` results and prints them as they happen."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        """Record one assertion and echo it."""
        self.results.append((name, bool(ok), detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""), flush=True)

    @property
    def failures(self) -> int:
        """How many checks failed."""
        return sum(1 for _, ok, _ in self.results if not ok)


class Panel:
    """The side panel of a running Swift page: reading it, and poking it."""

    def __init__(self, page: object) -> None:
        self.page = page

    def leaves(self) -> List[str]:
        """Every label, readout and button text in the panel.

        ``p``, ``label`` and ``button`` rather than "leaf elements": a readout whose
        lines are joined with ``<br>`` has element children, so a leaf filter drops
        exactly the text a readout is made of.
        """
        return self.page.evaluate(
            """() => [...document.querySelectorAll('#sidenav p, #sidenav label, #sidenav button')]
                     .map(e => (e.innerText || '').trim())
                     .filter(Boolean)"""
        )

    def find(self, needle: str) -> str:
        """The first panel text containing ``needle``, or the empty string."""
        for text in self.leaves():
            if needle in text:
                return text
        return ""

    def sliders(self) -> List[Dict[str, float]]:
        """The range inputs, with the numbers the browser is actually holding."""
        return self.page.evaluate(
            """() => [...document.querySelectorAll('#sidenav input[type=range]')].map(s => ({
                   min: Number(s.min), max: Number(s.max), step: Number(s.step),
                   value: Number(s.value)}))"""
        )

    def set_slider(self, index: int, value: float) -> None:
        """Move a slider the way a drag does: set the value, then fire the events."""
        self.page.evaluate(
            """([i, v]) => {
                const s = document.querySelectorAll('#sidenav input[type=range]')[i];
                s.value = String(v);
                s.dispatchEvent(new Event('input', {bubbles: true}));
                s.dispatchEvent(new Event('change', {bubbles: true}));
            }""",
            [index, value],
        )

    def click_radio(self, index: int) -> None:
        """Select the ``index``-th radio in the panel."""
        self.click("input[type=radio]", index)

    def click_checkbox(self, index: int) -> None:
        """Toggle the ``index``-th checkbox in the panel."""
        self.click("input[type=checkbox]", index)

    def click(self, selector: str, index: int) -> None:
        """Activate the ``index``-th element matching ``selector``."""
        self.page.evaluate(
            """([sel, i]) => { const items = [...document.querySelectorAll(sel)];
                               if (items[i]) { items[i].click(); } }""",
            [selector, index],
        )

    def click_button(self, needle: str) -> Optional[str]:
        """Press the first button whose text matches ``needle``; return its old text."""
        return self.page.evaluate(
            """(needle) => {
                const b = [...document.querySelectorAll('#sidenav button')]
                    .find(x => new RegExp(needle, 'i').test(x.innerText));
                if (!b) { return null; }
                const text = b.innerText;
                b.click();
                return text;
            }""",
            needle,
        )

    def wait_for(self, needle: str, timeout: float) -> bool:
        """Poll until some panel text contains ``needle``."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.find(needle):
                return True
            time.sleep(0.25)
        return False

    def digest(self, path: str) -> str:
        """A screenshot's hash, for "did the scene change?" without an eye."""
        data = self.page.screenshot(path=path)
        return hashlib.sha1(data).hexdigest()[:12]


def _number_after(text: str, marker: str) -> Optional[float]:
    """The first number following ``marker`` in ``text``, or ``None``.

    A readout line like ``q7 = -63.92 deg`` is compared numerically because the
    browser's slider holds a multiple of its step, not the number that was asked for.
    The marker matters: a bare "first number" search finds the 7 in "q7".
    """
    match = re.search(re.escape(marker) + r"\s*=?\s*([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _drive_07(panel: Panel, report: Report, shots: Path) -> None:
    """A joint-7 slider that re-solves, a play button, and a radio."""
    report.check("ui appeared", bool(panel.sliders()), f"{len(panel.sliders())} slider(s)")
    if not panel.sliders():
        return
    before = panel.find("solution")
    report.check("readout names the solution", "solution" in before, before[:60])
    first = panel.digest(str(shots / "07-a.png"))
    slider = panel.sliders()[0]
    middle = (slider["min"] + slider["max"]) / 2
    panel.set_slider(0, middle)
    time.sleep(3)
    # The range input holds a multiple of its step, anchored at ``min`` -- measured, a
    # request for -64.17 came back as -63.92 on a grid of 0.5 from -77.9166 -- so the
    # check is that the readout followed the slider, not that it holds the request.
    line = panel.find("q7 =")
    reported = _number_after(line, "q7")
    report.check(
        "the readout follows the slider",
        reported is not None and abs(reported - middle) <= 2 * slider["step"],
        f"want q7 ~ {middle:.2f} (step {slider['step']:g}), readout {reported} from {line[:34]!r}",
    )
    report.check("the scene redrew", first != panel.digest(str(shots / "07-b.png")))
    panel.click_button("play")
    time.sleep(2)
    running = panel.find("q7 =")
    time.sleep(5)
    report.check("play advances on its own", running != panel.find("q7 ="),
                 f"{running[:28]!r} -> {panel.find('q7 =')[:28]!r}")
    panel.click_button("pause")


def _drive_08(panel: Panel, report: Report, shots: Path) -> None:
    """The elbow slider, and the inner-surface checkbox."""
    sliders = panel.sliders()
    report.check("ui appeared", bool(sliders), f"{len(sliders)} slider(s)")
    if not sliders:
        return
    report.check(
        "the slider spans the elbow's one-sided range",
        abs(sliders[0]["min"] + 175.0) < 1e-9 and abs(sliders[0]["max"] + 5.0) < 1e-9,
        f"[{sliders[0]['min']}, {sliders[0]['max']}]",
    )
    report.check("the pose's own reach is on the readout", "0.54405" in panel.find("|x_sw|"),
                 panel.find("|x_sw|")[:60])
    # ``-27.0``, not the measured ``-26.7573``: a range input holds a multiple of its
    # step, which is 0.5 here, and the browser reports what it actually holds.
    panel.set_slider(0, -175.0)
    time.sleep(2.5)
    folded = panel.digest(str(shots / "08-folded.png"))
    report.check("a folded arm reaches 0.20685 m", "0.20685" in panel.find("|x_sw|"),
                 panel.find("|x_sw|")[:60])
    panel.set_slider(0, -27.0)
    time.sleep(2.5)
    stretched = panel.find("|x_sw|")
    report.check(
        "full stretch is the outer radius itself",
        "0.71935" in stretched and "100.0" in stretched,
        stretched[:80],
    )
    report.check("the elbow moved the arm", folded != panel.digest(str(shots / "08-stretch.png")))
    before = panel.digest(str(shots / "08-before-inner.png"))
    panel.click_checkbox(0)
    time.sleep(2.5)
    report.check("the inner surface appears", before != panel.digest(str(shots / "08-inner.png")))


def _drive_09(panel: Panel, report: Report, shots: Path) -> None:
    """A joint-7 slider that re-solves, and a radio that switches the view."""
    sliders = panel.sliders()
    report.check("ui appeared", bool(sliders), f"{len(sliders)} slider(s)")
    if not sliders:
        return
    start = panel.find("in-limit:")
    # ``--pose second_root`` is what the script runs, and it has seven solutions; the
    # count is the example's own claim, and it has to move when joint 7 does.
    report.check("the pose has the documented seven solutions", "in-limit: 7" in start, start[:70])
    roots = panel.digest(str(shots / "09-roots.png"))
    span = sliders[0]["max"] - sliders[0]["min"]
    panel.set_slider(0, sliders[0]["min"] + 0.35 * span)
    time.sleep(3)
    moved = panel.find("in-limit:")
    report.check("joint 7 re-solves", moved != start, f"{start[:34]!r} -> {moved[:34]!r}")
    panel.click_radio(1)  # the "all" option
    time.sleep(3)
    report.check("the radio switches to every solution", "(all)" in panel.find("drawn:"),
                 panel.find("drawn:")[:60])
    report.check("the scene redrew", roots != panel.digest(str(shots / "09-all.png")))


def _drive_10(panel: Panel, report: Report, shots: Path) -> None:
    """The joint-2 offset slider: the failure window, and the presets."""
    sliders = panel.sliders()
    report.check("ui appeared", bool(sliders), f"{len(sliders)} slider(s)")
    if not sliders:
        return
    report.check(
        "the slider is fine enough for the window",
        abs(sliders[0]["step"] - 1e-5) < 1e-15 and abs(sliders[0]["max"] - 0.01) < 1e-12,
        f"step {sliders[0]['step']:g}, +/-{sliders[0]['max']:g}",
    )
    report.check(
        "it starts on the singularity",
        "INSIDE" in panel.find("failure window") and "q2 = 0.00000" in panel.find("q2 ="),
        f"{panel.find('q2 =')[:44]!r}",
    )
    zero = panel.digest(str(shots / "10-zero.png"))
    panel.set_slider(0, 1e-4)
    time.sleep(2.5)
    report.check("1e-4 degrees recovers two solutions", "solved (2" in panel.find("solved ("),
                 panel.find("solved (")[:60])
    report.check("the arm moved", zero != panel.digest(str(shots / "10-1e-4.png")))
    panel.set_slider(0, 0.0)
    time.sleep(2.5)
    report.check("0 is inside the window again", "INSIDE" in panel.find("failure window"),
                 panel.find("failure window")[:60])
    panel.click_radio(2)  # the "0.0001" preset
    time.sleep(2.5)
    report.check("a preset jumps the slider", "solved (2" in panel.find("solved ("),
                 panel.find("solved (")[:60])


def _drive_11(panel: Panel, report: Report, shots: Path) -> None:
    """Scrubbing, playing, and the speed slider."""
    sliders = panel.sliders()
    report.check("ui appeared", len(sliders) == 2, f"{len(sliders)} slider(s)")
    if len(sliders) != 2:
        return
    report.check("the plan starts paused", "playing: False" in panel.find("playing:"),
                 panel.find("playing:")[:60])
    panel.click_button("play")
    time.sleep(1)
    first = panel.find("step ")
    time.sleep(4)
    report.check("play advances the plan", first != panel.find("step "),
                 f"{first[:30]!r} -> {panel.find('step ')[:30]!r}")
    panel.click_button("pause")
    time.sleep(1)
    panel.set_slider(0, 20.0)
    time.sleep(2)
    report.check("the step slider scrubs", panel.find("step ").startswith("step 21 of"),
                 panel.find("step ")[:40])
    twenty = panel.digest(str(shots / "11-step21.png"))
    panel.set_slider(0, 5.0)
    time.sleep(2)
    report.check("scrubbing redraws the arm", twenty != panel.digest(str(shots / "11-step6.png")))


SCENARIOS: Dict[str, Callable[[Panel, Report, Path], None]] = {
    "07": _drive_07,
    "08": _drive_08,
    "09": _drive_09,
    "10": _drive_10,
    "11": _drive_11,
}


def run(example: str, extra: List[str], *, timeout: float, shots: Path) -> int:
    """Run one example with a browser attached and report on its interactions.

    Args:
        example: The example's number, a key of :data:`EXAMPLES`.
        extra: The example's own command-line arguments.
        timeout: How long to wait for the side panel to fill, in seconds.  Example
            ``08`` needs a generous one when it was not given a small ``--markers``.
        shots: Directory for the screenshots, which are the artefact to look at when
            a check fails.

    Returns:
        The number of failed checks.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:  # pragma: no cover - the documented install step
        raise SystemExit(
            "this script needs Playwright: pip install playwright && playwright install chromium"
        ) from error

    shots.mkdir(parents=True, exist_ok=True)
    seen = threading.Event()
    captured: Dict[str, str] = {}

    def spy(url: str, *_args: object, **_kwargs: object) -> bool:
        captured["url"] = url
        seen.set()
        return True

    webbrowser.open = webbrowser.open_new = webbrowser.open_new_tab = spy  # type: ignore[assignment]

    def scene() -> None:
        """The example itself, in a thread: ``env.launch`` blocks on the handshake.

        The one exception that is expected is Swift reporting the browser gone as the
        report ends and this script closes the page; anything else is a failure of the
        example and is said out loud, because the thread's default behaviour is a
        traceback in the middle of the report.
        """
        import runpy

        sys.argv = [example, *extra]
        try:
            runpy.run_path(str(_REPO_ROOT / EXAMPLES[example]), run_name="__main__")
        except TimeoutError as error:
            if "stopped responding" not in str(error):
                raise
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            print(f"the example raised {type(error).__name__}: {error}", flush=True)

    threading.Thread(target=scene, daemon=True).start()
    if not seen.wait(timeout=timeout):
        print(f"the example never asked for a browser within {timeout:.0f} s")
        return 1
    print(f"example {example} {' '.join(extra)}".rstrip())
    print(f"url {captured['url']}")

    report = Report()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--no-sandbox", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
        )
        page = browser.new_page(viewport={"width": 1200, "height": 950})
        errors: List[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(captured["url"], wait_until="load")
        panel = Panel(page)
        # The panel is filled after the page opens, and example 08 fills it slowly.
        if not panel.wait_for(EXPECT[example], timeout):
            report.check(
                "the side panel filled",
                False,
                f"no {EXPECT[example]!r} after {timeout:.0f} s",
            )
        else:
            # The readout is filled before the control beside it exists, so the text
            # arriving is not the same instant as the panel being usable.
            time.sleep(2.0)
            SCENARIOS[example](panel, report, shots)
        report.check("no page errors", not errors, "; ".join(errors[:3]))
        browser.close()
    return report.failures


def main(argv: Optional[List[str]] = None) -> int:
    """Run the script."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("example", choices=sorted(EXAMPLES), help="which example to drive")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the side panel, which a marker cloud can dominate",
    )
    parser.add_argument(
        "--shots",
        type=Path,
        default=Path("/tmp/franka-ik-browser"),
        help="where to write the screenshots a failing check points at",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="arguments for the example itself, e.g. --model skeleton; put this "
        "script's own flags before the example number, because everything after it "
        "goes to the example",
    )
    args = parser.parse_args(argv)
    extra = DEFAULT_EXTRA.get(args.example, []) + list(args.extra)
    failures = run(args.example, extra, timeout=args.timeout, shots=args.shots)
    print(f"{failures} failed check(s); screenshots in {args.shots}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
