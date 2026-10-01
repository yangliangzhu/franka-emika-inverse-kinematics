/**
 * Headless smoke test for the generated HTML reports.
 *
 * The pages under `demos/` are canvas applications; a browser is the only real
 * way to look at them.  This is the franka_ik variant of the checker: the payload
 * holds `window.FRANKA_IK_DATA` with one polyline per solution branch.  This script gets most of the way there without one: it
 * loads a generated page, extracts the embedded `window.FRANKA_IK_DATA` payload and
 * the renderer, then executes the renderer against a minimal DOM and canvas stub
 * so that runtime errors surface.  It also checks that the drawing stayed inside
 * the canvas and that the 2D chart used its full width.
 *
 * It is not part of the pytest suite because it needs Node.js, not Python.  It
 * has already earned its keep: it caught a radian/degree mix-up that put the
 * reachability series and its x axis on different scales, and an undefined
 * variable that made the 3D camera produce `NaN` coordinates.
 *
 * Usage:
 *   node scripts/check_demo.js demos/arm-angle.html
 *   for f in demos/*.html; do node scripts/check_demo.js "$f" || exit 1; done
 *
 * Exit code 0 means the page rendered without throwing and stayed in bounds.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const file = process.argv[2];
if (!file) {
  console.error("usage: node check_demo.js <page.html>");
  process.exit(2);
}
const html = fs.readFileSync(file, "utf8");

// --- extract the data payload -------------------------------------------------
const dataMatch = html.match(/window\.FRANKA_IK_DATA = (\{[\s\S]*?\});\n<\/script>/);
if (!dataMatch) {
  console.error("FAIL: could not find window.FRANKA_IK_DATA in " + file);
  process.exit(1);
}
const data = JSON.parse(dataMatch[1]);
console.log("data payload parsed: kind=" + data.kind +
  " arms=" + (data.arms ? data.arms.length : 0) +
  " panels=" + (data.panels ? data.panels.length : 0));

// --- extract the renderer -----------------------------------------------------
const scripts = [...html.matchAll(/<script>\n([\s\S]*?)\n<\/script>/g)].map((m) => m[1]);
const renderer = scripts.filter((s) => s.indexOf("FRANKA_IK_DATA || {}") >= 0).pop();
if (!renderer) {
  console.error("FAIL: renderer script not found");
  process.exit(1);
}

// --- minimal DOM stub ---------------------------------------------------------
const calls = { fillRect: 0, lineTo: 0, fillText: 0, stroke: 0, badCoords: 0, minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
let currentCanvas = "?";
const perCanvas = {};
function notePoint(x, y) {
  const bucket = perCanvas[currentCanvas] || (perCanvas[currentCanvas] = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
  if (Number.isFinite(x)) { bucket.minX = Math.min(bucket.minX, x); bucket.maxX = Math.max(bucket.maxX, x); }
  if (Number.isFinite(y)) { bucket.minY = Math.min(bucket.minY, y); bucket.maxY = Math.max(bucket.maxY, y); }
  if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e5 || Math.abs(y) > 1e5) {
    calls.badCoords += 1;
    if (calls.badCoords <= 2) {
      console.error("bad coordinate x=" + x + " y=" + y + " canvas=" + currentCanvas);
      console.error(new Error("trace").stack.split("\n").slice(2, 8).join("\n"));
    }
    return;
  }
  if (x < calls.minX) { calls.minX = x; }
  if (x > calls.maxX) { calls.maxX = x; }
  if (y < calls.minY) { calls.minY = y; }
  if (y > calls.maxY) { calls.maxY = y; }
}
function makeContext() {
  const noop = () => {};
  const ctx = {
    canvas: null,
    setTransform: noop, clearRect: noop, save: noop, restore: noop,
    beginPath: noop, closePath: noop,
    moveTo: (x, y) => notePoint(x, y), arc: (x, y) => notePoint(x, y), rect: noop,
    strokeRect: noop, setLineDash: noop, translate: noop, rotate: noop,
    fillText: (str, x, y) => { calls.fillText += 1; notePoint(x, y); },
    strokeText: noop,
    fillRect: (x, y, w, h) => { calls.fillRect += 1; notePoint(x, y); notePoint(x + w, y + h); },
    stroke: () => { calls.stroke += 1; },
    fill: noop,
    lineTo: (x, y) => { calls.lineTo += 1; notePoint(x, y); },
  };
  return ctx;
}

function makeElement(tag) {
  const element = {
    tagName: tag,
    style: {},
    children: [],
    innerHTML: "",
    textContent: "",
    value: "0",
    min: "0",
    max: "1",
    step: "1",
    className: "",
    clientWidth: 800,
    clientHeight: 400,
    width: 800,
    height: 400,
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    removeEventListener: () => {},
    appendChild(child) { this.children.push(child); return child; },
    querySelector() { return makeElement("div"); },
    getBoundingClientRect() { return { width: 800, height: 430, top: 0, left: 0 }; },
    id2: "",
    getContext() { if (!this._ctx) { this._ctx = makeContext(); this._ctx.canvas = this; } currentCanvas = this.id || this.tagName; return this._ctx; },
  };
  return element;
}

const elements = {};
const ids = ["scene", "chart", "legend", "table", "notes", "branches"];
ids.forEach((id) => { elements[id] = makeElement(id === "scene" || id === "chart" ? "canvas" : "div"); elements[id].id = id; });

const documentStub = {
  readyState: "complete",
  getElementById: (id) => elements[id] || null,
  createElement: (tag) => makeElement(tag),
  createTextNode: (value) => ({ nodeValue: String(value) }),
  addEventListener: () => {},
};

const windowStub = {
  devicePixelRatio: 2,
  innerWidth: 1400,
  addEventListener: () => {},
  FRANKA_IK_DATA: data,
};

const sandbox = {
  window: windowStub,
  document: documentStub,
  console: console,
  Math: Math,
  JSON: JSON,
  isFinite: isFinite,
  setTimeout: setTimeout,
  clearInterval: clearInterval,
  setInterval: setInterval,
};
sandbox.globalThis = sandbox;

try {
  vm.createContext(sandbox);
  vm.runInContext(renderer, sandbox, { filename: path.basename(file) + "::renderer" });
} catch (error) {
  console.error("FAIL: renderer threw -> " + error.stack);
  process.exit(1);
}

// --- assertions ---------------------------------------------------------------
const failures = [];
if (calls.stroke === 0) { failures.push("no stroke() call: the chart or scene drew nothing"); }
if (calls.fillText === 0) { failures.push("no fillText() call: no labels were drawn"); }
if (elements.notes.innerHTML === "") { failures.push("notes were not rendered"); }
if (!data.panels || data.panels.length === 0) {
  failures.push("no chart panels in the payload");
}
// every polyline of every arm must be a full chain of finite 3D points
if (data.arms && data.arms.length) {
  data.arms.forEach((arm, index) => {
    if (!arm.points || arm.points.length < 8) {
      failures.push("arm " + index + " has only " + (arm.points ? arm.points.length : 0) + " points");
    }
    (arm.points || []).forEach((point) => {
      if (point.length !== 3 || point.some((v) => !Number.isFinite(v))) {
        failures.push("arm " + index + " has a non-finite point");
      }
    });
  });
}
if (data.kind === "branches" && (!data.arms || data.arms.length === 0)) {
  failures.push("the branches page carries no arms");
}
if (Object.prototype.hasOwnProperty.call(data, "table") && elements.table.innerHTML === "") {
  failures.push("table was not rendered");
}
if (calls.badCoords > 0) {
  failures.push(calls.badCoords + " drawing calls received non-finite or absurd coordinates");
}
// The 2D chart must stay inside its canvas (a small margin is allowed for text
// anchors).  The 3D scene may legitimately project geometry outside the canvas,
// because a perspective projection has no reason to fit -- the browser clips it.
Object.keys(perCanvas).forEach((name) => {
  const box = perCanvas[name];
  if (name !== "chart") { return; }
  if (box.minX < -20 || box.maxX > 820 || box.minY < -20 || box.maxY > 440) {
    failures.push("chart drawing escaped the canvas: x [" + box.minX.toFixed(1) + ", " +
      box.maxX.toFixed(1) + "], y [" + box.minY.toFixed(1) + ", " + box.maxY.toFixed(1) + "]");
  }
});
// The 2D chart must actually use most of its canvas, which catches a series
// drawn on a different scale than its axis.
Object.keys(perCanvas).forEach((name) => {
  if (name !== "chart") { return; }
  const box = perCanvas[name];
  if (box.maxX - box.minX < 200) {
    failures.push("chart compressed horizontally: width " + (box.maxX - box.minX).toFixed(1));
  }
});

if (failures.length) {
  console.error("per-canvas: " + JSON.stringify(perCanvas));
  console.error("FAIL " + file);
  failures.forEach((f) => console.error("  - " + f));
  process.exit(1);
}
console.log("per-canvas:", JSON.stringify(perCanvas));
console.log("OK " + path.basename(file) + "  (stroke=" + calls.stroke +
  ", fillText=" + calls.fillText + ", fillRect=" + calls.fillRect +
  ", x[" + calls.minX.toFixed(0) + "," + calls.maxX.toFixed(0) + "]" +
  ", y[" + calls.minY.toFixed(0) + "," + calls.maxY.toFixed(0) + "]" +
  ", notes=" + elements.notes.innerHTML.length + " chars)");
