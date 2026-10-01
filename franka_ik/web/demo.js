/* Interactive renderer for the franka_ik HTML reports.
 *
 * The Python side computes everything and embeds it as `window.FRANKA_IK_DATA`.
 * This file only draws it: a 3D wireframe scene that can show several arm
 * configurations at once (one per solution branch, which is the whole point of
 * this repository), a general multi-panel 2D chart, and a table.
 *
 * No external dependency, no CDN, no network: the pages must open from disk.
 *
 * Data schema (angles in degrees, positions in metres):
 *
 *   title        string
 *   kind         "index" | "branches" | "coverage" | "tracking"
 *   arms         [{label, color, points: [[x,y,z] x 8], qDeg[7],
 *                  poseError, withinLimits, recoversTarget, dim?}]
 *   target       {points, qDeg} | null       the configuration the pose came from
 *   tool         [x, y, z] | null            the target pose position
 *   panels       see below
 *   table        {headers, rows}
 *   notes        string[]
 *   toggleArms   boolean                     offer per-arm visibility checkboxes
 *
 * Panel schema:
 *   {title, series: [{label, values, color?, dashed?}], xValues?, xLabel?,
 *    markers: [{x, label, color}], bands: [[lo, hi]], fixedY?: [lo, hi]}
 */

(function () {
  "use strict";

  var DATA = window.FRANKA_IK_DATA || {};
  var COLORS = ["#08519c", "#cb181d", "#238b45", "#f16913", "#6a51a3", "#0c7489", "#8c510a"];
  var GRID = "#e2e2de";
  var AXIS = "#8a8a85";
  var INK = "#1c1c1c";
  var MUTED = "#6b6b6b";
  var FEASIBLE_FILL = "rgba(199, 233, 192, 0.5)";

  // ------------------------------------------------------------------ //
  // canvas helpers
  // ------------------------------------------------------------------ //

  function setupCanvas(canvas) {
    var ratio = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var width = Math.max(1, Math.round(rect.width));
    var height = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx: ctx, width: width, height: height };
  }

  function text(ctx, str, x, y, opts) {
    opts = opts || {};
    ctx.save();
    ctx.font = (opts.size || 11) + "px -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.fillStyle = opts.color || INK;
    ctx.textAlign = opts.align || "left";
    ctx.textBaseline = opts.baseline || "alphabetic";
    if (opts.rotate) {
      ctx.translate(x, y);
      ctx.rotate(opts.rotate);
      ctx.fillText(str, 0, 0);
    } else {
      ctx.fillText(str, x, y);
    }
    ctx.restore();
  }

  function niceTicks(min, max, count) {
    if (!(max > min)) { return [min]; }
    var span = max - min;
    var rawStep = span / Math.max(1, count);
    var magnitude = Math.pow(10, Math.floor(Math.log(rawStep) / Math.LN10));
    var candidates = [1, 2, 2.5, 5, 10];
    var step = magnitude * 10;
    for (var i = 0; i < candidates.length; i += 1) {
      if (magnitude * candidates[i] >= rawStep) { step = magnitude * candidates[i]; break; }
    }
    var ticks = [];
    var first = Math.ceil(min / step) * step;
    for (var v = first; v <= max + step * 1e-6; v += step) {
      ticks.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    }
    return ticks;
  }

  function isNum(value) {
    return typeof value === "number" && isFinite(value);
  }

  // ------------------------------------------------------------------ //
  // multi-panel chart
  // ------------------------------------------------------------------ //

  function drawChart(canvas, data) {
    var view = setupCanvas(canvas);
    var ctx = view.ctx;
    var panels = data.panels || [];
    if (!panels.length) { return; }

    var padLeft = 62;
    var padRight = 16;
    var padTop = 18;
    var padBottom = 46;
    var gap = 28;
    var plotWidth = view.width - padLeft - padRight;
    var plotHeight = (view.height - padTop - padBottom - gap * (panels.length - 1)) / panels.length;
    if (plotWidth < 40 || plotHeight < 30) { return; }

    panels.forEach(function (panel, index) {
      var top = padTop + index * (plotHeight + gap);
      var bottom = top + plotHeight;

      var xs = panel.xValues || [];
      var xMin = xs.length ? xs[0] : 0;
      var xMax = xs.length ? xs[xs.length - 1] : 1;
      var xTicks = niceTicks(xMin, xMax, 8);
      var tickStep = xTicks.length > 1 ? xTicks[1] - xTicks[0] : (xMax - xMin);
      function xOf(value) { return padLeft + ((value - xMin) / (xMax - xMin)) * plotWidth; }

      var yMin = Infinity;
      var yMax = -Infinity;
      if (panel.bars) {
        panel.bars.forEach(function (bar) { yMin = Math.min(yMin, 0, bar.y); yMax = Math.max(yMax, 0, bar.y); });
      }
      (panel.series || []).forEach(function (series) {
        series.values.forEach(function (value) {
          if (!isNum(value)) { return; }
          yMin = Math.min(yMin, value);
          yMax = Math.max(yMax, value);
        });
      });
      if (!isFinite(yMin) || !isFinite(yMax)) { yMin = -1; yMax = 1; }
      if (panel.fixedY) { yMin = panel.fixedY[0]; yMax = panel.fixedY[1]; }
      var span = yMax - yMin;
      if (span < 1e-12) { span = 1; }
      yMin -= span * 0.1;
      yMax += span * 0.1;
      function yOf(value) { return bottom - ((value - yMin) / (yMax - yMin)) * plotHeight; }

      ctx.fillStyle = "#f4f4f1";
      ctx.fillRect(padLeft, top, plotWidth, plotHeight);
      (panel.bands || []).forEach(function (band) {
        ctx.fillStyle = FEASIBLE_FILL;
        var left = xOf(band[0]);
        ctx.fillRect(left, top, xOf(band[1]) - left, plotHeight);
      });

      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      var yTicks = niceTicks(yMin, yMax, 4);
      ctx.beginPath();
      yTicks.forEach(function (value) {
        ctx.moveTo(padLeft, Math.round(yOf(value)) + 0.5);
        ctx.lineTo(padLeft + plotWidth, Math.round(yOf(value)) + 0.5);
      });
      ctx.stroke();
      yTicks.forEach(function (value) {
        text(ctx, value.toFixed(Math.abs(yMax - yMin) < 4 ? 2 : 0), padLeft - 6, yOf(value) + 3,
             { align: "right", size: 10, color: MUTED });
      });

      ctx.strokeStyle = "#c8c8c2";
      ctx.beginPath();
      ctx.moveTo(padLeft, yOf(0) + 0.5);
      ctx.lineTo(padLeft + plotWidth, yOf(0) + 0.5);
      ctx.stroke();

      // optional bar series, for the coverage and histogram panels
      (panel.bars || []).forEach(function (bar, barIndex) {
        var half = 0.5 * (panel.barWidth || 0.6);
        // clamp: a bar centred on the first or last sample would otherwise
        // stick out of the plot area by half its width
        var left = Math.max(padLeft, Math.min(xOf(bar.x - half), padLeft + plotWidth));
        var right = Math.max(padLeft, Math.min(xOf(bar.x + half), padLeft + plotWidth));
        var topY = yOf(bar.y);
        var zeroY = yOf(0);
        ctx.fillStyle = bar.color || COLORS[barIndex % COLORS.length];
        ctx.fillRect(left, Math.min(topY, zeroY), right - left, Math.abs(zeroY - topY));
        if (bar.label) {
          text(ctx, bar.label, 0.5 * (left + right), topY - 4,
               { align: "center", size: 10, color: INK });
        }
      });

      (panel.series || []).forEach(function (series, seriesIndex) {
        ctx.strokeStyle = series.color || COLORS[seriesIndex % COLORS.length];
        ctx.lineWidth = series.width || 1.8;
        if (series.dashed) { ctx.setLineDash([6, 4]); }
        ctx.beginPath();
        var started = false;
        xs.forEach(function (x, i) {
          var value = series.values[i];
          if (!isNum(value)) { started = false; return; }
          var px = xOf(x);
          var py = yOf(value);
          if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });

      (panel.markers || []).forEach(function (marker, markerIndex) {
        // a marker can be off scale, e.g. tan(theta4/2) as theta4 approaches
        // 180 degrees; pin it to the edge instead of drawing outside the canvas
        var raw = xOf(marker.x);
        var pinned = Math.max(padLeft + 2, Math.min(raw, padLeft + plotWidth - 2));
        ctx.strokeStyle = marker.color || "#08519c";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(pinned, top);
        ctx.lineTo(pinned, bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        if (marker.label) {
          var above = raw < padLeft + plotWidth * 0.5;
          text(ctx, marker.label,
               pinned + (above ? 4 : -4),
               top + 12 + 13 * markerIndex,
               { size: 10, color: marker.color, align: above ? "left" : "right" });
        }
      });

      ctx.strokeStyle = AXIS;
      ctx.lineWidth = 1;
      ctx.strokeRect(padLeft + 0.5, top + 0.5, plotWidth - 1, plotHeight - 1);
      text(ctx, panel.title || "", padLeft + 6, top + 14, { size: 11 });
      if (panel.yLabel) {
        text(ctx, panel.yLabel, 12, top + plotHeight / 2,
             { size: 10, color: MUTED, rotate: -Math.PI / 2 });
      }

      if (index === panels.length - 1) {
        ctx.strokeStyle = AXIS;
        ctx.beginPath();
        xTicks.forEach(function (value) {
          ctx.moveTo(xOf(value), bottom);
          ctx.lineTo(xOf(value), bottom + 4);
        });
        ctx.stroke();
        xTicks.forEach(function (value) {
          var decimals = tickStep < 0.1 ? 2 : (tickStep < 1 ? 1 : 0);
          text(ctx, value.toFixed(decimals), xOf(value), bottom + 17,
               { align: "center", size: 10, color: MUTED });
        });
        text(ctx, panel.xLabel || "", padLeft + plotWidth / 2, bottom + 36,
             { align: "center", size: 11, color: MUTED });
      }
    });

    drawLegend(panels);
  }

  function drawLegend(panels) {
    var host = document.getElementById("legend");
    if (!host) { return; }
    host.innerHTML = "";
    var seen = {};
    panels.forEach(function (panel) {
      (panel.series || []).forEach(function (series, seriesIndex) {
        if (!series.label || seen[series.label]) { return; }
        seen[series.label] = true;
        var span = document.createElement("span");
        var swatch = document.createElement("i");
        swatch.style.background = series.color || COLORS[seriesIndex % COLORS.length];
        span.appendChild(swatch);
        span.appendChild(document.createTextNode(series.label));
        host.appendChild(span);
      });
    });
  }

  // ------------------------------------------------------------------ //
  // 3D scene: several arm configurations at once
  // ------------------------------------------------------------------ //

  function Scene(canvas) {
    this.canvas = canvas;
    this.yaw = -0.75;
    this.pitch = 0.30;
    this.zoom = 1.0;
    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.visible = {};
    this.bounds = null;
    this.hidden = {};
    this.attach();
  }

  Scene.prototype.attach = function () {
    var self = this;
    this.canvas.addEventListener("mousedown", function (event) {
      self.dragging = true;
      self.lastX = event.clientX;
      self.lastY = event.clientY;
    });
    window.addEventListener("mouseup", function () { self.dragging = false; });
    window.addEventListener("mousemove", function (event) {
      if (!self.dragging) { return; }
      self.yaw += (event.clientX - self.lastX) * 0.01;
      self.pitch += (event.clientY - self.lastY) * 0.01;
      self.pitch = Math.max(-1.45, Math.min(1.45, self.pitch));
      self.lastX = event.clientX;
      self.lastY = event.clientY;
      self.render();
    });
    this.canvas.addEventListener("wheel", function (event) {
      event.preventDefault();
      self.zoom *= event.deltaY < 0 ? 1.08 : 1 / 1.08;
      self.zoom = Math.max(0.35, Math.min(4, self.zoom));
      self.render();
    }, { passive: false });
    window.addEventListener("resize", function () { self.render(); });
  };

  Scene.prototype.setHidden = function (index, hidden) {
    this.hidden[index] = hidden;
    this.render();
  };

  /** Bounding box and auto-fit scale over everything that may be drawn. */
  Scene.prototype.fit = function (view) {
    if (!this.bounds) {
      var min = [Infinity, Infinity, Infinity];
      var max = [-Infinity, -Infinity, -Infinity];
      var collections = (DATA.arms || []).map(function (arm) { return arm.points; });
      if (DATA.target) { collections.push(DATA.target.points); }
      collections.forEach(function (frames) {
        (frames || []).forEach(function (point) {
          for (var axis = 0; axis < 3; axis += 1) {
            if (!isNum(point[axis])) { continue; }
            if (point[axis] < min[axis]) { min[axis] = point[axis]; }
            if (point[axis] > max[axis]) { max[axis] = point[axis]; }
          }
        });
      });
      var centre = [0, 0, 0];
      var extent = 0.2;
      for (var axis2 = 0; axis2 < 3; axis2 += 1) {
        if (!isFinite(min[axis2]) || !isFinite(max[axis2])) { min[axis2] = -0.5; max[axis2] = 0.5; }
        centre[axis2] = 0.5 * (min[axis2] + max[axis2]);
        extent = Math.max(extent, max[axis2] - min[axis2]);
      }
      this.bounds = { centre: centre, extent: extent };
    }
    var span = Math.min(view.width, view.height) * 0.62;
    return {
      centre: this.bounds.centre,
      extent: this.bounds.extent,
      scale: (span / this.bounds.extent) * this.zoom,
    };
  };

  Scene.prototype.project = function (point, view, centre, scale) {
    var cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    var cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    var x = point[0] - centre[0];
    var y = point[1] - centre[1];
    var z = point[2] - centre[2];
    var x1 = x * cy + y * sy;
    var y1 = -x * sy + y * cy;
    var y2 = y1 * cp - z * sp;
    var z2 = y1 * sp + z * cp;
    return [view.width / 2 + x1 * scale, view.height * 0.58 - z2 * scale, y2];
  };

  Scene.prototype.render = function () {
    var view = setupCanvas(this.canvas);
    var ctx = view.ctx;
    var fit = this.fit(view);
    var centre = fit.centre;
    var scale = fit.scale;

    // ground grid at z = 0
    var gridHalf = Math.max(0.4, fit.extent * 0.6);
    ctx.strokeStyle = "#e6e6e0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var g = -gridHalf; g <= gridHalf * 1.001; g += gridHalf / 4) {
      var a = this.project([g, -gridHalf, 0], view, centre, scale);
      var b = this.project([g, gridHalf, 0], view, centre, scale);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      var c = this.project([-gridHalf, g, 0], view, centre, scale);
      var d = this.project([gridHalf, g, 0], view, centre, scale);
      ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]);
    }
    ctx.stroke();

    var self = this;

    function drawChain(points, color, width, alpha, dash) {
      var projected = points.map(function (point) {
        return self.project(point, view, centre, scale);
      });
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.lineJoin = "round";
      if (dash) { ctx.setLineDash(dash); }
      ctx.beginPath();
      ctx.moveTo(projected[0][0], projected[0][1]);
      for (var i = 1; i < projected.length; i += 1) {
        ctx.lineTo(projected[i][0], projected[i][1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      projected.forEach(function (point, index) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(point[0], point[1], index === 0 ? 4 : 3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }

    // the configuration the pose came from, drawn first as a reference
    if (DATA.target) {
      drawChain(DATA.target.points, "#111111", 1.4, 0.85, [6, 5]);
    }

    // one polyline per branch
    (DATA.arms || []).forEach(function (arm, index) {
      if (self.hidden[index]) { return; }
      drawChain(arm.points, arm.color || COLORS[index % COLORS.length],
                arm.recoversTarget ? 3.4 : 2.0,
                arm.recoversTarget ? 1.0 : 0.62, null);
    });

    // the target tool position
    if (DATA.tool) {
      var tip = this.project(DATA.tool, view, centre, scale);
      ctx.fillStyle = "#000000";
      ctx.beginPath();
      ctx.arc(tip[0], tip[1], 5, 0, Math.PI * 2);
      ctx.fill();
      text(ctx, "target pose", tip[0] + 8, tip[1] - 6, { size: 10, color: "#111111" });
    }

    // world axes at the origin
    var origin = this.project([0, 0, 0], view, centre, scale);
    var axisLength = Math.max(0.08, fit.extent * 0.2);
    [["x", "#cb181d", [axisLength, 0, 0]], ["y", "#238b45", [0, axisLength, 0]],
     ["z", "#08519c", [0, 0, axisLength]]].forEach(function (axis) {
      var tip2 = self.project(axis[2], view, centre, scale);
      ctx.strokeStyle = axis[1];
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(origin[0], origin[1]);
      ctx.lineTo(tip2[0], tip2[1]);
      ctx.stroke();
      text(ctx, axis[0], tip2[0] + 3, tip2[1], { size: 10, color: axis[1] });
    });

    text(ctx, "drag to orbit, wheel to zoom", 10, view.height - 10, { size: 10, color: MUTED });
  };

  // ------------------------------------------------------------------ //
  // page wiring
  // ------------------------------------------------------------------ //

  function buildTable(host, table) {
    if (!table || !table.rows || !table.rows.length) { return; }
    var html = "<table><thead><tr>";
    table.headers.forEach(function (header) { html += "<th>" + header + "</th>"; });
    html += "</tr></thead><tbody>";
    table.rows.forEach(function (row) {
      html += "<tr>";
      row.forEach(function (cell) {
        var cls = "";
        if (typeof cell === "string" && cell.indexOf("!") === 0) { cls = ' class="bad"'; cell = cell.slice(1); }
        html += "<td" + cls + ">" + cell + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    host.innerHTML = html;
  }

  function buildNotes(host, notes) {
    if (!notes || !notes.length) { return; }
    host.innerHTML = notes.map(function (note) {
      var ok = note.indexOf("OK:") === 0;
      return '<div class="note' + (ok ? " ok" : "") + '">' + note.replace(/^OK:\s*/, "") + "</div>";
    }).join("");
  }

  function buildBranchToggles(host, scene) {
    if (!host || !DATA.toggleArms || !DATA.arms) { return; }
    host.innerHTML = "";
    DATA.arms.forEach(function (arm, index) {
      var label = document.createElement("label");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = true;
      box.addEventListener("change", function () { scene.setHidden(index, !box.checked); });
      var swatch = document.createElement("i");
      swatch.style.background = arm.color || COLORS[index % COLORS.length];
      label.appendChild(box);
      label.appendChild(swatch);
      label.appendChild(document.createTextNode(
        arm.label + (arm.recoversTarget ? "  *target*" : "")
      ));
      host.appendChild(label);
    });
  }

  function main() {
    var sceneCanvas = document.getElementById("scene");
    var chartCanvas = document.getElementById("chart");
    var scene = sceneCanvas ? new Scene(sceneCanvas) : null;
    if (scene) { scene.render(); }
    if (chartCanvas) { drawChart(chartCanvas, DATA); }
    buildTable(document.getElementById("table"), DATA.table);
    buildNotes(document.getElementById("notes"), DATA.notes);
    buildBranchToggles(document.getElementById("branches"), scene);

    window.addEventListener("resize", function () {
      if (scene) { scene.render(); }
      if (chartCanvas) { drawChart(chartCanvas, DATA); }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
