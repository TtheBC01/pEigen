// Interactive sparse Cholesky benchmark chart: one problem x one metric at a time, as horizontal
// bars sorted fastest first. pEigen bars use the accent series color; other libraries are neutral.
// Data: window.PEIGEN_CHOLESKY_BENCH (assets/bench-cholesky-data.js). Plain script, no build step.
(function () {
  "use strict";

  var METRICS = [
    { id: "factor", label: "Factor", title: "Factorization time (analyze + numeric)", unit: "ms", better: "lower" },
    { id: "refactor", label: "Refactor", title: "Refactorization time (same pattern, new values)", unit: "ms", better: "lower" },
    { id: "solve1", label: "Solve, 1 RHS", title: "Solve time, one right-hand side", unit: "ms", better: "lower" },
    { id: "solve32", label: "Solve, 32 RHS", title: "Solve time, 32 right-hand sides", unit: "ms", better: "lower" },
    { id: "fill", label: "Fill-in", title: "Stored nonzeros in the factor (L, or L+U for LU)", unit: "nnz", better: "lower" }
  ];

  var SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function fmt(value, unit) {
    if (unit === "nnz") {
      if (value >= 1e6) return (value / 1e6).toFixed(value >= 1e7 ? 1 : 2) + "M";
      if (value >= 1e3) return Math.round(value / 1e3) + "k";
      return String(Math.round(value));
    }
    if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 1 : 2) + " s";
    if (value >= 100) return Math.round(value) + " ms";
    if (value >= 10) return value.toFixed(1) + " ms";
    return value.toFixed(2) + " ms";
  }

  function tickFmt(value, unit) {
    var trim = function (x) {
      return String(parseFloat(x.toFixed(2)));
    };
    if (unit === "nnz") return value >= 1e6 ? trim(value / 1e6) + "M" : value >= 1e3 ? trim(value / 1e3) + "k" : trim(value);
    return value >= 1000 ? trim(value / 1000) + " s" : trim(value) + " ms";
  }

  // Round tick step (1, 2, 2.5 or 5 times a power of ten) giving at most `count` intervals.
  function niceStep(max, count) {
    var raw = max / count;
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var steps = [1, 2, 2.5, 5, 10];
    for (var i = 0; i < steps.length; i++) if (steps[i] * p >= raw) return steps[i] * p;
    return 10 * p;
  }

  function init() {
    var root = document.getElementById("chol-viz");
    var data = window.PEIGEN_CHOLESKY_BENCH;
    if (!root || !data) return;

    var state = { metric: "factor", problem: "3D Laplacian 40^3", scale: "linear" };
    var labels = {};
    data.solvers.forEach(function (s) {
      labels[s.id] = s;
    });

    // ----- controls -----
    var controls = root.querySelector(".viz-controls");
    var seg = document.createElement("div");
    seg.className = "segmented";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Metric");
    METRICS.forEach(function (m) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = m.label;
      b.dataset.metric = m.id;
      b.addEventListener("click", function () {
        state.metric = m.id;
        render();
      });
      seg.appendChild(b);
    });
    controls.appendChild(seg);

    var lab = document.createElement("label");
    lab.textContent = "Problem";
    var select = document.createElement("select");
    var families = {};
    data.problems.forEach(function (p) {
      var fam = p.name.replace(/\s+\S+$/, "");
      if (!families[fam]) {
        families[fam] = document.createElement("optgroup");
        families[fam].label = fam;
        select.appendChild(families[fam]);
      }
      var o = document.createElement("option");
      o.value = p.name;
      o.textContent = p.name.replace("^2", "²").replace("^3", "³") + " (n = " + p.n.toLocaleString("en-US") + ")";
      families[fam].appendChild(o);
    });
    select.value = state.problem;
    select.addEventListener("change", function () {
      state.problem = select.value;
      render();
    });
    lab.appendChild(select);
    controls.appendChild(lab);

    var scaleSeg = document.createElement("div");
    scaleSeg.className = "segmented";
    scaleSeg.setAttribute("role", "group");
    scaleSeg.setAttribute("aria-label", "Axis scale");
    [["linear", "Linear"], ["log", "Log scale"]].forEach(function (pair) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = pair[1];
      b.dataset.scale = pair[0];
      b.addEventListener("click", function () {
        state.scale = pair[0];
        render();
      });
      scaleSeg.appendChild(b);
    });
    controls.appendChild(scaleSeg);

    var titleEl = root.querySelector(".viz-title");
    var subEl = root.querySelector(".viz-sub");
    var chart = root.querySelector(".chart");
    var tip = document.createElement("div");
    tip.className = "tooltip";
    tip.hidden = true;
    tip.setAttribute("role", "status");
    chart.appendChild(tip);
    var tableHost = root.querySelector(".table-host");

    function prettyProblem(name) {
      return name.replace("^2", "²").replace("^3", "³");
    }

    function render() {
      var metric = METRICS.filter(function (m) {
        return m.id === state.metric;
      })[0];
      seg.querySelectorAll("button").forEach(function (b) {
        b.setAttribute("aria-pressed", b.dataset.metric === state.metric ? "true" : "false");
      });
      scaleSeg.querySelectorAll("button").forEach(function (b) {
        b.setAttribute("aria-pressed", b.dataset.scale === state.scale ? "true" : "false");
      });
      var logScale = state.scale === "log";

      var values = data.metrics[state.metric][state.problem] || {};
      var rows = Object.keys(values)
        .map(function (id) {
          return { id: id, label: labels[id].label, peigen: labels[id].peigen, value: values[id] };
        })
        .sort(function (a, b) {
          return a.value - b.value;
        });
      var best = rows.length ? rows[0].value : 1;
      var ldlt = values["peigen ldlt"];

      var prob = data.problems.filter(function (p) {
        return p.name === state.problem;
      })[0];
      titleEl.textContent = metric.title + ": " + prettyProblem(state.problem);
      subEl.textContent =
        "n = " +
        prob.n.toLocaleString("en-US") +
        ", nnz(A) = " +
        prob.nnz.toLocaleString("en-US") +
        ". Lower is better; sorted fastest first." +
        (state.metric === "refactor" ? " SuperLU and SparseLU have no refactorize path, so they are omitted." : "");

      // ----- svg -----
      var old = chart.querySelector("svg");
      if (old) chart.removeChild(old);
      var width = Math.max(320, chart.clientWidth || 640);
      var narrow = width < 560;
      var labelW = narrow ? 0 : 230;
      var rowH = narrow ? 46 : 30;
      var barH = narrow ? 14 : 16;
      var top = 22;
      var valueW = 64;
      var plotW = width - labelW - valueW - 8;
      var height = top + rows.length * rowH + 4;
      var lo, hi, ticks, xOf;
      if (logScale) {
        lo = Math.pow(10, Math.floor(Math.log10(rows[0].value)));
        hi = Math.pow(10, Math.ceil(Math.log10(rows[rows.length - 1].value)));
        if (hi === lo) hi = lo * 10;
        ticks = [];
        for (var p = lo; p <= hi * 1.0001; p *= 10) ticks.push(p);
        xOf = function (v) {
          return labelW + (plotW * (Math.log10(v) - Math.log10(lo))) / (Math.log10(hi) - Math.log10(lo));
        };
      } else {
        var step = niceStep(rows[rows.length - 1].value, 5);
        var nTicks = Math.ceil(rows[rows.length - 1].value / step);
        lo = 0;
        hi = step * nTicks;
        ticks = [];
        for (var k = 0; k <= nTicks; k++) ticks.push(step * k);
        xOf = function (v) {
          return labelW + (plotW * v) / hi;
        };
      }

      var svg = el("svg", {
        viewBox: "0 0 " + width + " " + height,
        width: width,
        height: height,
        role: "img",
        "aria-label": metric.title + " for " + prettyProblem(state.problem) + ", by solver" + (logScale ? " (log scale)" : "")
      });

      ticks.forEach(function (v, t) {
        var x = xOf(v);
        el("line", { class: t === 0 ? "baseline" : "gridline", x1: x, x2: x, y1: top - 6, y2: height - 4 }, svg);
        var anchor = t === 0 ? "start" : t === ticks.length - 1 ? "end" : "middle";
        var tick = el("text", { class: "tick", x: x, y: 12, "text-anchor": anchor }, svg);
        tick.textContent = v === 0 ? "0" : tickFmt(v, metric.unit);
      });

      rows.forEach(function (r, i) {
        var y0 = top + i * rowH;
        var g = el("g", { class: "row", tabindex: "0", role: "listitem" }, svg);
        var barY = narrow ? y0 + 20 : y0 + (rowH - barH) / 2;
        if (narrow) {
          el("text", { class: "bar-label", x: 0, y: y0 + 13 }, g).textContent = r.label;
        } else {
          el("text", { class: "bar-label", x: labelW - 10, y: barY + barH / 2 + 4, "text-anchor": "end" }, g).textContent =
            r.label;
        }
        var x0 = labelW;
        var xv = xOf(r.value);
        var w = Math.max(2, xv - x0);
        var fill = r.peigen ? "var(--series-peigen)" : "var(--series-other)";
        if (logScale) {
          el("line", { x1: x0, x2: xv, y1: barY + barH / 2, y2: barY + barH / 2, stroke: fill, "stroke-width": 2 }, g);
          el("circle", { class: "bar", cx: xv, cy: barY + barH / 2, r: 5, fill: fill, stroke: "var(--surface)", "stroke-width": 2 }, g);
          w += 6;
        } else {
          // bar with 4px rounding on the data end only (square at the baseline)
          var rad = Math.min(4, w / 2);
          var d =
            "M" + x0 + "," + barY +
            "H" + (x0 + w - rad) +
            "Q" + (x0 + w) + "," + barY + " " + (x0 + w) + "," + (barY + rad) +
            "V" + (barY + barH - rad) +
            "Q" + (x0 + w) + "," + (barY + barH) + " " + (x0 + w - rad) + "," + (barY + barH) +
            "H" + x0 + "Z";
          el("path", { class: "bar", d: d, fill: fill }, g);
        }
        var vl = el("text", { class: "value-label", x: x0 + w + 6, y: barY + barH / 2 + 4 }, g);
        vl.textContent = fmt(r.value, metric.unit);
        var hit = el("rect", { class: "hit", x: 0, y: y0, width: width, height: rowH }, g);

        var ratioBest = r.value / best;
        var tipHtml =
          "<strong>" + r.label + "</strong><br>" +
          fmt(r.value, metric.unit) +
          (i === 0 ? " (fastest)" : " (" + ratioBest.toFixed(ratioBest < 10 ? 2 : 1) + "× the fastest)") +
          (ldlt && r.id !== "peigen ldlt"
            ? "<br>vs pEigen ldlt: " + (ldlt / r.value >= 1 ? (ldlt / r.value).toFixed(2) + "× faster" : (r.value / ldlt).toFixed(2) + "× slower")
            : "");
        function show(evt) {
          tip.innerHTML = tipHtml;
          tip.hidden = false;
          var box = chart.getBoundingClientRect();
          var px = evt && evt.clientX ? evt.clientX - box.left : x0 + w;
          var py = (barY / height) * chart.clientHeight;
          var tw = tip.offsetWidth;
          tip.style.left = Math.min(Math.max(0, px + 12), chart.clientWidth - tw) + "px";
          tip.style.top = Math.max(0, py - tip.offsetHeight - 6) + "px";
        }
        function hide() {
          tip.hidden = true;
        }
        hit.addEventListener("mousemove", show);
        hit.addEventListener("mouseleave", hide);
        g.addEventListener("focus", function () {
          show(null);
        });
        g.addEventListener("blur", hide);
        g.setAttribute("aria-label", r.label + ": " + fmt(r.value, metric.unit));
      });
      chart.insertBefore(svg, tip);
      renderTable(metric);
    }

    function renderTable(metric) {
      var all = data.metrics[metric.id];
      var ids = data.solvers
        .map(function (s) {
          return s.id;
        })
        .filter(function (id) {
          return Object.keys(all).some(function (p) {
            return id in all[p];
          });
        });
      var html = '<div class="table-wrap"><table><thead><tr><th>Problem</th>';
      ids.forEach(function (id) {
        html += '<th class="num">' + labels[id].label + "</th>";
      });
      html += "</tr></thead><tbody>";
      data.problems.forEach(function (p) {
        var row = all[p.name] || {};
        var best = Math.min.apply(
          null,
          ids
            .filter(function (id) {
              return id in row;
            })
            .map(function (id) {
              return row[id];
            })
        );
        html += "<tr><th>" + prettyProblem(p.name) + "</th>";
        ids.forEach(function (id) {
          var v = row[id];
          html += '<td class="num">' + (v === undefined ? "–" : (v === best ? "<strong>" + fmt(v, metric.unit) + "</strong>" : fmt(v, metric.unit))) + "</td>";
        });
        html += "</tr>";
      });
      html += "</tbody></table></div><p class=\"viz-sub\">Bold marks the fastest (or smallest) entry in each row.</p>";
      tableHost.innerHTML = html;
    }

    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(render, 120);
    });
    document.addEventListener("peigen:theme", render);
    render();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
