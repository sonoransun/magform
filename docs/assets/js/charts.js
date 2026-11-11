/* magform docs: a small dependency-free chart renderer.
 *
 * Every chart on the site is a JSON spec written by tools/build_docs.py from the Python models and
 * embedded next to its container (<div class="chart" data-chart><script type="application/json">).
 * Three forms: "xy" (lines, areas and points on linear or log axes), "dot" (one value or range per
 * labelled row on a shared axis) and "bars" (ranked horizontal bars from a zero baseline).
 * Each chart gets a legend (two or more series), a crosshair or per-row tooltip that also works
 * from the keyboard, and a data-table twin.  Colours come from CSS classes, so both themes work
 * without redrawing.
 */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function html(name, cls, parent, text) {
    var node = document.createElement(name);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function sup(n) { return String(n).split("").map(function (c) { return SUP[c] || c; }).join(""); }
  function trim(s) { return s.indexOf(".") >= 0 ? s.replace(/0+$/, "").replace(/\.$/, "") : s; }
  function fmt(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    var a = Math.abs(v);
    if (a === 0) return "0";
    var d = digits || 3;
    if (a >= 1e6 || a < 1e-3) {
      var e = Math.floor(Math.log10(a));
      var m = v / Math.pow(10, e);
      var ms = trim(m.toPrecision(Math.min(d, 3)));
      return (ms === "1" ? "" : ms + "×") + "10" + sup(e);
    }
    var n = Number(v.toPrecision(d));
    return a >= 10000 ? n.toLocaleString("en-US") : String(n);
  }
  function fmtTime(v) {
    if (v === null || v === undefined || !isFinite(v)) return "\u2013";
    var a = Math.abs(v);
    function unit(x, u) { return String(Number(x.toPrecision(3))) + " " + u; }
    if (a === 0) return "0";
    if (a < 1e-6) return unit(v * 1e9, "ns");
    if (a < 1e-3) return unit(v * 1e6, "\u00B5s");
    if (a < 1) return unit(v * 1e3, "ms");
    if (a < 120) return unit(v, "s");
    if (a < 7200) return unit(v / 60, "min");
    if (a < 172800) return unit(v / 3600, "h");
    return unit(v / 86400, "days");
  }
  function fmtOn(axis, v) {
    if (axis && axis.format === "time") return fmtTime(v);
    return fmt(v) + (axis && axis.unit ? " " + axis.unit : "");
  }
  function colourClass(item) {
    if (item && item.tone) return "c t-" + item.tone;
    if (item && item.slot !== undefined && item.slot !== null) return "c s" + item.slot;
    return "c s1";
  }
  var measurer = null;
  function textWidth(text, size) {
    if (!measurer) measurer = document.createElement("canvas").getContext("2d");
    measurer.font = (size || 11.5) + "px 'IBM Plex Sans', system-ui, sans-serif";
    return measurer.measureText(String(text)).width;
  }

  /* ---------------------------------------------------------------- scales and ticks */
  function niceStep(span, target) {
    var raw = span / Math.max(target, 1);
    var pow = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / pow;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * pow;
  }
  function makeScale(axis, lo, hi, p0, p1, target) {
    var log = axis.scale === "log";
    var a = log ? Math.log10(lo) : lo;
    var b = log ? Math.log10(hi) : hi;
    var s = { log: log, lo: lo, hi: hi };
    s.f = function (v) { var t = ((log ? Math.log10(v) : v) - a) / (b - a); return p0 + t * (p1 - p0); };
    s.inv = function (p) { var t = (p - p0) / (p1 - p0); var u = a + t * (b - a); return log ? Math.pow(10, u) : u; };
    var ticks = [];
    if (axis.ticks) {
      axis.ticks.forEach(function (t) {
        var v = typeof t === "number" ? t : t.v;
        if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) {
          ticks.push({ v: v, label: typeof t === "number" ? (axis.format === "time" ? fmtTime(v) : fmt(v)) : t.label });
        }
      });
    } else if (log) {
      var e0 = Math.ceil(a - 1e-9), e1 = Math.floor(b + 1e-9);
      var decades = b - a;
      var every = Math.max(1, Math.ceil((e1 - e0 + 1) / Math.max(target, 3)));
      var powers = lo < 1e-3 || hi > 1e5;  /* one label style per axis */
      for (var e = e0; e <= e1; e += 1) {
        if ((e - e0) % every !== 0) continue;
        var v = Math.pow(10, e);
        ticks.push({ v: v, label: axis.format === "time" ? fmtTime(v) : powers ? "10" + sup(e) : trim(v.toFixed(Math.max(0, -e))) });
        if (decades <= 2.2) [2, 5].forEach(function (m) { if (m * v <= hi * (1 + 1e-9)) ticks.push({ v: m * v, label: fmt(m * v) }); });
      }
    } else {
      var step = niceStep(hi - lo, target);
      var dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
      for (var t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step) {
        var tv = Math.abs(t) < step * 1e-9 ? 0 : t;
        ticks.push({ v: tv, label: Math.abs(tv) >= 10000 ? Number(tv.toFixed(dec)).toLocaleString("en-US") : tv.toFixed(dec) });
      }
    }
    s.ticks = ticks;
    return s;
  }
  function extent(values, axis) {
    var lo = Infinity, hi = -Infinity;
    values.forEach(function (v) { if (v !== null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } });
    if (axis.min !== undefined && axis.min !== null) lo = axis.min;
    if (axis.max !== undefined && axis.max !== null) hi = axis.max;
    if (!(hi > lo)) { hi = lo + 1; }
    return [lo, hi];
  }
  function interp(xs, ys, x, xlog, ylog) {
    var n = xs.length;
    if (n === 0 || x < Math.min(xs[0], xs[n - 1]) || x > Math.max(xs[0], xs[n - 1])) return null;
    for (var i = 1; i < n; i += 1) {
      var x0 = xs[i - 1], x1 = xs[i];
      if ((x >= x0 && x <= x1) || (x <= x0 && x >= x1)) {
        var y0 = ys[i - 1], y1 = ys[i];
        if (y0 === null || y1 === null) return null;
        if (x1 === x0) return y1;
        var t = xlog ? (Math.log(x) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (x - x0) / (x1 - x0);
        if (ylog && y0 > 0 && y1 > 0) return Math.exp(Math.log(y0) + t * (Math.log(y1) - Math.log(y0)));
        return y0 + t * (y1 - y0);
      }
    }
    return null;
  }

  /* ---------------------------------------------------------------- chart shell */
  function Chart(root) {
    this.root = root;
    var data = root.querySelector('script[type="application/json"]');
    this.spec = JSON.parse(data.textContent);
    this.width = 0;
    var spec = this.spec;

    var named = (spec.series || []).filter(function (s) { return s.name && s.legend !== false; });
    if (spec.type === "xy" && named.length >= 2) {
      var legend = html("ul", "chart__legend", root);
      named.forEach(function (s) {
        var li = html("li", null, legend);
        html("span", "key " + (s.kind === "points" ? "key--dot " : s.kind === "area" ? "key--box " : "") + colourClass(s), li);
        html("span", null, li, s.name);
      });
    }
    this.plot = html("div", "chart__plot", root);
    this.plot.setAttribute("tabindex", "0");
    this.plot.setAttribute("role", "img");
    this.plot.setAttribute("aria-label", spec.title + (spec.subtitle ? ". " + spec.subtitle : "") + ". Values are in the data table that follows.");
    this.tip = html("div", "tip", root);
    this.tip.hidden = true;

    var foot = html("div", "chart__foot", root);
    if (spec.note) html("p", "chart__note", foot, spec.note);
    if (spec.table) {
      var details = html("details", "chart__table", root);
      html("summary", null, details, "Data table");
      var scroll = html("div", "table-scroll", details);
      var table = html("table", null, scroll);
      var head = html("tr", null, html("thead", null, table));
      spec.table.columns.forEach(function (c) { html("th", null, head, c); });
      var body = html("tbody", null, table);
      spec.table.rows.forEach(function (r) {
        var tr = html("tr", null, body);
        r.forEach(function (cell) { html("td", null, tr, typeof cell === "number" ? fmt(cell, 4) : (cell === null ? "–" : cell)); });
      });
    }

    var self = this;
    var pending = null;
    var redraw = function () {
      var w = Math.floor(self.plot.clientWidth);
      if (w > 0 && Math.abs(w - self.width) > 3) { self.width = w; self.render(); }
    };
    if (window.ResizeObserver) {
      new ResizeObserver(function () { if (pending) cancelAnimationFrame(pending); pending = requestAnimationFrame(redraw); }).observe(this.plot);
    } else {
      window.addEventListener("resize", redraw);
    }
    redraw();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { self.width = 0; redraw(); });
  }

  /* swap in a new spec of the same form (used by the explorer); the legend and data table are not rebuilt */
  Chart.prototype.update = function (spec) {
    this.spec = spec;
    if (this.width > 0) this.render();
  };

  Chart.prototype.render = function () {
    this.plot.textContent = "";
    this.hideTip();
    if (this.spec.type === "dot" || this.spec.type === "bars") this.renderRows(); else this.renderXY();
  };

  Chart.prototype.showTip = function (px, py, head, rows, note) {
    var tip = this.tip;
    tip.textContent = "";
    if (head) html("div", "tip__head", tip, head);
    (rows || []).forEach(function (r) {
      var row = html("div", "tip__row", tip);
      html("span", "key " + (r.dot ? "key--dot " : "") + (r.cls || ""), row);
      html("span", "tip__name", row, r.name);
      html("span", "tip__val", row, r.value);
    });
    if (note) html("div", "tip__note", tip, note);
    tip.hidden = false;
    var box = this.root.getBoundingClientRect();
    var plotBox = this.plot.getBoundingClientRect();
    var left = plotBox.left - box.left + px + 14;
    var top = plotBox.top - box.top + py - 10;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    if (left + tw > box.width - 4) left = plotBox.left - box.left + px - tw - 14;
    if (left < 4) left = 4;
    if (top + th > plotBox.bottom - box.top) top = plotBox.bottom - box.top - th;
    if (top < 0) top = 0;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  };
  Chart.prototype.hideTip = function () { this.tip.hidden = true; };

  /* ---------------------------------------------------------------- xy charts */
  Chart.prototype.renderXY = function () {
    var spec = this.spec, self = this, W = this.width;
    var series = spec.series || [];
    var H = Math.round(Math.max(250, Math.min(420, W * (spec.aspect || 0.58))));
    var xs = [], ys = [];
    series.forEach(function (s) { xs = xs.concat(s.x); ys = ys.concat(s.y); if (s.y0) ys = ys.concat(s.y0); });
    (spec.marks || []).forEach(function (m) { xs.push(m.x); ys.push(m.y); });
    (spec.rules || []).forEach(function (r) { (r.axis === "y" ? ys : xs).push(r.at); });
    var xr = extent(xs, spec.x), yr = extent(ys, spec.y);
    if (spec.y.scale === "log" && (spec.y.min === undefined || spec.y.min === null)) yr[0] = yr[0] / 1.6;
    if (spec.y.scale === "log" && (spec.y.max === undefined || spec.y.max === null)) yr[1] = yr[1] * 1.6;
    if (spec.y.scale !== "log" && (spec.y.min === undefined || spec.y.min === null)) {
      if (yr[0] > 0 && yr[0] < 0.6 * yr[1]) yr[0] = 0;
      yr[1] += 0.06 * (yr[1] - yr[0]);
    }

    var endLabels = series.filter(function (s) { return s.end_label; });
    var right = 14;
    endLabels.forEach(function (s) { right = Math.max(right, textWidth(s.end_label === true ? s.name : s.end_label, 12) + 16); });
    right = Math.min(right, W * 0.3);
    var top = spec.y.label ? 26 : 12;
    var bottom = 44;
    var ySc0 = makeScale(spec.y, yr[0], yr[1], H - bottom, top, Math.max(3, Math.round((H - bottom - top) / (spec.y.scale === "log" ? 34 : 52))));
    var left = 10;
    ySc0.ticks.forEach(function (t) { left = Math.max(left, textWidth(t.label) + 12); });
    var x0 = left, x1 = W - right, y0 = H - bottom, y1 = top;
    var xSc = makeScale(spec.x, xr[0], xr[1], x0, x1, Math.max(3, Math.round((x1 - x0) / 84)));
    var ySc = ySc0;

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, "aria-hidden": "true" }, this.plot);
    /* labels of bands and rules are collected here and added after the series, so no line strikes through them */
    var late = [];
    function lateLabel(attrs, text) { late.push({ a: attrs, t: text }); }
    /* bands sit under everything */
    (spec.bands || []).forEach(function (b) {
      var cls = "band" + (b.tone ? " band--" + b.tone : "");
      if (b.axis === "y") {
        var ya = ySc.f(Math.min(Math.max(b.to, yr[0]), yr[1])), yb = ySc.f(Math.min(Math.max(b.from, yr[0]), yr[1]));
        el("rect", { x: x0, y: ya, width: x1 - x0, height: Math.max(0, yb - ya), "class": cls }, svg);
        if (b.label) lateLabel({ x: b.side === "left" ? x0 + 6 : x1 - 6, y: ya + 14, "text-anchor": b.side === "left" ? "start" : "end", "class": "dlabel dlabel--muted" }, b.label);
      } else {
        var xa = xSc.f(Math.min(Math.max(b.from, xr[0]), xr[1])), xb = xSc.f(Math.min(Math.max(b.to, xr[0]), xr[1]));
        el("rect", { x: xa, y: y1, width: Math.max(0, xb - xa), height: y0 - y1, "class": cls }, svg);
        if (b.label) lateLabel({ x: xa + 6, y: y1 + 14, "class": "dlabel dlabel--muted" }, b.label);
      }
    });
    /* grid, axes, ticks */
    ySc.ticks.forEach(function (t) {
      var y = Math.round(ySc.f(t.v)) + 0.5;
      el("line", { x1: x0, x2: x1, y1: y, y2: y, "class": "gridline" }, svg);
      el("text", { x: x0 - 8, y: y + 4, "text-anchor": "end" }, svg).textContent = t.label;
    });
    el("line", { x1: x0, x2: x1, y1: y0 + 0.5, y2: y0 + 0.5, "class": "axisline" }, svg);
    var lastTickEnd = -Infinity;
    xSc.ticks.forEach(function (t) {
      var x = Math.round(xSc.f(t.v)) + 0.5;
      el("line", { x1: x, x2: x, y1: y0, y2: y0 + 5, "class": "axisline" }, svg);
      var w = textWidth(t.label);
      if (x - w / 2 > lastTickEnd + 6) {
        el("text", { x: x, y: y0 + 18, "text-anchor": "middle" }, svg).textContent = t.label;
        lastTickEnd = x + w / 2;
      }
    });
    if (spec.x.label) el("text", { x: (x0 + x1) / 2, y: H - 8, "text-anchor": "middle", "class": "axis-title" }, svg).textContent = spec.x.label;
    if (spec.y.label) el("text", { x: 0, y: 13, "class": "axis-title" }, svg).textContent = spec.y.label;

    /* reference rules */
    (spec.rules || []).forEach(function (r) {
      if (r.axis === "y") {
        if (r.at < yr[0] || r.at > yr[1]) return;
        var y = Math.round(ySc.f(r.at)) + 0.5;
        el("line", { x1: x0, x2: x1, y1: y, y2: y, "class": "rule" }, svg);
        if (r.label) lateLabel({ x: r.side === "left" ? x0 + 6 : x1 - 6, y: y + (r.dy || -6), "text-anchor": r.side === "left" ? "start" : "end", "class": "dlabel dlabel--muted" }, r.label);
      } else {
        if (r.at < xr[0] || r.at > xr[1]) return;
        var x = Math.round(xSc.f(r.at)) + 0.5;
        el("line", { x1: x, x2: x, y1: y1, y2: y0, "class": "rule" }, svg);
        if (r.label) {
          var flip = x + textWidth(r.label, 12) + 10 > x1;
          lateLabel({ x: flip ? x - 6 : x + 6, y: y1 + (r.dy || 14), "text-anchor": flip ? "end" : "start", "class": "dlabel dlabel--muted" }, r.label);
        }
      }
    });

    /* series */
    function inRange(x, y) { return x !== null && y !== null && isFinite(x) && isFinite(y) && (!xSc.log || x > 0) && (!ySc.log || y > 0); }
    function clampY(v) { return Math.max(y1 - 2000, Math.min(y0 + 2000, v)); }
    function path(s, ykey) {
      var d = "", pen = false;
      for (var i = 0; i < s.x.length; i += 1) {
        var yv = s[ykey][i];
        if (!inRange(s.x[i], yv)) { pen = false; continue; }
        d += (pen ? "L" : "M") + xSc.f(s.x[i]).toFixed(1) + " " + clampY(ySc.f(yv)).toFixed(1);
        pen = true;
      }
      return d;
    }
    var clipId = "clip-" + Math.random().toString(36).slice(2);
    var clip = el("clipPath", { id: clipId }, el("defs", {}, svg));
    el("rect", { x: x0, y: y1 - 1, width: x1 - x0, height: y0 - y1 + 2 }, clip);
    var layer = el("g", { "clip-path": "url(#" + clipId + ")" }, svg);
    series.forEach(function (s) {
      var cls = colourClass(s);
      if (s.kind === "area") {
        var up = path(s, "y");
        if (up) {
          var base = "";
          if (s.y0) {
            for (var i = s.x.length - 1; i >= 0; i -= 1) if (inRange(s.x[i], s.y0[i])) base += "L" + xSc.f(s.x[i]).toFixed(1) + " " + clampY(ySc.f(s.y0[i])).toFixed(1);
          } else {
            base = "L" + xSc.f(s.x[s.x.length - 1]).toFixed(1) + " " + y0 + "L" + xSc.f(s.x[0]).toFixed(1) + " " + y0;
          }
          el("path", { d: up + base + "Z", "class": "ar " + cls }, layer);
          if (s.outline !== false) el("path", { d: up, "class": "ln " + cls }, layer);
        }
      } else if (s.kind === "points") {
        for (var k = 0; k < s.x.length; k += 1) if (inRange(s.x[k], s.y[k])) el("circle", { cx: xSc.f(s.x[k]), cy: ySc.f(s.y[k]), r: 4.5, "class": "pt " + cls }, layer);
      } else {
        el("path", { d: path(s, "y"), "class": "ln " + cls + (s.thin ? " ln--thin" : "") }, layer).style.strokeWidth = s.thin ? 1.25 : 2;
      }
    });
    late.forEach(function (l) { el("text", l.a, svg).textContent = l.t; });
    /* annotated marks (outside the clip so labels can breathe) */
    (spec.marks || []).forEach(function (m) {
      if (!inRange(m.x, m.y)) return;
      var cx = xSc.f(m.x), cy = ySc.f(m.y);
      el("circle", { cx: cx, cy: cy, r: 5, "class": "pt " + colourClass(m) }, svg);
      if (m.label) {
        var anchor = m.anchor || (cx + textWidth(m.label, 12) + 12 > x1 ? "end" : "start");
        el("text", { x: cx + (anchor === "end" ? -9 : 9) + (m.dx || 0), y: cy + 4 + (m.dy || 0), "text-anchor": anchor, "class": "dlabel" }, svg).textContent = m.label;
      }
    });
    /* selective direct labels at line ends (dropped if they would collide) */
    var ends = [];
    endLabels.forEach(function (s) {
      for (var i = s.x.length - 1; i >= 0; i -= 1) {
        if (inRange(s.x[i], s.y[i])) { ends.push({ s: s, x: xSc.f(s.x[i]), y: Math.max(y1 + 6, Math.min(y0 - 4, ySc.f(s.y[i]))) }); break; }
      }
    });
    ends.sort(function (a, b) { return a.y - b.y; });
    var clear = ends.every(function (e, i) { return i === 0 || e.y - ends[i - 1].y >= 14; });
    if (clear) ends.forEach(function (e) {
      el("text", { x: Math.min(e.x + 8, x1 + 8), y: e.y + 4, "class": "dlabel" }, svg).textContent = e.s.end_label === true ? e.s.name : e.s.end_label;
    });

    /* hover / keyboard layer */
    var cross = el("line", { y1: y1, y2: y0, "class": "crosshair" }, svg);
    cross.style.display = "none";
    var dots = el("g", {}, svg);
    var hit = el("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1, "class": "hit" }, svg);
    var lineSeries = series.filter(function (s) { return s.kind !== "points" && s.name; });
    var pointItems = [];
    series.forEach(function (s) {
      if (s.kind === "points") s.x.forEach(function (x, i) { if (inRange(x, s.y[i])) pointItems.push({ x: x, y: s.y[i], label: (s.labels && s.labels[i]) || s.name, item: s }); });
    });
    (spec.marks || []).forEach(function (m) { if (inRange(m.x, m.y)) pointItems.push({ x: m.x, y: m.y, label: m.tip || m.label, item: m }); });

    function probe(px, py) {
      px = Math.max(x0, Math.min(x1, px));
      var near = null, best = 26;
      pointItems.forEach(function (p) {
        var d = Math.hypot(xSc.f(p.x) - px, ySc.f(p.y) - (py === null ? ySc.f(p.y) : py));
        if (d < best) { best = d; near = p; }
      });
      dots.textContent = "";
      if (near && (lineSeries.length === 0 || best < 14)) {
        cross.style.display = "none";
        el("circle", { cx: xSc.f(near.x), cy: ySc.f(near.y), r: 7, "class": "pt " + colourClass(near.item) }, dots);
        self.showTip(xSc.f(near.x), ySc.f(near.y), near.label, [
          { name: spec.x.short || spec.x.label, value: fmtOn(spec.x, near.x), cls: "", dot: true },
          { name: spec.y.short || spec.y.label, value: fmtOn(spec.y, near.y), cls: "", dot: true }
        ].map(function (r) { r.cls = "c t-muted"; return r; }));
        return;
      }
      if (lineSeries.length === 0) { self.hideTip(); return; }
      var xv = xSc.inv(px);
      var rows = [];
      lineSeries.forEach(function (s) {
        var yv = interp(s.x, s.y, xv, xSc.log, ySc.log);
        if (yv === null) return;
        rows.push({ name: s.name, value: fmtOn(spec.y, yv), cls: colourClass(s) });
        if (inRange(xv, yv)) el("circle", { cx: px, cy: ySc.f(yv), r: 4.5, "class": "pt " + colourClass(s) }, dots);
      });
      cross.setAttribute("x1", Math.round(px) + 0.5);
      cross.setAttribute("x2", Math.round(px) + 0.5);
      cross.style.display = "";
      self.showTip(px, (y0 + y1) / 2 - 30, (spec.x.short || spec.x.label) + ": " + fmtOn(spec.x, xv), rows);
    }
    function leave() { cross.style.display = "none"; dots.textContent = ""; self.hideTip(); }
    function local(ev) { var b = svg.getBoundingClientRect(); return [(ev.clientX - b.left) * (W / b.width), (ev.clientY - b.top) * (H / b.height)]; }
    hit.addEventListener("pointermove", function (ev) { var p = local(ev); probe(p[0], p[1]); });
    hit.addEventListener("pointerdown", function (ev) { var p = local(ev); probe(p[0], p[1]); });
    hit.addEventListener("pointerleave", leave);
    var kx = (x0 + x1) / 2;
    this.plot.onkeydown = function (ev) {
      var step = (x1 - x0) / (ev.shiftKey ? 10 : 50);
      if (ev.key === "ArrowLeft") kx -= step; else if (ev.key === "ArrowRight") kx += step;
      else if (ev.key === "Home") kx = x0; else if (ev.key === "End") kx = x1;
      else if (ev.key === "Escape") { leave(); return; } else return;
      ev.preventDefault();
      kx = Math.max(x0, Math.min(x1, kx));
      probe(kx, null);
    };
    this.plot.onfocus = function () { if (self.plot.matches(":focus-visible")) probe(kx, null); };
    this.plot.onblur = leave;
  };

  /* ---------------------------------------------------------------- dot and bar rows */
  Chart.prototype.renderRows = function () {
    var spec = this.spec, self = this, W = this.width;
    var rows = spec.rows || [];
    var bars = spec.type === "bars";
    var stacked = W < 540;
    var rowH = stacked ? 44 : (spec.row_height || 30);
    var top = (spec.bands || []).some(function (b) { return b.label; }) || (spec.rules || []).some(function (r) { return r.label; }) ? 24 : 8;
    var bottom = 42;
    var H = top + rows.length * rowH + bottom;
    var labelW = 0;
    rows.forEach(function (r) { labelW = Math.max(labelW, textWidth(r.label, 12.5)); });
    var valueW = 0;
    rows.forEach(function (r) { valueW = Math.max(valueW, textWidth(r.text || fmt(r.value), 12)); });
    var x0 = stacked ? 4 : Math.min(labelW + 16, W * 0.42);
    var x1 = W - valueW - 22;
    var vals = [];
    rows.forEach(function (r) { vals.push(r.value); if (r.value_hi !== undefined && r.value_hi !== null) vals.push(r.value_hi); });
    (spec.bands || []).forEach(function (b) { vals.push(b.from); vals.push(b.to); });
    (spec.rules || []).forEach(function (r) { vals.push(r.at); });
    var xr = extent(vals, spec.x);
    if (bars && spec.x.scale !== "log") xr[0] = 0;
    if (spec.x.scale !== "log" && (spec.x.max === undefined || spec.x.max === null)) xr[1] += 0.04 * (xr[1] - xr[0]);
    var xSc = makeScale(spec.x, xr[0], xr[1], x0, x1, Math.max(3, Math.round((x1 - x0) / 84)));
    var yBase = top + rows.length * rowH;

    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, "aria-hidden": "true" }, this.plot);
    (spec.bands || []).forEach(function (b) {
      var xa = xSc.f(Math.max(b.from, xr[0])), xb = xSc.f(Math.min(b.to, xr[1]));
      el("rect", { x: xa, y: top - 2, width: Math.max(0, xb - xa), height: yBase - top + 2, "class": "band" + (b.tone ? " band--" + b.tone : "") }, svg);
      if (b.label) el("text", { x: (xa + xb) / 2, y: top - 8, "text-anchor": "middle", "class": "dlabel dlabel--muted" }, svg).textContent = b.label;
    });
    var lastEnd = -Infinity;
    xSc.ticks.forEach(function (t) {
      var x = Math.round(xSc.f(t.v)) + 0.5;
      el("line", { x1: x, x2: x, y1: top - 2, y2: yBase, "class": "gridline" }, svg);
      var w = textWidth(t.label);
      if (x - w / 2 > lastEnd + 6) { el("text", { x: x, y: yBase + 17, "text-anchor": "middle" }, svg).textContent = t.label; lastEnd = x + w / 2; }
    });
    el("line", { x1: x0, x2: x1, y1: yBase + 0.5, y2: yBase + 0.5, "class": "axisline" }, svg);
    if (spec.x.label) el("text", { x: (x0 + x1) / 2, y: H - 8, "text-anchor": "middle", "class": "axis-title" }, svg).textContent = spec.x.label;
    (spec.rules || []).forEach(function (r) {
      if (r.at < xr[0] || r.at > xr[1]) return;
      var x = Math.round(xSc.f(r.at)) + 0.5;
      el("line", { x1: x, x2: x, y1: top - 2, y2: yBase, "class": "rule" }, svg);
      if (r.label) el("text", { x: x, y: top - 8, "text-anchor": "middle", "class": "dlabel dlabel--muted" }, svg).textContent = r.label;
    });

    var hits = [];
    rows.forEach(function (r, i) {
      var yTop = top + i * rowH;
      var cy = stacked ? yTop + rowH - 13 : yTop + rowH / 2;
      var rowBg = el("rect", { x: 0, y: yTop, width: W, height: rowH, "class": "row-bg" }, svg);
      hits.push(rowBg);
      var cls = colourClass(r);
      if (stacked) el("text", { x: x0, y: yTop + 15, "class": "dlabel" }, svg).textContent = r.label;
      else el("text", { x: x0 - 12, y: cy + 4, "text-anchor": "end", "class": "dlabel" }, svg).textContent = r.label;
      var px = xSc.f(Math.max(r.value, xSc.log ? xr[0] : -Infinity));
      var end = px;
      if (bars) {
        var h = 14, w = Math.max(0, px - x0), rr = Math.min(4, w);
        el("path", { d: "M" + x0 + " " + (cy - h / 2) + "h" + (w - rr) + "a" + rr + " " + rr + " 0 0 1 " + rr + " " + rr + "v" + (h - 2 * rr) + "a" + rr + " " + rr + " 0 0 1 " + (-rr) + " " + rr + "h" + (-(w - rr)) + "Z", "class": "bar " + cls }, svg);
      } else if (r.value_hi !== undefined && r.value_hi !== null) {
        end = xSc.f(Math.min(r.value_hi, xr[1]));
        var line = el("line", { x1: px, x2: end, y1: cy, y2: cy, "class": "ln " + cls }, svg);
        line.style.strokeWidth = 4;
        el("circle", { cx: px, cy: cy, r: 4.5, "class": "pt " + cls }, svg);
        el("circle", { cx: end, cy: cy, r: 4.5, "class": "pt " + cls }, svg);
      } else {
        el("circle", { cx: px, cy: cy, r: 5, "class": "pt " + cls }, svg);
      }
      el("text", { x: end + 10, y: cy + 4, "class": "dlabel" + (r.emphasis ? "" : " dlabel--muted") }, svg).textContent = r.text || fmt(r.value);
      var hitRect = el("rect", { x: 0, y: yTop, width: W, height: rowH, "class": "row-hit" }, svg);
      var show = function () {
        hits.forEach(function (hrect) { hrect.classList.remove("is-active"); });
        rowBg.classList.add("is-active");
        self.showTip(Math.min(end + 10, W - 40), yTop + rowH, r.label, [{ name: spec.x.short || spec.x.label, value: r.text || fmt(r.value), cls: cls, dot: true }], r.detail);
      };
      hitRect.addEventListener("pointermove", show);
      hitRect.addEventListener("pointerdown", show);
      rowBg._show = show;
    });
    svg.addEventListener("pointerleave", function () { hits.forEach(function (hrect) { hrect.classList.remove("is-active"); }); self.hideTip(); });
    var kr = -1;
    this.plot.onkeydown = function (ev) {
      if (ev.key === "ArrowDown" || ev.key === "ArrowRight") kr = Math.min(rows.length - 1, kr + 1);
      else if (ev.key === "ArrowUp" || ev.key === "ArrowLeft") kr = Math.max(0, kr - 1);
      else if (ev.key === "Escape") { hits.forEach(function (hrect) { hrect.classList.remove("is-active"); }); self.hideTip(); return; }
      else return;
      ev.preventDefault();
      if (hits[kr]) hits[kr]._show();
    };
    this.plot.onblur = function () { hits.forEach(function (hrect) { hrect.classList.remove("is-active"); }); self.hideTip(); };
  };

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-chart]"), function (node) {
      if (node._chart) return;
      try { node._chart = new Chart(node); } catch (err) { if (window.console) console.error("chart failed", node, err); }
    });
  }
  window.MagformCharts = { init: init, fmt: fmt, update: function (node, spec) { if (node && node._chart) node._chart.update(spec); } };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
