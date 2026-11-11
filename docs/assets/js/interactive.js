/* magform docs: the home-page dial and the explorer.
 *
 * Formulas come from magform.js (checked against the Python package by the tests); thresholds, materials
 * and regime data come from the JSON that tools/build_docs.py writes into the page.
 */
(function () {
  "use strict";
  var M = window.Magform, dataNode = document.getElementById("magform-data");
  if (!M || !dataNode) return;
  var D = JSON.parse(dataNode.textContent), T = D.thresholds;
  var SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };

  /* ---------------------------------------------------------------- formatting */
  function num(x, digits) {
    return x.toLocaleString("en-US", { minimumFractionDigits: digits || 0, maximumFractionDigits: digits || 0 });
  }
  function sig(x, n) {                       /* n significant digits, with thousands separators, or a × 10ᵇ */
    if (!isFinite(x)) return "–";
    if (x === 0) return "0";
    var a = Math.abs(x), e = Math.floor(Math.log10(a));
    if (a >= 1e-3 && a < 1e6) return num(Number(x.toPrecision(n)), Math.max(0, n - 1 - e));
    var m = Number((x / Math.pow(10, e)).toPrecision(n));
    if (m >= 10) { m /= 10; e += 1; }
    var exp = String(e).split("").map(function (ch) { return SUP[ch] || ch; }).join("");
    return (m === 1 ? "" : m + " × ") + "10" + exp;
  }
  function scaled(x, units, n) {             /* units: [[factor, label], ...] ascending */
    var pick = units[0];
    units.forEach(function (u) { if (Math.abs(x) >= u[0]) pick = u; });
    return { value: sig(x / pick[0], n || 3), unit: pick[1] };
  }
  var ENERGY = [[1, "J"], [1e3, "kJ"], [1e6, "MJ"], [1e9, "GJ"], [1e12, "TJ"]];
  var POWER = [[1, "W"], [1e3, "kW"], [1e6, "MW"], [1e9, "GW"], [1e12, "TW"]];
  var TIME = [[1e-12, "ps"], [1e-9, "ns"], [1e-6, "µs"], [1e-3, "ms"], [1, "s"]];
  var LENGTH = [[1e-6, "µm"], [1e-3, "mm"], [1, "m"], [1e3, "km"]];
  var MASS = [[1e-9, "µg"], [1e-6, "mg"], [1e-3, "g"], [1, "kg"], [1e3, "t"]];
  function text(u) { return u.value + " " + u.unit; }

  function stamp(v, label) {
    return '<span class="stamp stamp--' + v + '"><span class="stamp__icon"><svg aria-hidden="true"><use href="#i-' + v +
      '"/></svg></span><span class="stamp__label">' + (label || v.replace(/_/g, " ")) + "</span></span>";
  }
  function set(root, key, value) {
    Array.prototype.forEach.call(root.querySelectorAll('[data-out="' + key + '"]'), function (el) { el.innerHTML = value; });
  }
  function between(pressure) {               /* nearest cited working pressures below and above */
    var below = null, above = null;
    D.pressures.forEach(function (b) { if (b.value <= pressure) below = b; else if (!above) above = b; });
    function name(b) { return b.name + " (" + sig(b.value / 1e9, 2) + " GPa)"; }
    if (below && above) return "Above " + name(below) + ", below " + name(above) + ".";
    return below ? "Above " + name(below) + "." : "Below " + name(above) + ".";
  }

  /* ---------------------------------------------------------------- home-page dial */
  var dial = document.querySelector("[data-dial]");
  if (dial) {
    var slider = dial.querySelector("#dial-p"), out = dial.querySelector("#dial-p-out");
    var ref = M.solenoid(D.reference.bore / 2, D.reference.bore / 2 + D.reference.build, D.reference.length);
    var copper = D.materials[0];
    var lines = dial.querySelector(".fl");
    var show = function (key, value) { var el = dial.querySelector('[data-dial="' + key + '"]'); if (el) el.innerHTML = value; };
    var update = function () {
      var p = Number(slider.value) * 1e9, b = M.fieldForPressure(p);
      out.textContent = num(p / 1e9, 2) + " GPa";
      show("field", num(b, 1));
      show("energy", sig(ref.storedEnergy(b) / 1e6, 3));
      var heat = M.skinTemperatureRise(b, copper.density, copper.specific_heat);
      show("heat", num(heat, 0));
      var melted = b >= copper.skin_melting_field;
      show("places", between(p) + " From a superconducting magnet: " + stamp(M.superconductingStatus(b, T)) +
        " From a pulsed copper magnet with this bore: " + stamp(M.pulsedStatus(b, D.reference.bore, T)) +
        (melted ? " The copper surface is molten." : ""));
      if (lines) lines.style.opacity = String(0.25 + 0.65 * Math.min(1, b / 110));
    };
    slider.addEventListener("input", update);
    update();
  }

  /* ---------------------------------------------------------------- explorer: tabs */
  var explorer = document.querySelector("[data-explorer]");
  if (!explorer) return;
  var tabs = explorer.querySelectorAll('[role="tab"]');
  function select(tab) {
    Array.prototype.forEach.call(tabs, function (t) {
      var on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (window.MagformCharts) window.MagformCharts.init();
  }
  Array.prototype.forEach.call(tabs, function (tab, i) {
    tab.addEventListener("click", function () { select(tab); if (history.replaceState) history.replaceState(null, "", "#" + tab.dataset.tab); });
    tab.addEventListener("keydown", function (ev) {
      var next = ev.key === "ArrowRight" ? tabs[(i + 1) % tabs.length] : ev.key === "ArrowLeft" ? tabs[(i + tabs.length - 1) % tabs.length] : null;
      if (next) { ev.preventDefault(); next.focus(); select(next); }
    });
  });
  function fromHash() {
    var wanted = location.hash.replace("#", "");
    Array.prototype.forEach.call(tabs, function (t) { if (t.dataset.tab === wanted) select(t); });
  }
  window.addEventListener("hashchange", fromHash);
  fromHash();

  /* ---------------------------------------------------------------- explorer: the press */
  var press = document.getElementById("panel-press");
  var material = press.querySelector("#x-material");
  D.materials.forEach(function (m) {
    var o = document.createElement("option"); o.value = m.key; o.textContent = m.name; material.appendChild(o);
  });
  function val(root, id) { return Number(root.querySelector("#" + id).value); }
  function label(root, id, textValue) { var o = root.querySelector('output[for="' + id + '"]'); if (o) o.textContent = textValue; }

  function updatePress() {
    var p = val(press, "x-pressure") * 1e9, bore = val(press, "x-bore") / 1e3;
    var build = val(press, "x-build") * bore, length = val(press, "x-length") * bore;
    var wall = val(press, "x-wall") / 1e3;
    var m = D.materials.filter(function (x) { return x.key === material.value; })[0] || D.materials[0];
    var b = M.fieldForPressure(p), coil = M.solenoid(bore / 2, bore / 2 + build, length);
    var energy = coil.storedEnergy(b), rise = M.matchedRiseTime(m.resistivity, wall, D.constants.depth_ratio);
    var power = M.peakPower(energy, rise), hoop = coil.maxHoopStress(b), bank = M.supply(coil, b, rise, T);
    var vStatus = M.voltageStatus(bank.voltage, T), iStatus = M.currentStatus(bank.current, T);
    var heat = M.skinTemperatureRise(b, m.density, m.specific_heat), melts = b >= m.skin_melting_field;
    var sc = M.superconductingStatus(b, T), pulsed = M.pulsedStatus(b, bore, T);
    var eStatus = M.energyStatus(energy, T), pStatus = M.powerStatus(power, T);
    var overall = M.worst([pulsed, eStatus, pStatus, vStatus, iStatus, bank.coilSkinMelts ? "infeasible" : "state_of_the_art"]);

    label(press, "x-pressure", num(p / 1e9, 2) + " GPa");
    label(press, "x-bore", num(bore * 1e3, 0) + " mm");
    label(press, "x-build", num(build * 1e3, 0) + " mm");
    label(press, "x-length", num(length, 2) + " m");
    label(press, "x-wall", num(wall * 1e3, 1) + " mm");

    set(press, "field", num(b, 1) + " <small>T</small>");
    set(press, "field-note", "Superconducting: " + stamp(sc) + "<br>Pulsed copper, this bore: " + stamp(pulsed));
    var e = scaled(energy, ENERGY);
    set(press, "energy", e.value + " <small>" + e.unit + "</small>");
    set(press, "energy-note", "As much as " + text(scaled(energy / D.constants.tnt_per_kg, MASS)) + " of TNT, on every shot. " + stamp(eStatus));
    var t = scaled(rise, TIME);
    set(press, "rise", t.value + " <small>" + t.unit + "</small>");
    set(press, "rise-note", "The field must rise this fast to stay out of a " + num(wall * 1e3, 1) + " mm " + m.name.toLowerCase() + " wall.");
    var w = scaled(power, POWER);
    set(press, "power", w.value + " <small>" + w.unit + "</small>");
    set(press, "power-note", "Peak power from the supply, πE/2t. " + stamp(pStatus));
    set(press, "supply", sig(bank.voltage / 1e3, 3) + " <small>kV</small> × " + sig(bank.current / 1e6, 3) + " <small>MA</small>");
    set(press, "supply-note", (bank.turns === 1 ? "One turn" : bank.turns + " turns") + ". Voltage times current is fixed by the energy and the rise time. " +
      stamp(M.worst([vStatus, iStatus])) + (bank.coilSkinMelts ? " " + stamp("infeasible", "coil surface melts") : ""));
    set(press, "hoop", num(hoop / 1e9, 2) + " <small>GPa</small>");
    set(press, "hoop-note", num(hoop / p, 2) + " times the magnetic pressure, carried by the winding. Current density " +
      num(coil.currentDensityForField(b) / 1e6, 0) + " A/mm².");
    set(press, "heat", "+" + num(heat, 0) + " <small>K</small>");
    set(press, "heat-note", m.name + " melts at its surface above " + num(m.skin_melting_field, 0) + " T. " +
      (melts ? stamp("infeasible", "skin melts") : stamp("state_of_the_art", "stays solid")));
    set(press, "verdict", "Built as a pulsed copper magnet: " + stamp(overall) + " &nbsp; Field alone, from a superconducting magnet: " + stamp(sc) +
      '<br><span class="muted small">The first verdict is the worst of field and bore, stored energy, peak power, bank voltage and current, and whether a one-turn coil melts its own surface. The second leaves out the pulse, which the full model finds no superconducting winding survives.</span>');
    set(press, "between", between(p));
    set(press, "command", "magform press --pressure " + num(p / 1e6, 0).replace(/,/g, "") + " --bore " + bore.toPrecision(3).replace(/\.?0+$/, "") +
      " --length " + Number(length.toPrecision(3)) + " --winding " + Number(build.toPrecision(3)) + " --workpiece " + m.key + " --wall " + Number(wall.toPrecision(3)));

    var chart = document.getElementById("x-ladder");
    if (chart && chart._chart && window.MagformCharts.update) {
      var rows = D.pressures.map(function (x) { return { label: x.name.charAt(0).toUpperCase() + x.name.slice(1), value: x.value / 1e9, text: sig(x.value / 1e9, 3) + " GPa", tone: "muted" }; });
      rows.push({ label: "This press", value: p / 1e9, text: num(p / 1e9, 2) + " GPa", tone: "accent", emphasis: true });
      rows.sort(function (a, c) { return a.value - c.value; });
      var spec = JSON.parse(JSON.stringify(chart._chart.spec));
      spec.rows = rows;
      window.MagformCharts.update(chart, spec);
    }
  }
  Array.prototype.forEach.call(press.querySelectorAll("input, select"), function (el) { el.addEventListener("input", updatePress); });

  /* ---------------------------------------------------------------- explorer: the magnetar mode */
  var mag = document.getElementById("panel-magnetar");
  function updateMagnetar() {
    var b = Math.pow(10, val(mag, "y-field")), r = Math.pow(10, val(mag, "y-radius"));
    var energy = M.sourceEnergy(b, r), pressure = M.magneticPressure(b);
    var tier = M.tierFor(b, D.magnetar.tier_bounds), info = D.magnetar.tiers[tier];
    var crossed = D.magnetar.thresholds.filter(function (x) { return b >= x.field; });
    label(mag, "y-field", sig(b, 2) + " T");
    label(mag, "y-radius", text(scaled(r, LENGTH, 2)));
    set(mag, "pressure", sig(pressure, 2) + " <small>Pa</small>");
    set(mag, "pressure-note", sig(pressure / D.window.p_hi, 2) + " times the top of the note's window.");
    var e = scaled(energy, ENERGY);
    set(mag, "energy", (energy >= 1e15 ? sig(energy, 2) + " <small>J</small>" : e.value + " <small>" + e.unit + "</small>"));
    set(mag, "energy-note", energy >= 0.001 * D.constants.megaton ? sig(energy / D.constants.megaton, 3) + " megatons of TNT; " +
      text(scaled(M.massEquivalent(energy), MASS)) + " by E = mc²." : text(scaled(energy / D.constants.tnt_per_kg, MASS)) + " of TNT.");
    set(mag, "life", text(scaled(M.confinementTime(r), TIME, 2)));
    set(mag, "life-note", "Light crosses the source in this time. Unheld, the field is gone by then.");
    var record = T.pulsed_record;
    set(mag, "ratio", sig(b / record, 2) + " <small>×</small>");
    set(mag, "ratio-note", "the strongest field a magnet has survived (" + record + " T)" +
      (b > D.magnetar.destructive_record ? ", and " + sig(b / D.magnetar.destructive_record, 2) + " times the destructive record." : "."));
    var work = M.standoffRadius(b, r, M.fieldForPressure(D.window.p_hi));
    set(mag, "work", b > M.fieldForPressure(D.window.p_hi) ? text(scaled(work, LENGTH, 3)) : "at the source");
    set(mag, "work-note", "Distance at which the field has fallen to 100 T, the top of the note's window.");
    var pace = D.magnetar.standoff.filter(function (x) { return x.key === "pacemaker_line"; })[0];
    set(mag, "keepout", text(scaled(M.standoffRadius(b, r, pace.field), LENGTH, 3)));
    set(mag, "keepout-note", "Distance to the 0.5 mT line for implanted medical devices.");
    set(mag, "tier", info.name.charAt(0).toUpperCase() + info.name.slice(1));
    set(mag, "state", crossed.length ? crossed[crossed.length - 1].state.charAt(0).toUpperCase() + crossed[crossed.length - 1].state.slice(1) + "." :
      "Solid and elastic: the pressure is below the yield strength of iron and nothing permanent happens.");
    var list = "";
    Object.keys(D.magnetar.process_names).forEach(function (k) {
      list += "<tr><td><b>" + D.magnetar.process_names[k] + "</b></td><td>" + info.processes[k] + "</td></tr>";
    });
    set(mag, "processes", list);
    set(mag, "command", "magform magnetar --field " + b.toExponential(2).replace("e+", "e") + " --radius " + r.toExponential(2).replace("e+", "e"));
  }
  Array.prototype.forEach.call(mag.querySelectorAll("input"), function (el) { el.addEventListener("input", updateMagnetar); });

  function start() { updatePress(); updateMagnetar(); }
  if (document.readyState === "complete") start(); else window.addEventListener("load", start);
})();
