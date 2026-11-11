/* magform docs: the closed-form relations behind the explorer and the home-page dial.
 *
 * Each function mirrors one in the Python package (named in its comment), and tests/test_docs.py
 * runs this file under Node against the package, so the page cannot drift from the models.
 * No thresholds or material data live here: they arrive in the JSON written by tools/build_docs.py.
 */
(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Magform = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var MU0 = 1.25663706212e-6;      /* magform.constants.MU0 */
  var C = 299792458.0;             /* magform.constants.C_LIGHT */

  /* magform.magnetics.magnetic_pressure / field_for_pressure */
  function magneticPressure(field) { return field * field / (2 * MU0); }
  function fieldForPressure(pressure) { return Math.sqrt(2 * MU0 * pressure); }

  /* magform.magnetics.Solenoid: uniform-current thick solenoid (a1 inner radius, a2 outer radius, L length) */
  function solenoid(a1, a2, length) {
    var alpha = a2 / a1, beta = length / (2 * a1), build = a2 - a1;
    var logTerm = Math.log((alpha + Math.sqrt(alpha * alpha + beta * beta)) / (1 + Math.sqrt(1 + beta * beta)));
    function currentDensityForField(b) { return b / (MU0 * a1 * beta * logTerm); }
    function windingField(r, b) { return r <= a1 ? b : (r >= a2 ? 0 : b * (a2 - r) / build); }
    return {
      innerRadius: a1, outerRadius: a2, length: length, alpha: alpha, beta: beta, build: build,
      fieldFactor: beta / (alpha - 1) * logTerm,                                   /* Solenoid.field_factor */
      currentDensityForField: currentDensityForField,                             /* .current_density_for_field */
      storedEnergy: function (b) {                                                /* .stored_energy */
        return magneticPressure(b) * Math.PI * length * (a1 * a1 + 2 * build * (a2 / 3 - build / 4));
      },
      maxHoopStress: function (b) {                                               /* .max_hoop_stress with the Montgomery J */
        var r = a2 <= 2 * a1 ? a1 : 0.5 * a2;
        return r * currentDensityForField(b) * windingField(r, b);
      },
      averageHoopStress: function (b) { return magneticPressure(b) * (a2 + 2 * a1) / (3 * build); }  /* .average_hoop_stress */
    };
  }

  /* magform.eddy.skin_depth / frequency_for_skin_depth / pulse_frequency_recommendation(...).quarter_period */
  function skinDepth(resistivity, frequency) { return Math.sqrt(resistivity / (Math.PI * MU0 * frequency)); }
  function frequencyForSkinDepth(resistivity, depth) { return resistivity / (Math.PI * MU0 * depth * depth); }
  function matchedRiseTime(resistivity, wall, depthRatio) {
    return 0.25 / frequencyForSkinDepth(resistivity, wall / (depthRatio || 3));
  }
  /* magform.eddy.sheet_shielding_factor */
  function pressureTransfer(wall, depth) { return -Math.expm1(-2 * wall / depth); }
  /* magform.eddy.skin_temperature_rise with kappa = 1 */
  function skinTemperatureRise(field, density, specificHeat) { return magneticPressure(field) / (density * specificHeat); }
  /* magform.pulse.discharge_peak_power */
  function peakPower(energy, riseTime) { return 0.5 * Math.PI * energy / riseTime; }

  /* magform.benchmarks status helpers; t is the "thresholds" object of _data/explorer.json */
  function superconductingStatus(field, t) {
    return field <= t.sc_record ? "state_of_the_art" : field <= t.sc_conductor ? "extrapolation" : "aspirational";
  }
  function pulsedStatus(field, bore, t) {
    if (field > t.pulsed_record) return "aspirational";
    if (field <= t.pulsed_routine_field) return bore <= t.pulsed_routine_bore ? "state_of_the_art" : "extrapolation";
    return bore <= t.pulsed_bore * 1.05 ? "state_of_the_art" : "extrapolation";
  }
  function energyStatus(energy, t) {
    return energy <= t.energy_sota ? "state_of_the_art" : energy <= t.energy_record ? "extrapolation" : "aspirational";
  }
  function powerStatus(power, t) {
    power = Math.abs(power);
    return power <= t.power_sota ? "state_of_the_art" : power <= t.power_record ? "extrapolation" : "aspirational";
  }
  function voltageStatus(voltage, t) {                                           /* pulsed_voltage_status */
    voltage = Math.abs(voltage);
    return voltage <= t.voltage_sota ? "state_of_the_art" : voltage <= t.voltage_extrapolation ? "extrapolation" : "aspirational";
  }
  function currentStatus(current, t) {                                           /* pulsed_current_status */
    current = Math.abs(current);
    return current <= t.current_sota ? "state_of_the_art" : current <= t.current_record ? "extrapolation" : "aspirational";
  }

  /* magform.pulse: the quarter-period law V0 I_pk = pi E / t_rise (voltage_current_product) for a coil.
   * ampereTurns is Solenoid.current_for_field(B, 1).  The turn count is the largest that keeps the bank
   * at or below t.max_voltage, or one; assess_pulse_network also allows for resistance and stray
   * inductance, so its count can differ by a turn.  A one-turn coil carries the whole field step on its
   * own surface, which melts above t.coil_skin_melting_field (the thermal rule of assess_pulse_network). */
  function supply(coil, field, riseTime, t) {
    var energy = coil.storedEnergy(field);
    var ampereTurns = coil.currentDensityForField(field) * coil.length * coil.build;
    var singleTurnVoltage = Math.PI * energy / (riseTime * ampereTurns);
    var turns = Math.max(1, Math.floor(t.max_voltage / singleTurnVoltage));
    var forced = singleTurnVoltage > t.max_voltage;
    return {
      energy: energy, ampereTurns: ampereTurns, singleTurnVoltage: singleTurnVoltage, turns: turns,
      voltage: singleTurnVoltage * turns, current: ampereTurns / turns,
      coilSkinMelts: forced && field >= t.coil_skin_melting_field
    };
  }

  var RANK = { state_of_the_art: 0, extrapolation: 1, aspirational: 2, infeasible: 3 };
  /* magform.status.worst */
  function worst(list) { return list.reduce(function (a, b) { return RANK[b] > RANK[a] ? b : a; }); }

  /* magform.magnetar: MagnetarSource.energy / mass_equivalent / confinement_time / assembly_power, standoff_radius */
  function sourceEnergy(field, radius) { return 1.5 * magneticPressure(field) * (4 / 3) * Math.PI * radius * radius * radius; }
  function massEquivalent(energy) { return energy / (C * C); }
  function confinementTime(radius) { return radius / C; }
  function standoffRadius(surfaceField, radius, threshold) {
    return Math.max(radius, radius * Math.cbrt(surfaceField / threshold));
  }
  /* which regime tier a field falls in; bounds is "tier_bounds" of _data/explorer.json (ascending) */
  function tierFor(field, bounds) {
    var tier = bounds[0].tier;
    bounds.forEach(function (b) { if (field >= b.from) tier = b.tier; });
    return tier;
  }

  return {
    MU0: MU0, C: C,
    magneticPressure: magneticPressure, fieldForPressure: fieldForPressure, solenoid: solenoid,
    skinDepth: skinDepth, frequencyForSkinDepth: frequencyForSkinDepth, matchedRiseTime: matchedRiseTime,
    pressureTransfer: pressureTransfer, skinTemperatureRise: skinTemperatureRise, peakPower: peakPower,
    superconductingStatus: superconductingStatus, pulsedStatus: pulsedStatus, energyStatus: energyStatus,
    powerStatus: powerStatus, voltageStatus: voltageStatus, currentStatus: currentStatus, supply: supply, worst: worst,
    sourceEnergy: sourceEnergy, massEquivalent: massEquivalent, confinementTime: confinementTime,
    standoffRadius: standoffRadius, tierFor: tierFor
  };
});
