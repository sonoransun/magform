---
title: Explorer
kicker: Tools
doc: MF-X1
description: Size a magnetic press in the browser, or turn the field up to magnetar strength, and watch the verdicts change. The same relations as the package, checked against it by the test suite.
wide: true
scripts: [magform.js, interactive.js]
---
{% assign n = site.data.n %}
<div data-explorer>

<p class="explorer-intro">This page runs the closed-form part of the package in your browser: the field for a pressure, the energy of a coil, the rise time a workpiece demands and the power that follows from it. Thresholds and material data are read from the same generated file as the rest of the site. For the full models, use the command printed under the readouts.</p>

<div class="tabs" role="tablist" aria-label="Explorer">
  <button class="tab" type="button" role="tab" id="tab-press" data-tab="press" aria-controls="panel-press" aria-selected="true">The press</button>
  <button class="tab" type="button" role="tab" id="tab-magnetar" data-tab="magnetar" aria-controls="panel-magnetar" aria-selected="false" tabindex="-1">The magnetar mode</button>
</div>

<section class="tabpanel" id="panel-press" role="tabpanel" aria-labelledby="tab-press">
  <div class="panel">
    <div class="controls">
      <div class="control"><label for="x-pressure"><span>Magnetic pressure</span><output for="x-pressure">4.00 GPa</output></label><input type="range" id="x-pressure" min="0.1" max="6" step="0.05" value="4"></div>
      <div class="control"><label for="x-bore"><span>Bore diameter</span><output for="x-bore">100 mm</output></label><input type="range" id="x-bore" min="10" max="500" step="5" value="100"></div>
      <div class="control"><label for="x-build"><span>Winding build, in bores</span><output for="x-build">200 mm</output></label><input type="range" id="x-build" min="0.5" max="4" step="0.25" value="2"></div>
      <div class="control"><label for="x-length"><span>Coil length, in bores</span><output for="x-length">0.50 m</output></label><input type="range" id="x-length" min="1" max="10" step="0.5" value="5"></div>
      <div class="control"><label for="x-material"><span>Workpiece</span></label><select id="x-material"></select></div>
      <div class="control"><label for="x-wall"><span>Workpiece wall</span><output for="x-wall">40.0 mm</output></label><input type="range" id="x-wall" min="0.5" max="60" step="0.5" value="40"></div>
    </div>
  </div>

  <p class="verdictline" data-out="verdict" role="status">The verdicts appear here when the page script has loaded.</p>

  <div class="readouts">
    <div class="readout"><div class="readout__label">Field at the workpiece</div><div class="readout__value" data-out="field">{{ n.p_field_hi }} <small>T</small></div><div class="readout__note" data-out="field-note"></div></div>
    <div class="readout"><div class="readout__label">Energy stored in the field</div><div class="readout__value" data-out="energy">{{ n.e_energy_hi }} <small>MJ</small></div><div class="readout__note" data-out="energy-note"></div></div>
    <div class="readout"><div class="readout__label">Rise time the workpiece needs</div><div class="readout__value" data-out="rise">{{ n.m_rise_billet }} <small>ms</small></div><div class="readout__note" data-out="rise-note"></div></div>
    <div class="readout"><div class="readout__label">Peak power</div><div class="readout__value" data-out="power">{{ n.e_slow_gw }} <small>GW</small></div><div class="readout__note" data-out="power-note"></div></div>
    <div class="readout"><div class="readout__label">Bank voltage and peak current</div><div class="readout__value" data-out="supply">{{ n.e_slow_kv }} <small>kV</small> × {{ n.e_slow_ma }} <small>MA</small></div><div class="readout__note" data-out="supply-note"></div></div>
    <div class="readout"><div class="readout__label">Peak hoop stress in the coil</div><div class="readout__value" data-out="hoop">{{ n.m_hoop_hi }} <small>GPa</small></div><div class="readout__note" data-out="hoop-note"></div></div>
    <div class="readout"><div class="readout__label">Workpiece surface heating</div><div class="readout__value" data-out="heat">+{{ n.f_skin_cu_hi }} <small>K</small></div><div class="readout__note" data-out="heat-note"></div></div>
  </div>

  <figure class="figure figure--wide">
    <div class="chart" data-chart id="x-ladder">
      <script type="application/json">{{ site.data.figures.explorer_ladder | jsonify }}</script>
      <p class="chart__title">{{ site.data.figures.explorer_ladder.title }}</p>
      <p class="chart__sub">{{ site.data.figures.explorer_ladder.subtitle }}</p>
    </div>
    <figcaption data-out="between"></figcaption>
  </figure>

  <p class="small muted" style="margin-top:1.25rem">The full model for these inputs, with the conductor, the structure, the pulse network and six ways of building the magnet:</p>
  <code class="command" data-out="command">magform press --pressure 4000</code>
</section>

<section class="tabpanel" id="panel-magnetar" role="tabpanel" aria-labelledby="tab-magnetar" hidden>
  <div class="note" data-label="Premise">
    <p>{{ site.data.magnetar.premise }} {% include stamp.html v="assumed" %} See <a href="{{ '/story/magnetar.html' | relative_url }}">chapter 11</a> for the argument.</p>
  </div>
  <div class="panel" style="margin-top:1.25rem">
    <div class="controls">
      <div class="control"><label for="y-field"><span>Field at the surface of the source</span><output for="y-field">10¹⁰ T</output></label><input type="range" id="y-field" min="2" max="11.5" step="0.05" value="10"></div>
      <div class="control"><label for="y-radius"><span>Radius of the source</span><output for="y-radius">10 mm</output></label><input type="range" id="y-radius" min="-6" max="0" step="0.05" value="-2"></div>
    </div>
  </div>

  <div class="readouts">
    <div class="readout"><div class="readout__label">Magnetic pressure</div><div class="readout__value" data-out="pressure"></div><div class="readout__note" data-out="pressure-note"></div></div>
    <div class="readout"><div class="readout__label">Energy of one shot</div><div class="readout__value" data-out="energy"></div><div class="readout__note" data-out="energy-note"></div></div>
    <div class="readout"><div class="readout__label">Life of the field, unheld</div><div class="readout__value" data-out="life"></div><div class="readout__note" data-out="life-note"></div></div>
    <div class="readout"><div class="readout__label">Against the record</div><div class="readout__value" data-out="ratio"></div><div class="readout__note" data-out="ratio-note"></div></div>
    <div class="readout"><div class="readout__label">Working distance</div><div class="readout__value" data-out="work"></div><div class="readout__note" data-out="work-note"></div></div>
    <div class="readout"><div class="readout__label">Keep-out distance</div><div class="readout__value" data-out="keepout"></div><div class="readout__note" data-out="keepout-note"></div></div>
  </div>

  <div class="matter">
    <p class="label">An iron workpiece at the surface of the source</p>
    <h3 data-out="tier"></h3>
    <p data-out="state"></p>
    <div class="table-scroll" style="margin-top:1rem">
      <table><thead><tr><th>Process</th><th>What it becomes at this field</th></tr></thead><tbody data-out="processes"></tbody></table>
    </div>
  </div>
  <p class="small muted" style="margin-top:1.25rem">The full assessment, with stand-off distances and the gap to every record:</p>
  <code class="command" data-out="command">magform magnetar --field 1e10 --radius 0.01</code>
</section>

<script type="application/json" id="magform-data">{{ site.data.explorer | jsonify }}</script>
</div>
