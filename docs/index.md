---
layout: home
title: magform
titleblock: false
pager: false
description: An engineering review of magnetic forging at 1.6 to 4 GPa. Every number is computed by an open Python package and can be reproduced.
scripts: [magform.js, interactive.js]
---
{% assign n = site.data.n %}
<section class="hero">
  <div>
    <p class="eyebrow">Feasibility review &middot; computed, cited, reproducible</p>
    <h1>Can a magnet be a forging press?</h1>
    <p class="hero__lede">A concept note proposes squeezing metal with 1.6 to 4 GPa of magnetic pressure from a superconducting magnet: tungsten forged cold, unweldable alloys joined, diamond made in bulk. This site works the numbers. The pressure is real. The machine, and most of what it was meant to do, is not.</p>
    <div class="hero__actions">
      <a class="btn btn--primary" href="{{ '/story/' | relative_url }}">Read the case</a>
      <a class="btn" href="{{ '/ledger.html' | relative_url }}">{{ n.l_total }} claims, graded</a>
      <a class="btn" href="{{ '/explorer.html' | relative_url }}">Try the numbers</a>
    </div>
  </div>
  <div class="dial" data-dial>
    <div class="dial__drawing">{% include generated/press-section.svg %}</div>
    <p class="dial__caption">The reference press in section: a 100 mm bore, half a metre long, around an 80 mm billet. Field lines computed by <code>magform.fieldmap</code>. None enters the billet, which is why it is squeezed.</p>
    <div class="dial__control">
      <label for="dial-p"><span>Magnetic pressure on the billet</span><output id="dial-p-out" for="dial-p">4.00 GPa</output></label>
      <input type="range" id="dial-p" min="0.1" max="6" step="0.05" value="4">
    </div>
    <ul class="stats dial__readout">
      <li class="stat"><div class="stat__label">Field needed</div><div class="stat__value"><span data-dial="field">{{ n.p_field_hi }}</span><span class="stat__unit">T</span></div></li>
      <li class="stat"><div class="stat__label">Stored energy</div><div class="stat__value"><span data-dial="energy">{{ n.m_energy_hi }}</span><span class="stat__unit">MJ</span></div></li>
      <li class="stat"><div class="stat__label">Copper skin heats by</div><div class="stat__value"><span data-dial="heat">{{ n.f_skin_cu_hi }}</span><span class="stat__unit">K</span></div></li>
    </ul>
    <p class="dial__places" data-dial="places" role="status">Above {{ n.h_below }}, below {{ n.h_above }}. From a superconducting magnet: {% include stamp.html v=n.h_sc_status %} From a pulsed copper magnet with this bore: {% include stamp.html v=n.h_pulsed_status %}</p>
    <script type="application/json" id="magform-data">{{ site.data.explorer | jsonify }}</script>
  </div>
</section>

<section class="band">
  <p class="eyebrow">The verdict</p>
  <h2>{{ n.l_total }} statements, each checked against a model</h2>
  {% include segbar.html %}
  <div class="cols">
    <div><h3>The pressure is real</h3><p>A field of {{ n.p_field_hi }} T presses on a conductor with 4 GPa. Such a field has been made: {{ n.m_pulsed_record }} T, in a {{ n.m_pulsed_bore }} mm bore, for about a hundredth of a second.</p></div>
    <div><h3>The machine is the wrong one</h3><p>A field pushes only while it is changing, so the magnet must be pulsed. The best superconducting magnet reaches {{ n.m_sc_record }} T and cannot be pulsed. Copper coils fired from capacitor banks can.</p></div>
    <div><h3>The uses want something else</h3><p>Welds form at {{ n.w_field_min }} to {{ n.w_field_max }} T and fail above. Pores heal at {{ n.z_hip_mpa }} MPa, held hot for hours. Diamond needs heat and time as well as pressure.</p></div>
  </div>
</section>

<section class="band">
  <p class="eyebrow">Part one</p>
  <h2>The press</h2>
  <p class="lede">What 4 GPa is, what kind of magnet could deliver it, and what has to feed that magnet.</p>
  {% include case-tiles.html from=1 to=3 %}
</section>

<section class="band">
  <p class="eyebrow">Part two</p>
  <h2>Six things the note would do with it</h2>
  <p class="lede">Each application is taken in the note's own order and tested against the process that exists today.</p>
  {% include case-tiles.html from=4 to=9 %}
</section>

<section class="band">
  <p class="eyebrow">A thought experiment</p>
  <h2>And if the field were no object?</h2>
  <p class="lede">Suppose a source in space could supply the field of a magnetar, {{ n.x_gap_field }} times the record, on demand. The premise is granted, not graded, and the package computes what follows.</p>
  <ul class="stats">
    <li class="stat"><div class="stat__label">Energy in one cubic millimetre</div><div class="stat__value">{{ n.x_mm3_mt }}<span class="stat__unit">megatons</span></div><div class="stat__note">of TNT, at 10<sup>10</sup> T</div></li>
    <li class="stat"><div class="stat__label">Iron is squeezed to</div><div class="stat__value">{{ n.x_iron_ratio }}<span class="stat__unit">&times;</span></div><div class="stat__note">its bench density, and starts turning into manganese</div></li>
    <li class="stat"><div class="stat__label">Useful working distance</div><div class="stat__value">{{ n.x_r_100t }}<span class="stat__unit">m</span></div><div class="stat__note">from a 1 cm source, where the field is back to 100 T</div></li>
    <li class="stat"><div class="stat__label">Status of the source</div><div style="margin-top:0.45rem">{% include stamp.html v="assumed" %}</div><div class="stat__note">never counted for or against the note</div></li>
  </ul>
  <div class="hero__actions"><a class="btn" href="{{ '/story/magnetar.html' | relative_url }}">The magnetar mode</a></div>
</section>

<section class="band">
  <p class="eyebrow">Use it</p>
  <h2>Every number here can be re-run</h2>
  <div class="cols">
    <div><h3>Read</h3><p>Eleven short chapters, each opening with the passages under test and closing with the commands that reproduce its figures. Start with <a href="{{ '/story/' | relative_url }}">the case in brief</a>.</p></div>
    <div><h3>Check</h3><p>The <a href="{{ '/ledger.html' | relative_url }}">claims ledger</a> lists all {{ n.l_total }} statements with the quoted text, the verdict, the evidence and the functions that produced it.</p></div>
    <div><h3>Run</h3><p>The <a href="{{ '/explorer.html' | relative_url }}">explorer</a> sizes a press in the browser. The package does the rest: {{ site.data.meta.modules }} modules, standard library only.</p></div>
  </div>
<div class="prose prose--wide" style="margin-top:1.4rem" markdown="1">

```console
$ pip install -e .
$ magform press --pressure 4000          # the reference press, end to end
$ magform weld al steel --pressure 100   # a weld that works
$ magform claims C02                # one verdict with its evidence
```

</div>
</section>

<section class="band" style="border-bottom:0">
  <p class="eyebrow">Reading the stamps</p>
  <h2>Six verdicts and one premise</h2>
  <div class="table-scroll" style="max-width:46rem; margin-top:1rem">
  <table class="legend-table">
  <tbody>
  {% for v in site.data.claims.counts %}<tr><td>{% include stamp.html v=v.verdict %}</td><td>{{ v.meaning }}</td></tr>
  {% endfor %}<tr><td>{% include stamp.html v="assumed" %}</td><td>Granted as a premise and not graded. Used only for the magnetar thought experiment.</td></tr>
  </tbody>
  </table>
  </div>
</section>
