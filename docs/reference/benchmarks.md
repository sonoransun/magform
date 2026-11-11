---
title: Benchmarks
kicker: Reference
doc: MF-R2
description: What has actually been demonstrated, with a date and a source for each entry. Every status on this site is a comparison with one of these tables.
models: [benchmarks]
wide: true
---
{% assign b = site.data.benchmarks %}

<div class="prose" markdown="1">

A status on this site is a comparison. The package computes a field, an energy or a pressure, places it on a ladder of things that have been done, and reads the label off the ladder. This page lists the {{ b.count }} entries in those ladders, the rule each status helper applies and the thresholds behind the rules.

The **Data** column is a provenance tag. `handbook` is a primary measurement or publication, `representative` a typical literature value, `estimated` a derived figure and `reported` an institutional announcement that is not yet in a journal.

</div>

{% include figure.html id="ref_field_ladder" caption="<b>Record fields on a log scale.</b> Each dot is one row of the first table below. Hover or use the arrow keys for the source of each." %}

{% for l in b.ladders %}

<h2 id="{{ l.key }}">{{ l.title }}</h2>

{% if l.intro != "" %}
<div class="prose">
<p>{{ l.intro | escape }}</p>
</div>
{% endif %}

{% if l.key == "energy" %}
{% include figure.html id="ref_energy_ladder" caption="<b>Pulsed energy stores on a log scale.</b> The band is the field energy of the site's reference press between 1.6 and 4 GPa." %}
{% endif %}

<div class="table-scroll" tabindex="0" role="region" aria-label="{{ l.title }}">
<table style="table-layout: fixed; min-width: 46rem">
<colgroup><col style="width: 36%"><col style="width: 9%"><col style="width: 6%"><col style="width: 37%"><col style="width: 12%"></colgroup>
<thead>
<tr><th>Achievement</th><th class="r">Value<br><span style="text-transform: none">{{ l.unit }}</span></th><th class="r">Year</th><th>Source</th><th>Data</th></tr>
</thead>
<tbody>
{% for r in l.rows %}
<tr id="{{ r.key }}">
<td>{{ r.description | escape }}{% if r.note %}<br><span style="color: var(--ink-3); font-size: 0.88em">{{ r.note | escape }}</span>{% endif %}</td>
<td class="r"><strong>{{ r.value }}</strong></td>
<td class="r">{{ r.year }}</td>
<td>{{ r.citation | escape }}</td>
<td>{{ r.confidence }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

{% endfor %}

<h2 id="rules">How a status is assigned</h2>

<div class="prose">
<p>Each helper in <code>magform.benchmarks</code> takes one computed quantity and returns a status. The bands below are read from the helpers themselves by probing them, so this table always matches the code.</p>
</div>

<div class="table-scroll" tabindex="0" role="region" aria-label="Status rules">
<table style="table-layout: fixed; min-width: 46rem">
<colgroup><col style="width: 42%"><col style="width: 30%"><col style="width: 28%"></colgroup>
<thead>
<tr><th>What is graded</th><th>Bands</th><th>Helper</th></tr>
</thead>
<tbody>
{% for r in b.rules %}
<tr>
<td>{{ r.what | escape }}</td>
<td>{% for band in r.bands %}{% include stamp.html v=band.status %} {% if band.upto %}up to {{ band.upto }}{% else %}beyond that{% endif %}{% unless forloop.last %}<br>{% endunless %}{% endfor %}</td>
<td><code>{{ r.function }}</code></td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<div class="prose" markdown="1">

{% assign record = b.thresholds | where: "name", "PULSED_NONDESTRUCTIVE_RECORD" | first %}{% assign bore = b.thresholds | where: "name", "PULSED_NONDESTRUCTIVE_BORE" | first %}
Two bands are judgement calls and are marked as estimated in the source: the sweep rate above which a superconducting winding is treated as aspirational, and the voltages above which insulation is. A pulsed field is graded on its bore as well as its strength, because stored energy grows with the bore volume and the {{ record.value }} record was set in a {{ bore.value }} bore.

</div>

<h2 id="thresholds">Named thresholds</h2>

<div class="prose">
<p>The constants the helpers compare against. Other modules import these names, so a record that changes is changed in one place.</p>
</div>

<div class="table-scroll" tabindex="0" role="region" aria-label="Named thresholds">
<table>
<thead>
<tr><th>Constant</th><th class="r">Value</th><th>Meaning</th></tr>
</thead>
<tbody>
{% for t in b.thresholds %}
<tr>
<td><code>{{ t.name }}</code></td>
<td class="r" style="white-space: nowrap">{{ t.value }}</td>
<td>{{ t.meaning | escape }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<div class="prose" markdown="1">

## Check it yourself

```python
from magform import benchmarks

benchmarks.superconducting_field_status(63.4)      # aspirational
benchmarks.pulsed_field_status(100.27, 0.010)      # state_of_the_art: the Los Alamos bore
benchmarks.pulsed_field_status(100.27, 0.10)       # extrapolation: same field, 100 mm bore
benchmarks.where_does_it_sit(4e9).nearest          # the benchmark closest to 4 GPa
```

</div>
