---
title: Materials
kicker: Reference
doc: MF-R1
description: The property data every model draws on, with the condition and the provenance of each entry.
models: [materials]
wide: true
---
{% assign mat = site.data.materials %}
{% assign w = mat.rows | where: "key", "tungsten" | first %}

<div class="prose" markdown="1">

The registry holds {{ mat.count }} materials in {{ mat.categories.size }} groups. Values are representative engineering data. They are not datasheet guarantees, and strength in particular depends on the condition of the metal. Inside the package everything is stored in SI units. The tables below use the units engineers quote.

Models look a material up by key or by alias, and fail with a suggestion when the name is unknown.

```python
from magform import materials

w = materials.get("W")                    # alias of "tungsten"
w.yield_strength / 1e6, w.dbtt            # {{ w.yield }} MPa, {{ w.dbtt }} K
materials.list_materials("refractory")    # every record in one group
```

## Where each number comes from

Every record carries one of four provenance tags, shown under its key in the tables. The same tags are used for constitutive fits, Hugoniots and benchmark entries.

<table>
<thead><tr><th>Tag</th><th>Meaning</th><th class="r">Materials</th></tr></thead>
<tbody>
{% for c in mat.confidence %}
<tr><td><code>{{ c.key }}</code></td><td>{{ c.meaning | escape }}</td><td class="r">{{ c.count }}</td></tr>
{% endfor %}
</tbody>
</table>

## Reading the tables

- **Yield** and **UTS** are tensile values at room temperature for the condition named with the material. A dash means the material does not yield (fibres, ceramics) or the value is not recorded.
- **Resistivity** is the value at 293 K. Models scale it linearly with temperature.
- **Hardness** is the recorded Vickers hardness. Where none is recorded the models estimate it as three times the flow stress.
- **DBTT** is the quasi-static ductile-brittle transition temperature. Below it the forgeability model asks whether the part is confined.
- **Flow model** lists the published constitutive fits on record: JC is a Johnson-Cook fit, ZA a Zerilli-Armstrong fit. {{ mat.with_fit }} materials have one. The others use a power law built from the tensile data.
- **Hugoniot** says whether the shock response is measured or estimated. {{ mat.with_measured_hugoniot }} are measured. A dash means the models estimate it from the bulk sound speed when they need it.
- **Hall-Petch** is the slope that links yield strength to grain size. "est." marks an estimated slope.

</div>

{% for c in mat.categories %}
{% assign rows = mat.rows | where: "category", c.key %}
{% assign noted = rows | where_exp: "m", "m.notes != nil" %}

<h2 id="{{ c.key }}">{{ c.label }}</h2>

<div class="table-scroll" tabindex="0" role="region" aria-label="{{ c.label }}">
<table style="table-layout: fixed; min-width: 58rem">
<colgroup>
<col style="width: 21%"><col style="width: 7%"><col style="width: 7%"><col style="width: 6%"><col style="width: 6%"><col style="width: 6%">
<col style="width: 9.5%"><col style="width: 7.5%"><col style="width: 5.5%"><col style="width: 6.5%"><col style="width: 8.5%"><col style="width: 9.5%">
</colgroup>
<thead>
<tr>
<th>Material</th>
<th class="r">Density<br><span style="text-transform: none">kg/m³</span></th>
<th class="r">Modulus<br><span style="text-transform: none">GPa</span></th>
<th class="r">Yield<br><span style="text-transform: none">MPa</span></th>
<th class="r">UTS<br><span style="text-transform: none">MPa</span></th>
<th class="r">Melts<br><span style="text-transform: none">K</span></th>
<th class="r">Resistivity<br><span style="text-transform: none">µΩ·cm</span></th>
<th class="r">Hardness<br><span style="text-transform: none">GPa</span></th>
<th class="r">DBTT<br><span style="text-transform: none">K</span></th>
<th>Flow<br>model</th>
<th>Hugoniot</th>
<th class="r">Hall-Petch<br><span style="text-transform: none">MPa·√m</span></th>
</tr>
</thead>
<tbody>
{% for m in rows %}
<tr id="{{ m.key }}">
<td>{{ m.name | escape }}{% if m.condition_extra %}<br><span style="color: var(--ink-2)">{{ m.condition_extra | escape }}</span>{% endif %}<br><code>{{ m.key }}</code> <span class="mono" style="color: var(--ink-3)">{{ m.confidence }}</span></td>
<td class="r">{{ m.density }}</td>
<td class="r">{{ m.modulus }}</td>
<td class="r">{{ m.yield | default: "–" }}</td>
<td class="r">{{ m.ultimate | default: "–" }}</td>
<td class="r">{{ m.melting }}</td>
<td class="r">{% if m.insulator %}insulator{% else %}{{ m.resistivity }}{% endif %}</td>
<td class="r">{{ m.hardness | default: "–" }}</td>
<td class="r">{{ m.dbtt | default: "–" }}</td>
<td{% if m.fit %} title="{{ m.fit | escape }}"{% endif %}>{{ m.fit_short | default: "–" }}</td>
<td>{{ m.hugoniot | default: "–" }}</td>
<td class="r">{% if m.hall_petch %}{{ m.hall_petch }}{% if m.hall_petch_estimated %} est.{% endif %}{% else %}–{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

{% if noted.size > 0 %}
<details>
<summary>Notes on these entries</summary>
<ul>
{% for m in noted %}
<li><code>{{ m.key }}</code> {{ m.notes | escape }}{% if m.working %} Conventional working range {{ m.working }}{% if m.working_note %} ({{ m.working_note | escape }}){% endif %}.{% endif %}</li>
{% endfor %}
</ul>
</details>
{% endif %}

{% endfor %}

<h2 id="aliases">Aliases</h2>

<div class="prose">
<p>Short names accepted wherever a material is expected. Lookup ignores case.</p>
<table>
<thead><tr><th>Key</th><th>Aliases</th></tr></thead>
<tbody>
{% for m in mat.rows %}{% if m.aliases.size > 0 %}
<tr><td><code>{{ m.key }}</code></td><td>{% for a in m.aliases %}<code>{{ a | escape }}</code>{% unless forloop.last %} {% endunless %}{% endfor %}</td></tr>
{% endif %}{% endfor %}
</tbody>
</table>
</div>
