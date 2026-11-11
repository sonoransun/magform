---
title: Models and validity
kicker: Reference
doc: MF-R4
description: What each model computes, the relations it uses, and where those relations stop being valid.
---
{% assign me = site.data.methods %}

The package is a set of small modules, each with one job. This page gives, for every model, the question it answers, the relations it computes, the limits of those relations and the functions to call. The relations here are summaries. The docstring of each function carries the full equation, its units, its validity and its source.

There are {{ me.count }} model modules with {{ me.callable_count }} public functions and classes. Each has its own test file under `tests/`, and the tests pin the worked numbers quoted in the docstrings.

## Conventions

- All quantities are SI unless a name says otherwise. Angles are in radians.
- Every result is a frozen record that renders as text, a dictionary or strict JSON. Units are attached to the fields.
- A non-physical input raises `InputError`. An unknown material raises `UnknownMaterialError` with suggestions.
- An assessment carries a `status`, a `margin` that is at least 1 when the requirement is met, and the `warnings` and `notes` that explain the status.
- The package uses only the Python standard library.

## Status vocabulary

Every assessment returns one of these labels. A roll-up takes the worst label among the parts that were assessed.

<table class="legend-table">
<thead><tr><th>Status</th><th>Meaning</th></tr></thead>
<tbody>
{% for s in me.statuses %}
<tr><td>{% include stamp.html v=s.key %}</td><td>{{ s.meaning | escape }}</td></tr>
{% endfor %}
</tbody>
</table>

The [claims ledger]({{ '/ledger.html' | relative_url }}) grades sentences from the concept note. It uses two more labels, for a statement that is false and for one the models cannot test.

<table class="legend-table">
<thead><tr><th>Verdict</th><th>Meaning</th></tr></thead>
<tbody>
{% for v in me.verdicts %}
<tr><td>{% include stamp.html v=v.key %}</td><td>{{ v.meaning | escape }}</td></tr>
{% endfor %}
</tbody>
</table>

## The models at a glance

<table>
<thead><tr><th>Module</th><th>Answers</th><th class="r">Sources</th><th>Tests</th></tr></thead>
<tbody>
{% for m in me.modules %}
<tr>
<td><a href="#{{ m.module }}"><code>{{ m.module }}</code></a></td>
<td>{% if m.answers != "" %}{{ m.answers | escape }}{% else %}{{ m.title | escape }}.{% endif %}</td>
<td class="r">{% if m.reference_count > 0 %}{{ m.reference_count }}{% else %}–{% endif %}</td>
<td class="small muted">{% if m.has_tests %}<code>test_{{ m.module }}.py</code>{% else %}with the foundation tests{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>

{% for m in me.modules %}

<h2 id="{{ m.module }}">magform.{{ m.module }}</h2>

<p>{{ m.title | escape }}.{% if m.answers != "" %} {{ m.answers | escape }}{% endif %}{% if m.chapter %} <a href="{{ m.chapter | relative_url }}">Read it in context</a>.{% endif %}</p>

{% if m.relations.size > 0 %}
<h4>Relations</h4>
<ul>
{% for r in m.relations %}
<li>{{ r | escape }}</li>
{% endfor %}
</ul>
{% endif %}

{% if m.limits.size > 0 %}
<h4>Limits</h4>
<ul>
{% for l in m.limits %}
<li>{{ l | escape }}</li>
{% endfor %}
</ul>
{% endif %}

{% if m.entry_points.size > 0 %}
<h4>Entry points</h4>
<table>
<thead><tr><th>Name</th><th>What it gives</th></tr></thead>
<tbody>
{% for e in m.entry_points %}
<tr><td><code>{{ e.name }}</code></td><td>{{ e.summary | escape }}</td></tr>
{% endfor %}
</tbody>
</table>
{% endif %}

<details>
<summary>All {{ m.callable_count }} public functions and classes</summary>
<ul>
{% for e in m.all_entry_points %}
<li><code>{% if e.signature != "" %}{{ e.signature | escape }}{% else %}{{ e.name }}{% endif %}</code><br>{{ e.summary | escape }}</li>
{% endfor %}
</ul>
</details>

{% endfor %}

## Supporting modules

<table>
<thead><tr><th>Module</th><th>Purpose</th></tr></thead>
<tbody>
{% for s in me.support %}
<tr><td><code>{{ s.module }}</code></td><td>{{ s.title | escape }}.</td></tr>
{% endfor %}
</tbody>
</table>

## Check it yourself

Every relation on this page can be read from the code it summarises.

```python
import magform
from magform import magnetics

help(magnetics.max_hoop_stress_ratio)      # equation, units, validity and reference
magnetics.REFERENCES["moon1982"]           # the citation behind the virial bound
```
