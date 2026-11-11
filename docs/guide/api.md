---
title: API map
kicker: Guide
doc: MF-G4
description: Every public class and function in the package, grouped by what it is for, with the first sentence of its docstring. Generated from the code.
wide: true
---
{% assign n = site.data.n %}

The package has {{ n.g_modules }} modules with {{ n.g_functions }} public functions and {{ n.g_classes }} classes. This page is generated from their docstrings, so it cannot fall behind the code. Each docstring in the source goes on to give the equation, the units, the range of validity and a reference.

## The short list

The names most people need are importable from the top level:

<p>{% for name in site.data.api_top %}<code>magform.{{ name }}</code>{% unless forloop.last %}, {% endunless %}{% endfor %}</p>

```python
import magform

magform.field_for_pressure(4e9)                  # T
magform.design_press(magform.reference_spec())   # the whole press, graded
magform.evaluate_claims()                        # the ledger
magform.get_material("tungsten")
```

Everything else lives in the module for its subject, for example `magform.welding.assess_weld`.

## Conventions

- **Units** are SI throughout: Pa, T, m, s, K, A, V, J, W, kg.
- **Materials** are passed as a registry key, an alias, or a `Material` object.
- **`assess_...` functions** return a dataclass with `status`, `margin`, `warnings` and `notes` in front of its own fields.
- **Results** have `.text()`, `.as_dict()` and `.json()`.
- **Bad input** raises `magform.errors.InputError`. A calculation that cannot converge raises `ConvergenceError`. Both derive from `MagformError`.
- **References**: each module has a `REFERENCES` dictionary of full citations, collected on the [references](../reference/references.md) page.

{% for g in site.data.api %}
## {{ g.title }}

{% for m in g.modules %}
<h3 id="{{ m.name }}"><code>magform.{{ m.name }}</code></h3>
<p>{{ m.summary }}</p>
{% if m.classes.size > 0 %}<p class="small"><span class="label">Classes</span> {% for c in m.classes %}<code title="{{ c.summary | escape }}">{{ c.name }}</code>{% unless forloop.last %} {% endunless %}{% endfor %}</p>{% endif %}
{% if m.functions.size > 0 %}
<div class="table-scroll">
<table class="api">
<tbody>
{% for f in m.functions %}<tr><td><code>{{ f.name }}</code></td><td class="small">{{ f.summary | escape }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
{% endif %}
{% endfor %}
{% endfor %}
