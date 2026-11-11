---
title: References
kicker: Reference
doc: MF-R3
description: The sources behind the models. Every function names the work it rests on, and this page collects them.
---
{% assign refs = site.data.references %}

Each model cites its sources by a short key in its docstring, and each module keeps the full citations in a `REFERENCES` table. This page is built from those tables. It lists {{ refs.unique }} works, cited {{ refs.count }} times across {{ refs.module_count }} modules.

The first list merges citations of the same work and trims them to the bibliographic entry. The second gives each module's table as written, with the section or figure it relies on.

```python
from magform import welding

welding.REFERENCES["wittman_1973"]    # source of the upper weld limit
```

## Bibliography

<ul>
{% for w in refs.bibliography %}
<li><strong>{{ w.label | escape }}</strong><br>{{ w.text | escape }} <span class="mono" style="color: var(--ink-3)">{{ w.modules | join: " · " }}</span></li>
{% endfor %}
</ul>

## By module

{% for m in refs.modules %}

<h3 id="{{ m.module }}">magform.{{ m.module }}</h3>

<table>
<thead><tr><th>Key</th><th>Citation</th></tr></thead>
<tbody>
{% for e in m.entries %}
<tr><td><code>{{ e.key | escape }}</code></td><td>{{ e.text | escape }}</td></tr>
{% endfor %}
</tbody>
</table>

{% endfor %}
