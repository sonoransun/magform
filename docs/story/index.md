---
title: The case in brief
kicker: The case
doc: MF-00
description: A concept note proposes to forge, weld and synthesise materials with 1.6 to 4 GPa of magnetic pressure from a superconducting magnet. This is the short version of what the numbers say about it.
models: [claims, press]
---
{% assign n = site.data.n %}

The [concept note](../original-concept.md) under review proposes a new kind of press. A superconducting magnet, held together by carbon nanosheets, would produce "1,600 to over 4,000 MPa of magnetic pressure" and apply it to metal with no die and no moving parts. The note lists six uses: forging, shaping refractory metals cold, welding dissimilar alloys, making diamond in bulk, removing porosity from turbine parts, and containing railguns and fusion plasmas.

This site checks those statements one at a time. Each is turned into a calculation, run with cited material data, and compared with what has been built. The calculations are a Python package, `magform`, and every figure quoted here is produced by it.

## The answer

{% include segbar.html %}

Of {{ n.l_total }} statements, {{ n.l_holds }} hold today or would hold with more engineering, and {{ n.l_false }} are false or conflict with the physics. The short version has three parts.

**The pressure is real.** A field of {{ n.p_field_hi }} T exerts 4 GPa on a conductor it has not penetrated. Such a field has been made, at {{ n.m_pulsed_record }} T, in a bore {{ n.m_pulsed_bore }} mm across, for about a hundredth of a second.

**The proposed machine cannot supply it.** A magnetic field presses on metal only while it is changing, so the magnet has to be pulsed. A superconducting magnet cannot be pulsed like that, cannot reach the field, and cannot be made stiff enough. The magnets that do reach 100 T are copper coils driven in short pulses.

**Most of the applications do not want it.** The processes that work with magnetic pressure work well below the note's window. The ones the note adds need heat or time that a pulse cannot give.

## How statements are graded

Every result in the package carries one of four ratings, and the ledger adds two for statements of fact.

<div class="table-scroll">
<table>
<thead><tr><th>Rating</th><th>Meaning</th></tr></thead>
<tbody>
{% for v in site.data.claims.counts %}
<tr><td>{% include stamp.html v=v.verdict %}</td><td>{{ v.meaning }}</td></tr>
{% endfor %}
<tr><td>{% include stamp.html v="assumed" %}</td><td>Granted as a premise and not graded. Used only for the thought experiment in chapter 11.</td></tr>
</tbody>
</table>
</div>

The ratings are not opinions attached afterwards. Each comes out of a function that compares a computed quantity with a cited record or a material limit, and the comparison is printed with the result.

## The chapters

Chapters 1 to 3 are about the press itself. Chapters 4 to 9 take the note's six applications in its own order. Each opens with the passages under test and closes with the commands that reproduce its numbers.

{% include chapter-table.html %}

## Reading this as an engineer

If you would sooner run the numbers than read about them:

```console
$ pip install -e .
$ magform press --pressure 4000        # the reference press, end to end
$ magform claims                       # every verdict
```

The [explorer](../explorer.md) does the same in the browser for any pressure and bore. The [guide](../guide/getting-started.md) covers the package, and [models and validity](../reference/methods.md) lists every equation with its range.
