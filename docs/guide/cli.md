---
title: Command line
kicker: Guide
doc: MF-G2
description: Fourteen commands, one per family of models. Each prints a report with a verdict, takes --json, and exits non-zero on bad input.
wide: true
---
{% assign n = site.data.n %}
{% assign ex = site.data.cli.examples %}

The `magform` command has one sub-command per family of models. Pressures are in MPa, fields in tesla, lengths in metres, times in seconds and temperatures in kelvin. Every command accepts `--json` for strict JSON, and `-h` for its options. The output shown on this page is real: it is captured when the site data is built.

## Commands

<div class="table-scroll">
<table>
<thead><tr><th>Command</th><th>What it answers</th><th>Options</th></tr></thead>
<tbody>
{% for c in site.data.cli.commands %}
<tr><td><code>magform {{ c.name }}</code></td><td>{{ c.help }}</td><td class="small muted">{% for o in c.options %}{% unless o.flag == "--json" %}<code>{{ o.flag }}</code>{% unless forloop.last %} {% endunless %}{% endunless %}{% endfor %}</td></tr>
{% endfor %}
</tbody>
</table>
</div>

Status marks in text output: `[ok]` state of the art, `[~]` extrapolation, `[*]` aspirational, `[x]` infeasible, `[?]` assumed.

## The press

`press` sizes a coil for a pressure and grades the concept note's superconducting magnet against it. With no geometry it uses the reference press.

```console
$ {{ ex.press.command }}
{{ ex.press.output }}
```

`architectures` compares six ways of building the same press. Here the workpiece is a tube with a 2 mm wall, which demands a much faster pulse.

```console
$ {{ ex.architectures.command }}
{{ ex.architectures.output }}
  ... {{ ex.architectures.truncated }} more lines: one paragraph per architecture
```

`pulse` sizes the capacitor bank for a coil and a rise time.

```console
$ {{ ex.pulse.command }}
{{ ex.pulse.output }}
  ... {{ ex.pulse.truncated }} more lines
```

## The applications

```console
$ {{ ex.forge.command }}
{{ ex.forge.output }}
  ... {{ ex.forge.truncated }} more lines
```

```console
$ {{ ex.weld.command }}
{{ ex.weld.output }}
  ... {{ ex.weld.truncated }} more lines
```

```console
$ {{ ex.synth.command }}
{{ ex.synth.output }}
  ... {{ ex.synth.truncated }} more lines: one paragraph per route
```

```console
$ {{ ex.pores.command }}
{{ ex.pores.output }}
  ... {{ ex.pores.truncated }} more lines
```

```console
$ {{ ex.contain.command }}
{{ ex.contain.output }}
```

## The ledger

`claims` prints the verdict table. Name one or more entries to see their findings and evidence.

```console
$ {{ ex.claims.command }}
{{ ex.claims.output }}
```

## The magnetar mode

`magnetar` runs the thought experiment of [chapter 11](../story/magnetar.md). The first line of its output says that the source is assumed.

```console
$ {{ ex.magnetar.command }}
{{ ex.magnetar.output }}
  ... {{ ex.magnetar.truncated }} more lines
```

## In scripts

Text output is for reading. For anything else use `--json`, which prints the same dataclass the Python API returns. Numbers that are not finite become `null`.

```console
$ magform press --pressure 1600 --json > press.json
$ magform claims --json > ledger.json
```

Exit status is 0 on success and 2 on bad input, with the reason on standard error:

```console
$ magform forge unobtainium --pressure 2000
magform: unknown material 'unobtainium'; did you mean niobium, titanium, aluminium, tantalum?
```
