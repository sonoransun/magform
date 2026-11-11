---
title: Getting started
kicker: Guide
doc: MF-G1
description: Install the package, run one command, and learn to read a result. Ten minutes, no dependencies.
---
{% assign n = site.data.n %}

`magform` is a Python package with no dependencies beyond the standard library. It needs Python 3.10 or later.

## Install

From a checkout of the repository:

```console
$ python -m pip install -e .
$ magform --version
```

To run the tests as well, install the `dev` extra, which adds pytest:

```console
$ python -m pip install -e ".[dev]"
$ python -m pytest
```

## One command

Ask what field makes 4 GPa and where that pressure sits:

```console
$ {{ site.data.cli.examples.field.command }}
{{ site.data.cli.examples.field.output }}
```

Three things in that output recur everywhere. Pressures on the command line are in megapascals, because that is how the concept note states them. Every result carries a status in square brackets. And the last line is a note from the model about something the bare number hides.

## The same thing in Python

In Python everything is in SI units: pascals, tesla, metres, seconds, kelvin.

```python
import magform

field = magform.field_for_pressure(4e9)             # 100.27 T
magform.magnetic_pressure(field) / 1e9              # 4.0 GPa

design = magform.design_press(magform.reference_spec(pressure=4e9))
design.status                                       # Status.INFEASIBLE: the note's own magnet
design.best_architecture, design.best_status        # 'pulsed_resistive', Status.EXTRAPOLATION
```

`reference_spec` is the press used throughout this site: a 100 mm bore, 0.5 m long, with a 0.2 m winding, around a solid copper billet.

## Reading a result

Every assessment in the package is a frozen dataclass with the same four fields in front of its own.

| Field | What it holds |
|---|---|
| `status` | One of `state_of_the_art`, `extrapolation`, `aspirational`, `infeasible`, plus `not_applicable` and `assumed`. |
| `margin` | A number above 1 when there is room to spare and below 1 when there is not. What it measures is stated in the class docstring. |
| `warnings` | Things that drove the status down, in plain words. |
| `notes` | Context: which model was used, what it was compared with, what it ignores. |

And three methods:

```python
from magform import welding

weld = welding.assess_weld("aluminum_6061", "steel_1018", 1e-3, 2e-3, 0.1e9)
weld.status.label              # 'state of the art'
print(weld.text())             # a readable report with units
data = weld.as_dict()          # plain dicts, lists and numbers
text = weld.json()             # strict JSON: no NaN, no Infinity
```

The status is never set by hand. Each `assess_` function compares a computed quantity with a cited record or a material limit, takes the worst of its checks, and writes the reason into `warnings`. The statuses are ordered, and `magform.status.worst` and `best` combine them.

## Materials

Materials are looked up by key or by a common alias. Each record carries its sources and a confidence level.

```python
from magform import materials

w = materials.get("tungsten")                 # aliases work: materials.get("W")
w.yield_strength / 1e9, w.melting_point       # 1.3 GPa, 3695 K
w.confidence.value                            # how far to trust the record
[m.key for m in materials.list_materials("metal")][:4]
```

The full registry is on the [materials](../reference/materials.md) page.

## When the input is wrong

Bad input raises `magform.errors.InputError`, a subclass of both `MagformError` and `ValueError`, with a message that says what was expected. The command line turns it into exit status 2.

```python
from magform import errors, magnetics

try:
    magnetics.Solenoid(inner_radius=0.05, outer_radius=0.04, length=0.5)
except errors.InputError as problem:
    print(problem)             # outer_radius (0.04) must exceed inner_radius (0.05)
```

## Where to go next

- [The command line](cli.md) lists all {{ n.g_commands }} commands with real output.
- [A worked press design](worked-design.md) follows one design from pressure to power supply.
- [The API map](api.md) lists every public function with a one-line summary.
- [Models and validity](../reference/methods.md) gives each model's equations and limits.

## Rebuilding this site

Every number, table and chart on the site is written by one script from the package. After changing a model or a material record:

```console
$ python tools/build_docs.py            # rewrite docs/_data and the generated includes
$ python tools/build_docs.py --check    # exit 1 if they are out of date (the tests run this)
```

The site itself is plain Jekyll and builds on GitHub Pages from the `docs/` folder with no extra configuration. To preview it locally you need Ruby and Bundler:

```console
$ cd docs
$ bundle install
$ bundle exec jekyll serve
```
