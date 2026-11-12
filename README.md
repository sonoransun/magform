# magform

An engineering review of "magnetic forging", written as a Python package and a documentation site.

Explore the site: <a href="https://sonoransun.github.io/magform/"><b>https://sonoransun.github.io/magform/</b></a>

A concept note proposed squeezing metal with 1,600 to 4,000 MPa of magnetic pressure from a
carbon-nanosheet-reinforced superconducting magnet, and listed six things that would follow:
die-less forging, cold forming of refractory metals, cold welding of dissimilar alloys, bulk
synthesis of diamond, pore-free turbine alloys, and housings for railguns and fusion magnets.
`magform` turns each of those statements into a calculation, runs it with cited material data,
and compares the result with what has been built. Every result carries a verdict.

The note is preserved unchanged in [`docs/archive/original-README.md`](docs/archive/original-README.md).

## What it finds

Thirty statements from the note are graded in a ledger (`magform claims`):

<!-- ledger:begin -->
| Verdict | Statements | Meaning |
|---|---:|---|
| state of the art | 6 | True of today's technology, under the conditions stated. |
| extrapolation | 6 | Beyond what has been shown, with no known barrier: a matter of scaling. |
| aspirational | 1 | Needs a material or technology that does not exist. |
| infeasible | 5 | Conflicts with physics as modelled. |
| contradicted | 10 | A statement of fact that the numbers show to be false. |
| not modelled | 2 | Outside what the package computes. |
<!-- ledger:end -->

In three sentences:

- **The pressure is real.** 4 GPa is the pressure of a 100.27 T field, and a magnet has reached
  100.75 T without destroying itself, in a bore about 10 mm across.
- **The proposed machine cannot supply it.** A field presses on metal only while it is changing,
  so the magnet must be pulsed. No superconducting magnet reaches the field, survives the pulse
  or can be made stiff enough. Pulsed copper coils are the route that could be built.
- **Most of the applications want something else.** Pulse welds form at 15 to 27 T and fail
  above that. Pores are healed by 172 MPa held hot for hours. Diamond needs heat and time as
  well as pressure.

One further premise is worked through as a thought experiment: a source in space that supplies
the field of a magnetar on demand. It is marked `assumed`, is never graded, and is kept apart
from the ledger (`magform magnetar`, and chapter 11 of the site).

## Quick start

Python 3.10 or later. No dependencies.

```console
$ python -m pip install -e .
$ magform field --pressure 4000           # the field behind a pressure, and where it sits
$ magform press --pressure 4000           # the reference press, end to end
$ magform weld al steel --pressure 100    # a weld that works
$ magform claims C02                      # one verdict, with its evidence
```

```python
import magform

magform.field_for_pressure(4e9)                        # 100.27 T
design = magform.design_press(magform.reference_spec(pressure=4e9))
design.status                                          # infeasible: the note's own magnet
design.best_architecture, design.best_status           # 'pulsed_resistive', extrapolation
print(design.workpiece.text())                         # any result prints itself, with units
```

Pressures are in MPa on the command line and in pascals in Python, where everything is SI.
Every command takes `--json`.

## The documentation site

`docs/` is a Jekyll site that GitHub Pages builds as it stands. To publish it, open the
repository's **Settings → Pages**, choose **Deploy from a branch**, and select the `main`
branch and the `/docs` folder. No workflow or build step is needed.

The site has eleven short chapters (the press, its power supply, the six applications, a
verdict, and the magnetar thought experiment), the claims ledger with filters, an explorer
that sizes a press in the browser, a guide and reference pages.

Its numbers, tables and charts are not typed by hand. One script writes them from the package:

```console
$ python tools/build_docs.py            # rewrite docs/_data/ and docs/_includes/generated/
$ python tools/build_docs.py --check    # exit 1 if they are out of date
```

To preview locally (Ruby and Bundler required):

```console
$ cd docs
$ bundle install
$ bundle exec jekyll serve
```

## What is in the package

| Area | Modules |
|---|---|
| Foundations | `constants`, `errors`, `status`, `benchmarks`, `materials` |
| The press | `magnetics`, `fieldmap`, `conductor`, `pulse`, `press` |
| The workpiece | `eddy`, `plasticity`, `microstructure`, `shock` |
| The applications | `welding`, `synthesis`, `porosity`, `containment` |
| Beyond the note | `magnetar` |
| Ledger and command line | `claims`, `cli` |

Each public function's docstring gives its equation, units, range of validity and a reference.
Each `assess_...` function returns a dataclass with a `status`, a `margin`, `warnings` and
`notes`, and the status is always the result of comparing a computed quantity with a cited
record or a material limit.

## Tests

```console
$ python -m pip install -e ".[dev]"
$ python -m pytest
```

The tests pin the worked numbers in the docstrings, the verdict of every ledger entry, and the
documentation: that the generated data are current, that every command and code example the
pages quote actually runs, and (when Node.js is installed) that the browser explorer computes
the same values as the package.

## Limits

These are engineering estimates built from closed-form relations and one-dimensional models.
They say which side of a limit a design falls on and by roughly how much. They are not a
finite-element analysis of a real coil, they do not model fracture, fatigue or cost, and a
verdict of "extrapolation" means only that no physical barrier was found. Each model's range
of validity is listed on the site under *Models and validity*.

Benchmark records were checked against public sources in November 2026. Two are institutional
announcements rather than journal papers and are tagged `reported`: the 35.6 T
all-superconducting magnet (January 2026) and the 48.7 T superconducting test coil (August 2025).

## Repository layout

```
magform/     the package (standard library only)
tests/       pytest suite, one file per module, plus the documentation checks
tools/       build_docs.py and the generators it runs
docs/        the Jekyll site; _data/ and _includes/generated/ are written by tools/build_docs.py
examples/    short runnable scripts
```
