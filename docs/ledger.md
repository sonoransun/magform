---
title: Claims ledger
kicker: Tools
doc: MF-L1
description: Every checkable statement in the concept note, quoted as written, with its verdict, the evidence behind it and the functions that produced the evidence.
wide: true
---
{% assign n = site.data.n %}
{% assign cl = site.data.claims %}

The note makes {{ n.l_claims }} statements that can be tested and asks {{ n.l_questions }} questions. Each entry below quotes the note, gives a verdict and a finding in plain words, lists the numbers the verdict rests on, and names the functions in the package that computed them. The same ledger is available from the command line as `magform claims`.

{% include segbar.html %}

<div class="filters" data-ledger-filters role="group" aria-label="Filter by verdict">
  <button class="chip" type="button" data-verdict="all" aria-pressed="true">All <span class="chip__n">{{ cl.total }}</span></button>
  {% for v in cl.counts %}{% if v.count > 0 %}<button class="chip" type="button" data-verdict="{{ v.verdict }}" aria-pressed="false">{{ v.verdict | replace: "_", " " | capitalize }} <span class="chip__n">{{ v.count }}</span></button>{% endif %}{% endfor %}
</div>
<p class="small muted" data-ledger-count role="status" style="margin-top:0.6rem">Showing all {{ cl.total }} entries.</p>

{% for sec in cl.sections %}
<section class="ledger__section" data-section="{{ sec.key }}">
<h2 id="{{ sec.key }}">{{ sec.title }}</h2>
<div class="ledger">
{% for id in sec.ids %}{% include claim.html id=id %}{% endfor %}
</div>
</section>
{% endfor %}

## Reproduce it

```console
$ magform claims                          # the table of verdicts
$ magform claims C24                      # one entry with its evidence
$ magform claims --verdict contradicted   # every entry with that verdict
$ magform claims --json                   # everything, for a script
```

```python
from magform import claims

result = claims.evaluate("C15")
result.verdict, result.finding
for line in result.evidence:
    print(line)
```
