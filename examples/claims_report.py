"""Print the ledger grouped by verdict, then one entry in full.

    python examples/claims_report.py
"""

from magform import claims

results = claims.evaluate_all()
counts = claims.summary(results)

for verdict, count in counts.items():
    ids = " ".join(r.id for r in results if r.verdict.value == verdict)
    print(f"{verdict.replace('_', ' '):17s} {count:2d}  {ids}")

entry = claims.evaluate("C15")
print(f'\n{entry.id}  "{entry.quote}"')
print(f"{entry.verdict.value}: {entry.finding}")
for line in entry.evidence:
    print("  -", line)
