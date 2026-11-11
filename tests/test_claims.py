"""Tests for magform.claims: the ledger of the concept note's claims."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

import magform
from magform import claims
from magform.claims import CLAIMS, Claim, ClaimResult
from magform.errors import InputError
from magform.status import Verdict

ARCHIVE = pathlib.Path(__file__).resolve().parents[1] / "docs" / "archive" / "original-README.md"


@pytest.fixture(scope="module")
def results():
    return claims.evaluate_all()


@pytest.fixture(scope="module")
def by_id(results):
    return {r.id: r for r in results}


def _flat(text: str) -> str:
    return " ".join(text.split())


# ------------------------------------------------------------------ the claims themselves
def test_every_quote_is_verbatim_from_the_archived_note():
    note = _flat(ARCHIVE.read_text(encoding="utf-8"))
    for c in CLAIMS:
        assert _flat(c.quote) in note, f"{c.id}: quote is not in the archived concept note"


def test_claim_table_is_complete_and_ordered():
    ids = [c.id for c in CLAIMS]
    assert ids == [f"C{i:02d}" for i in range(1, 29)] + ["Q1", "Q2"]
    assert len(set(ids)) == 30
    sections = [key for key, _ in claims.SECTIONS]
    assert [c.section for c in CLAIMS] == sorted((c.section for c in CLAIMS), key=sections.index)   # note order
    assert {c.section for c in CLAIMS} == set(sections)
    for c in CLAIMS:
        assert isinstance(c, Claim) and c.topic and c.chapter in claims.CHAPTERS
        assert c.kind == ("question" if c.id.startswith("Q") else "claim")
    # every sentence-level claim of the note's six applications is covered: each bullet has an entry
    note = ARCHIVE.read_text(encoding="utf-8")
    assert note.count("• The Edge:") == 6 and note.count("• The Process:") + note.count("• The Application:") == 6


def test_get_claim_and_errors():
    assert claims.get_claim("c08").id == "C08" and claims.get_claim(" Q1 ").kind == "question"
    for bad in ("C00", "C29", "Q3", "", "pressure"):
        with pytest.raises(InputError):
            claims.get_claim(bad)
    with pytest.raises(InputError):
        claims.evaluate("C99")


# ------------------------------------------------------------------ results
def test_every_claim_evaluates_with_evidence(results):
    assert len(results) == 30 and [r.id for r in results] == [c.id for c in CLAIMS]
    for r, c in zip(results, CLAIMS):
        assert isinstance(r, ClaimResult) and isinstance(r.verdict, Verdict)
        assert r.quote == c.quote and r.topic == c.topic and r.section == c.section
        assert r.section_title == dict(claims.SECTIONS)[c.section]
        assert len(r.finding) > 80 and r.finding[0].isupper() and r.finding.rstrip().endswith(".")
        assert r.evidence and all(isinstance(e, str) and e for e in r.evidence)
        assert any(ch.isdigit() for e in r.evidence for ch in e) or r.verdict is Verdict.INFEASIBLE
        assert r.models and all(m.startswith("magform.") for m in r.models)
        assert r.chapter.startswith("/story/") and r.chapter.endswith(".html")
        assert "nan" not in r.finding.lower().split() and "inf " not in r.finding
        for e in r.evidence:
            assert " nan" not in e and "None" not in e, (r.id, e)


def test_models_named_in_the_ledger_exist(results):
    for r in results:
        for dotted in r.models:
            parts = dotted.split(".")
            assert parts[0] == "magform"
            obj = getattr(magform, parts[1])
            for name in parts[2:]:
                obj = getattr(obj, name)
            assert obj is not None, dotted


def test_verdict_counts(results):
    counts = claims.summary(results)
    assert counts == claims.summary()
    assert sum(counts.values()) == 30 and list(counts) == [v.value for v in Verdict]
    assert counts == {"state_of_the_art": 6, "extrapolation": 6, "aspirational": 1, "infeasible": 5,
                      "contradicted": 10, "not_modelled": 2}
    grouped = claims.by_section(results)
    assert list(grouped) == [key for key, _ in claims.SECTIONS]
    assert sum(len(v) for v in grouped.values()) == 30 and len(grouped["questions"]) == 2


@pytest.mark.parametrize("claim_id, verdict", [
    ("C01", Verdict.STATE_OF_THE_ART),   # the pressure exists (pulsed, 10 mm bore)
    ("C02", Verdict.INFEASIBLE),         # not from a superconducting magnet
    ("C03", Verdict.CONTRADICTED), ("C04", Verdict.CONTRADICTED), ("C05", Verdict.CONTRADICTED),
    ("C06", Verdict.CONTRADICTED), ("C07", Verdict.EXTRAPOLATION), ("C08", Verdict.CONTRADICTED),
    ("C09", Verdict.CONTRADICTED), ("C10", Verdict.CONTRADICTED), ("C11", Verdict.EXTRAPOLATION),
    ("C12", Verdict.EXTRAPOLATION), ("C13", Verdict.NOT_MODELLED), ("C14", Verdict.STATE_OF_THE_ART),
    ("C15", Verdict.CONTRADICTED), ("C16", Verdict.EXTRAPOLATION), ("C17", Verdict.STATE_OF_THE_ART),
    ("C18", Verdict.STATE_OF_THE_ART), ("C19", Verdict.CONTRADICTED), ("C20", Verdict.INFEASIBLE),
    ("C21", Verdict.INFEASIBLE), ("C22", Verdict.INFEASIBLE), ("C23", Verdict.STATE_OF_THE_ART),
    ("C24", Verdict.INFEASIBLE), ("C25", Verdict.NOT_MODELLED), ("C26", Verdict.STATE_OF_THE_ART),
    ("C27", Verdict.ASPIRATIONAL), ("C28", Verdict.CONTRADICTED), ("Q1", Verdict.EXTRAPOLATION),
    ("Q2", Verdict.EXTRAPOLATION),
])
def test_verdicts_are_pinned(by_id, claim_id, verdict):
    assert by_id[claim_id].verdict is verdict


def test_headline_numbers_appear_in_the_findings(by_id):
    def has(claim_id: str, *fragments: str) -> None:
        text = by_id[claim_id].finding + " " + " ".join(by_id[claim_id].evidence)
        for fragment in fragments:
            assert fragment in text, (claim_id, fragment)

    has("C01", "63.41 T", "100.27 T", "100.75 T", "10 mm")
    has("C02", "35.6 T", "48.7 T", "0 Pa", "pulsed_resistive")
    has("C03", "2.5 GPa", "6 GPa")
    has("C04", "0.29 eV", "3.49 eV", "0.40 eV")
    has("C06", "58 T", "96 T", "149 T")
    has("C07", "1260 m/s", "50 T")
    has("C08", "5.7 times", "0.970", "43 %")
    has("C09", "+464 K", "+1160 K")
    has("C10", "1200-1600 C", "1000-1300 C")
    has("C11", "2.05 GPa", "186 C")
    has("C12", "3.10", "4e-89")
    has("C15", "374", "649", "1540", "2434", "41 times")
    has("C18", "0.65 nm", "44.7 um")
    has("C20", "4.5 GPa", "7 of 7")
    has("C21", "2113 m/s", "22.3 GPa", "52.2 GPa")
    has("C24", "4.95 GPa", "1260 C", "172 MPa")
    has("C27", "10.4 GPa")
    has("C28", "42", "184 MPa")
    has("Q1", "99.5 MJ", "117 V")
    has("Q2", "1.65 um", "+189 K")


def test_results_are_cached_and_deterministic(results):
    assert claims.evaluate("C08") is claims.evaluate("c08")
    assert claims.evaluate_all() == results
    again = json.loads(json.dumps([r.as_dict() for r in results]))
    assert [a["verdict"] for a in again] == [r.verdict.value for r in results]


def test_findings_are_plain_text(results):
    # no markup, no model jargon leaking into the prose
    for r in results:
        assert not re.search(r"[<>{}]|\\\\", r.finding), r.id
        assert "state_of_the_art" not in r.finding and "_" not in r.finding.replace("pulsed_resistive", ""), r.id


def test_json_text_and_package_export(by_id):
    r = by_id["C08"]
    data = json.loads(r.json())
    assert data["verdict"] == "contradicted" and data["id"] == "C08" and isinstance(data["evidence"], list)
    assert "verdict = contradicted" in r.text()
    assert magform.evaluate_claims is claims.evaluate_all
    assert claims.REFERENCES and set(claims.__all__) <= set(dir(claims))
    for name in claims.__all__:
        obj = getattr(claims, name)
        if callable(obj):
            assert obj.__doc__ and "Reference" in obj.__doc__, name
