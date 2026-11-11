"""Feasibility status and claim-verdict vocabularies (single source of truth).

Every assessment in magform carries a :class:`Status`.  The claims ledger
(:mod:`magform.claims`) carries a :class:`Verdict`.  Both enums subclass
``str`` so they serialise to their value, and override ``__str__`` so that
f-strings print the value on every supported Python version (3.10+).

Status meanings
---------------
``state_of_the_art``  demonstrated today at the stated scale (cited benchmark).
``extrapolation``     beyond demonstrated values but no known physical barrier;
                      requires scaling of an existing technology.
``aspirational``      requires materials or technology that do not exist
                      (e.g. carbon-nanosheet laminates beyond any measured value).
``infeasible``        violates a physical limit within the model (e.g. no
                      finite elastic wall exists, superconductor quenches every
                      pulse, dwell time cannot be met by a pulse).
``not_applicable``    the subsystem was not evaluated for this input.
``assumed``           granted by an explicit premise of the analysis and not assessed
                      against today's technology (the thought-experiment label of
                      :mod:`magform.magnetar`); ignored by :func:`worst`/:func:`best`.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Status(str, Enum):
    """Feasibility label for a computed design quantity or process."""

    STATE_OF_THE_ART = "state_of_the_art"
    EXTRAPOLATION = "extrapolation"
    ASPIRATIONAL = "aspirational"
    INFEASIBLE = "infeasible"
    NOT_APPLICABLE = "not_applicable"
    ASSUMED = "assumed"

    def __str__(self) -> str:
        return self.value

    @property
    def rank(self) -> int:
        """Ordering used by :func:`worst`; NOT_APPLICABLE and ASSUMED rank lowest."""
        return _STATUS_RANK[self]

    @property
    def assessed(self) -> bool:
        """False for the two labels that carry no feasibility judgement."""
        return self not in (Status.NOT_APPLICABLE, Status.ASSUMED)

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``'state of the art'``."""
        return self.value.replace("_", " ")


_STATUS_RANK = {
    Status.ASSUMED: -2,
    Status.NOT_APPLICABLE: -1,
    Status.STATE_OF_THE_ART: 0,
    Status.EXTRAPOLATION: 1,
    Status.ASPIRATIONAL: 2,
    Status.INFEASIBLE: 3,
}


def worst(*statuses: Status | Iterable[Status]) -> Status:
    """Return the most severe status, ignoring ``NOT_APPLICABLE`` and ``ASSUMED``.

    Accepts individual statuses and/or iterables of statuses.  Returns
    ``NOT_APPLICABLE`` if nothing assessed was given.
    """
    flat: list[Status] = []
    for s in statuses:
        if isinstance(s, Status):
            flat.append(s)
        else:
            flat.extend(s)
    live = [s for s in flat if s.assessed]
    if not live:
        return Status.NOT_APPLICABLE
    return max(live, key=lambda s: s.rank)


def best(*statuses: Status | Iterable[Status]) -> Status:
    """Return the least severe assessed status (useful for 'any route works')."""
    flat: list[Status] = []
    for s in statuses:
        if isinstance(s, Status):
            flat.append(s)
        else:
            flat.extend(s)
    live = [s for s in flat if s.assessed]
    if not live:
        return Status.NOT_APPLICABLE
    return min(live, key=lambda s: s.rank)


class Verdict(str, Enum):
    """Verdict on a sentence of the original concept README (claims ledger)."""

    STATE_OF_THE_ART = "state_of_the_art"   # true with today's technology
    EXTRAPOLATION = "extrapolation"         # plausible by scaling known technology
    ASPIRATIONAL = "aspirational"           # needs materials/technology that do not exist
    INFEASIBLE = "infeasible"               # conflicts with physics as modelled
    CONTRADICTED = "contradicted"           # a factual statement that is false
    NOT_MODELLED = "not_modelled"           # outside what this toolkit can compute

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


def verdict_from_status(status: Status) -> Verdict:
    """Map a feasibility status onto the corresponding claim verdict."""
    return {
        Status.STATE_OF_THE_ART: Verdict.STATE_OF_THE_ART,
        Status.EXTRAPOLATION: Verdict.EXTRAPOLATION,
        Status.ASPIRATIONAL: Verdict.ASPIRATIONAL,
        Status.INFEASIBLE: Verdict.INFEASIBLE,
        Status.NOT_APPLICABLE: Verdict.NOT_MODELLED,
        Status.ASSUMED: Verdict.NOT_MODELLED,
    }[status]


class Confidence(str, Enum):
    """Provenance tag for data and models."""

    HANDBOOK = "handbook"              # handbook / datasheet / primary measurement
    REPRESENTATIVE = "representative"  # typical literature value, condition-dependent
    ESTIMATED = "estimated"            # derived, fitted or order-of-magnitude
    REPORTED = "reported"              # recent announcement, not yet peer reviewed

    def __str__(self) -> str:
        return self.value


__all__ = ["Status", "Verdict", "Confidence", "worst", "best", "verdict_from_status"]
