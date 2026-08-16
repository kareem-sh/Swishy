"""The phase vocabulary, loaded from config/phase_model.yaml.

Two layers, and the distinction matters everywhere downstream:

    CORE_STATES     what the live state machine tracks while frames arrive.
                    Four of them. Small on purpose -- every state is another
                    place a shot can get stuck and be lost.

    PHASE_ORDER     what the coaching report talks about. Seven of them,
                    assigned after the shot is captured by
                    feedback.phase_refiner, which can see the whole motion at
                    once and so finds boundaries by extrema rather than by
                    live thresholds.

`PHASE_LABELS` covers both vocabularies so display code can render whichever
it happens to hold without caring which layer produced it.

Nothing here is hardcoded. Editing config/phase_model.yaml is the supported
way to add, rename or remove a phase.
"""

from typing import Dict, List

from utils.config_loader import load_yaml

_MODEL = load_yaml("phase_model.yaml")

_DETECTOR = _MODEL.get("detector", {}) or {}
_ANALYSIS = _MODEL.get("analysis", {}) or {}

# ------------------------------------------------------------- layer 1: FSM
_STATES: Dict[str, dict] = _DETECTOR.get("states", {}) or {}

CORE_STATES: List[str] = list(_STATES.keys())
CORE_LABELS: Dict[str, str] = {
    key: (spec or {}).get("label", key.replace("_", " ").title())
    for key, spec in _STATES.items()
}
CORE_TRANSITIONS: Dict[str, List[str]] = {
    state: list(nxt or [])
    for state, nxt in (_DETECTOR.get("transitions", {}) or {}).items()
}

# The state a shot is anchored on. Reaching it is what proves a shot happened,
# so the tracker opens a candidate around it even if none was open at the time.
CORE_ANCHOR: str = _DETECTOR.get("anchor", "release")

CORE_REST: str = CORE_STATES[0] if CORE_STATES else "ready"

# States in which a shot is considered to be underway.
CORE_ACTIVE = frozenset(CORE_STATES) - {CORE_REST}

# -------------------------------------------------------- layer 2: analysis
PHASE_ORDER: List[str] = list(_ANALYSIS.get("order", []) or [])

_ANALYSIS_LABELS: Dict[str, str] = dict(_ANALYSIS.get("labels", {}) or {})

PHASE_LABELS: Dict[str, str] = {
    **CORE_LABELS,
    **{p: _ANALYSIS_LABELS.get(p, p.replace("_", " ").title()) for p in PHASE_ORDER},
}

# Analysis phases that carry coaching rules.
#
# Derived from biomechanics.yaml, NOT from position in the list. This was
# `PHASE_ORDER[1:]`, which happened to be right only while the first entry was
# the rule-less `ready_stance`: removing that entry would have silently started
# excluding `loading` -- the phase carrying the knee and hip rules -- with
# nothing to raise. A positional assumption about a config-driven list is a
# trap for whoever edits the config next.
_RULE_PHASES = {
    phase
    for rule in ((load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {}).values()
    for phase in (rule.get("phases") or [])
}
ACTIVE_PHASES = frozenset(p for p in PHASE_ORDER if p in _RULE_PHASES)


def phase_index(phase: str) -> int:
    """Position in the analysis timeline, or -1 if not an analysis phase."""
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return -1
