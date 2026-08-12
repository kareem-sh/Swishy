# Shot phases in chronological order during a jump shot.

PHASE_ORDER = [
    "ready_stance",
    "loading",
    "knee_flexion",
    "ball_lift",
    "jump",
    "release",
    "follow_through",
    "landing",
]

PHASE_LABELS = {
    "ready_stance": "Ready Stance",
    "loading": "Loading",
    "knee_flexion": "Knee Flexion",
    "ball_lift": "Ball Lift",
    "jump": "Jump",
    "release": "Release",
    "follow_through": "Follow-Through",
    "landing": "Landing",
}

# Valid FSM transitions (current -> next)
TRANSITIONS = {
    "ready_stance": ["loading", "ball_lift"],
    "loading": ["knee_flexion", "ball_lift"],
    "knee_flexion": ["ball_lift"],
    "ball_lift": ["jump", "release"],
    "jump": ["release"],
    "release": ["follow_through"],
    "follow_through": ["landing"],
    "landing": ["ready_stance"],
}
