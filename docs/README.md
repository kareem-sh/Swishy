# Swichy Documentation Index

Read these in order if you are learning the system from scratch.

## Start here

| # | Document | What you will learn |
|---|----------|---------------------|
| **0** | [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md) | **All phases 1→5 — implementation map and roadmap** |
| 1 | [ARCHITECTURE.md](ARCHITECTURE.md) | Module design, separation of concerns |
| 2 | [PIPELINE.md](PIPELINE.md) | End-to-end data flow |
| 3 | [ANGLES_3D.md](ANGLES_3D.md) | Phase 1 — 3D vector math, world landmarks |
| 4 | [FILTERS.md](FILTERS.md) | Phase 2 — One Euro Filter |
| 5 | [VISIBILITY.md](VISIBILITY.md) | Phase 2 — Occlusion handling |
| 6 | [PHASE_DETECTION.md](PHASE_DETECTION.md) | Phase 3 — 8-phase FSM |
| 7 | [BIOMECHANICS.md](BIOMECHANICS.md) | Phase 4 — Rule engine |

| 8 | [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md) | Phase 5 — Shot score + coaching tips |
| 9 | [REPORTING.md](REPORTING.md) | Phase 5b — Detailed reports + key frames |
| 10 | [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | Phase 6 — Ball tracking + make/miss (planned) |

## Roadmap

| Document | Content |
|----------|---------|
| [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | Phase 5+ and commercial enhancements |

## Quick Reference — Full Pipeline

```
Camera → Pose → Filter → Visibility → 3D Angles → Features → Phases → Rules → Display
         [P1]   [P2]     [P2]         [P1]        [P3]      [P3]     [P4]
```

Entry point: [`pipeline.py`](../pipeline.py) → `ShotAnalysisPipeline.process_frame()`

## Development phase status

| Phase | Status |
|-------|--------|
| 1 — Foundation | Done |
| 2 — Filtering & visibility | Done |
| 3 — Phase detection | Done |
| 4 — Biomechanical rules | Done |
| 5 — Scoring & feedback | Done |
| 5b — Session reports | Done |
| 6 — Ball tracking & shot outcome | Planned — see [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) |
