# Swichy — Evidence Map (`resources.md`)

**Single source of truth for every research claim in Swichy.**
Last updated: 9 August 2026 · Branch: `pose-edits`

**Project framing (v2):** Swichy is an **AI basketball shooting coach for practice**. It teaches technique and consistency. It is **not** a game-analytics system, **not** a defender/tactical analyser, and **not** a make/miss predictor.

---

## 0. How to read this file

Every claim carries an evidence class:

| Class | Meaning |
|---|---|
| `DIRECTLY SUPPORTED` | Multiple independent, indexed sources agree; effect measured, not inferred |
| `SUPPORTED WITH LIMITATIONS` | Real evidence exists but is single-source, low-powered, or contested |
| `INFERRED` | Follows logically from supported findings but not measured directly |
| `ENGINEERING DECISION` | A threshold or design choice we made. **Not** a literature-derived norm |
| `NOT SUPPORTED` | Evidence is absent, null, or self-contradictory. Must not drive scoring |

Verification status:

| Marker | Meaning |
|---|---|
| ✅ FULL TEXT VERIFIED | Numbers read from the paper's own results tables |
| ⚠️ ABSTRACT ONLY | Numbers from publisher abstract; full text not retrieved |
| ❌ UNVERIFIED | Citation metadata incomplete — must not be cited until resolved |

---

# PART A — CORE SOURCES

## A1. Cabarkapa et al. 2022 — Distance & Proficiency (professional males)

- **Authors:** Cabarkapa, D., Cabarkapa, D.V., Philipp, N.M., Eserhaut, D.A., Downey, G.G., Fry, A.C.
- **Year:** 2022 · **Journal:** *Journal of Functional Morphology and Kinesiology* 7(4):78
- **DOI:** 10.3390/jfmk7040078 · **PMC:** PMC9590067
- **Source type:** Primary empirical · **Evidence tier:** Tier 1 (peer-reviewed, Scopus + PubMed/PMC; WoS **ESCI only**)
- **Verification:** ✅ FULL TEXT VERIFIED

### Why we trust it
n=10 **professional** players (rare — most studies use recreational), 150 shots, 120 fps, explicit variable definitions quoted verbatim from the paper.

### What we take from it
Complete angle matrix by shot distance (prep phase): knee 121.5±8.8 / 116.7±7.4 / 112.5±7.4°; hip 143.9±8.4 / 141.1±8.1 / 135.5±8.4°; ankle 61.2 / 61.5 / 58.5°; elbow 61.9±13.7 / 58.2±16.6 / 63.6±21.6°; shoulder 78.9±26.9 / 72.7±26.4 / 65.8±31.4°. Vertical displacement 15.3±5.1 cm (FT) vs 26.9±5.6 / 31.2±7.3 cm (jump shots).

### Exact Swichy relevance
- Distance-specific knee/hip targets (`config/biomechanics.yaml`)
- **Set-vs-jump auto-classification via vertical displacement**
- Source of the verbatim variable definitions Swichy must match

### Player-height relevance
**Directly relevant.** This paper normalises release height and elbow height by **body height** (ratios, not metres). This is our primary literature precedent for height normalisation.

### What we DO NOT take from it
- Its "release angle" is **arm elevation**, *not* ball launch angle — see §D1. Never conflate.
- No proficiency differences were significant → do not cite for proficiency claims.

### Limitations
n=10. **Single 2D camera** — so it is *not* a ground-truth reference for validating a 2D system. Shoulder SD ±26.9–31.4° (≈40% of mean).

### Used in
`research.md` §2.1–2.4 · `config/biomechanics.yaml` (knee, hip, elbow rules) · set/jump classifier design

---

## A2. Cabarkapa et al. 2023a — Proficient Free-Throw Shooters (markerless 3D)

- **Authors:** Cabarkapa, D., Cabarkapa, D.V., Miller, J.D., Templin, T.T., Frazer, L.L., Nicolella, D.P., Fry, A.C.
- **Year:** 2023 · **Journal:** *Frontiers in Sports and Active Living* 5:1208915
- **DOI:** 10.3389/fspor.2023.1208915 · **PMC:** PMC10436204
- **Evidence tier:** Tier 2 (Scopus + PMC confirmed directly; WoS **ESCI only**)
- **Verification:** ✅ FULL TEXT VERIFIED

### Why we trust it
n=34, **3D markerless** (9 cameras, 120 Hz) — larger sample and better instrumentation than the 2D studies.

### What we take from it
Release height (÷ body height) **1.12 (non-prof) vs 1.17 (prof), p=0.010** — the only significant discriminator. Release angle p=0.179 (n.s.). Knee prep 107.3±14.9 vs 113.3±9.1, p=0.183 (n.s.). Elbow prep 78.0±27.4 vs 85.1±20.6, **p=0.471 (n.s.)**.

### Exact Swichy relevance
- Release-height-ratio as the primary consistency metric
- Elbow p=0.471 is one leg of the "elbow invariance" prior — **see §D2 for why this is NOT proof**

### Player-height relevance
**Highest relevance in the entire corpus.** Release height is reported *only* as a body-height ratio. This is direct precedent that release height must be height-normalised to be comparable across players.

### What we DO NOT take from it
Do not read p=0.471 as evidence of elbow invariance (null-hypothesis fallacy — §D2).

### Limitations
"Recreationally active males", not elite. Proprietary SwRI system limits replication.

### Used in
`research.md` §2.3 · release-height rule · player-height normalisation argument

---

## A3. Cabarkapa et al. 2026 — Proficient 3-Point Shooters (markerless 3D)

- **Year:** 2026 · **Journal:** *Frontiers in Sports and Active Living* 8:1732293
- **Evidence tier:** Tier 2 · **Verification:** ✅ FULL TEXT VERIFIED (volume/pages ⚠️ confirm at DOI)

### What we take from it
Knee prep 113.2 (non-prof) vs **94.3 (prof), p<0.001**; hip 155.9 vs 143.1, p<0.001; ankle 62.3 vs 50.1, p<0.001. Elbow prep p=0.706 (n.s.); elbow at release 158.8 vs 159.3, p=0.228 (n.s.). Knee at release 172.2 vs **164.9, p=0.010**.

### Exact Swichy relevance
Only study where load depth discriminated skill — **but only at 3-point distance**. Contradicted at other distances (§D3).

### Player-height relevance
None reported.

### Limitations
n=24. Contradicted by A1 (two-point, n.s.) and by the 2026 free-throw 2D study (n.s. after Bonferroni).

### Used in
`research.md` §2.1 · **must be cited alongside its contradictions**

---

## A4. Miller & Bartlett 1996 — Shooting Kinematics, Distance & Position

- **Authors:** Miller, S., Bartlett, R. · **Journal:** *Journal of Sports Sciences* 14(3):243–253
- **DOI:** 10.1080/02640419608727708 · **PMID:** 8809716
- **Evidence tier:** **Tier 1 — Scopus + WoS SCIE + MEDLINE**
- **Verification:** ⚠️ ABSTRACT ONLY (numbers from PubMed abstract)

### Why we trust it
Highest-indexed journal in the corpus. Bartlett is a foundational sports-biomechanics figure. 3D cinematography at 100 Hz.

### What we take from it
**Ball launch angle** 52–55° at 2.74 m and 4.57 m; 48–50° at 6.40 m. n=15.

### Player-height relevance
None directly, but establishes that launch angle **decreases with distance** — consistent with the Brancazio geometry (§B1).

### Limitations
1996 instrumentation. n=5 per position group. ⚠️ Full text not retrieved.

### Used in
`research.md` §2.3 · ball launch angle target band

---

## A5. Tran & Silverberg 2008 — Optimal Free-Throw Release Conditions

- **Journal:** *Journal of Sports Sciences* 26(11):1147–1155 · **DOI:** 10.1080/02640410802004948
- **Evidence tier:** **Tier 1 — Scopus + WoS SCIE + MEDLINE** · **Verification:** ⚠️ ABSTRACT ONLY

### What we take from it
10⁵+ 3D trajectory simulations → launch **52°** to horizontal, up to **3 Hz backspin**, aim at back of ring.

### Player-height relevance
**Indirect but important.** This is a *physics optimisation given release conditions*, so its output depends on release height — which depends on player height. It supports **computing** an optimal angle rather than looking one up.

### Used in
`research.md` §2.3 · geometric personalisation argument (§B1)

---

## A6. Amaro et al. 2025 — Jump & Release Parameters vs Accuracy

- **Journal:** *Journal of Functional Morphology and Kinesiology* 10(4):459 · **PMC:** PMC12641682
- **Evidence tier:** Tier 2 · **Verification:** ✅ FULL TEXT VERIFIED

### What we take from it
n=18, 90 shots each, force plates + 200 Hz ball kinematics. Release height 2.02–2.13 m; **launch angle 54.5–57.9°**; release velocity 6.7–7.0 m/s; jump height ~0.23–0.25 m. Successful shots had significantly higher **release height and jump height**; **release angle n.s.** between made and missed.

### ⚠️ The number Swichy must state out loud
**Spearman correlations between biomechanical parameters and accuracy: R² = 0.005–0.012.** Authors conclude these parameters are *"necessary conditions"* but do not explain accuracy in isolation.

### Exact Swichy relevance
**This is the single most important citation for scoping the product.** It is the evidence that Swichy must NOT claim make/miss prediction.

### Used in
`research.md` §2.3 · Part 20 scoping · every "we do not predict makes" statement

---

## A7. Okubo & Hubbard 2018 — Set-Shot vs Jump-Shot Kinematics

- **Journal:** *Proceedings* (MDPI) 2(6):201, ISEA conference · **DOI:** 10.3390/proceedings2060201
- **Evidence tier:** **Tier 3 — conference proceedings, not WoS/Scopus indexed**
- **Verification:** ⚠️ ABSTRACT ONLY (full text 403)

### What we take from it
**Conceptual definition only:** set vs jump shot are distinguished by the **vertical velocity and acceleration of the shooting-side shoulder at release** (non-zero upward vs ≈zero). In a set shot, upward body motion contributes to release speed/angle/backspin; in a jump shot the arm alone must produce them.

### What we DO NOT take from it
Any numerical value. Tier 3 venue — cite for the **model**, never for data.

### Used in
`research.md` §2.4 · set/jump classifier rationale

---

## A8. Okazaki & Rodacki 2012 — Increased Shooting Distance

- **Journal:** *Journal of Sports Science and Medicine* 11:231–237 · **Evidence tier:** Tier 2 (Scopus, WoS, PubMed)
- **Verification:** ⚠️ ABSTRACT/PARTIAL

### What we take from it
n=10 experts. **"Increased distance of shooting did not change the maximum and minimum joint angles"** (F<3.5, p>0.05). Release height decreased 2.46 → 2.38 → 2.33 m with distance.

### ⚠️ Definitional warning
Its "ball release angle" values (78.92° → 65.60°) are **not comparable** to Miller & Bartlett's launch angle — a different construct. Do not pool.

### Used in
`research.md` §2.4 (invariance prior) — **with the null-result caveat of §D2**

---

## A9. Okazaki, Rodacki & Satern 2015 — A Review on the Basketball Jump Shot

- **Journal:** *Sports Biomechanics* 14(2):190–205 · **DOI:** 10.1080/14763141.2015.1052541
- **Evidence tier:** **Tier 1 — Scopus + WoS SCIE**, official ISBS journal (IF 2.0, CiteScore 4.3, Q2)
- **Verification:** ⚠️ ABSTRACT ONLY

### Exact Swichy relevance
Highest-quality *review* available; use for framing and for the phase decomposition, not for numbers.

---

# PART B — PLAYER HEIGHT & ANTHROPOMETRY

## B1. Brancazio geometric rule (physics, secondary-sourced)

- **Source type:** Physics model, reported via secondary sources · **Evidence tier:** Tier 4 (⚠️ primary not yet retrieved)
- **Verification:** ❌ UNVERIFIED — **must locate Brancazio's primary publication before thesis citation**

### The claim
Optimal launch angle ≈ **45° + ½ × (angle between the ball at launch and the basket)**. Because a taller player releases higher, the ball-to-basket angle shrinks, so the optimal launch angle **decreases with release height**. Reported values: ≈52.2° at 5'4" down to ≈48.7° at 7'0"; for most players at 10–25 ft, least-effort angle 47–52°.

### Why this matters more than any height lookup table
It converts "player height" from an **empirical category** into a **computable geometric input**. Swichy would not need a height→angle table; it could compute the target from measured release height + horizontal distance + 3.05 m rim height.

### ⚠️ Blocking dependency
Swichy currently measures **neither** ball launch angle **nor** horizontal distance to the hoop. See `AUDIT_AND_PLAN.md` §12.

### Evidence class
`INFERRED` (geometry is sound; the specific numbers are secondary-sourced and unverified)

---

## B2. Height normalisation precedent — Cabarkapa group

Across A1, A2, A3 and the female study (JFMK 8(3):129), the group reports **release height and elbow height as ratios of body height**, never as raw metres.

### Evidence class for "normalise release height by player height"
`SUPPORTED WITH LIMITATIONS` — consistent precedent across 4 papers from **one group**. No independent replication found.

### What this does NOT license
It does **not** license height-conditioned *angle* targets. No source supports "tall player should have X knee angle."

---

## B3. Monocular height estimation — feasibility

**Findings:**
- MediaPipe world landmarks are produced by fitting the **GHUM statistical body model** (BlazePose GHUM Holistic, arXiv:2206.11678). Output is metric-scale but derived from a fitted statistical body, carrying anatomical priors.
- Monocular scale ambiguity is **fundamental**: projecting 3D onto 2D creates size/distance ambiguity resolvable only with semantic scale cues or a reference object.
- The best-performing MediaPipe height study (166 subjects, **1.54 ± 0.64%** error) **required a reference object** for calibration.
- Without calibration, published monocular approaches rely on bone-length proportion priors → estimates regress toward population means.

### Evidence class
`NOT SUPPORTED` for feeding **scoring**. `SUPPORTED WITH LIMITATIONS` as a **sanity check** on user-typed height.

### Swichy decision
**User-typed height is authoritative. Vision height must never silently override it, and must never feed scoring until validated against known-height players.** Height remains **optional** — `height_cm = None` is a first-class state, and no default (e.g. 180 cm) is ever fabricated.

---

# PART C — MOTOR LEARNING & COACHING FEEDBACK

## C1. Wulf — Attentional focus and motor learning

- **Key source:** Wulf, G. (2013). *Attentional focus and motor learning: a review of 15 years.* International Review of Sport and Exercise Psychology.
- **Supporting:** free-throw study (n=50) comparing internal vs external focus at 33% and 100% feedback frequency; imagery/external-focus free-throw study (PMC8085315).
- **Evidence tier:** Tier 1–2 · **Verification:** ⚠️ ABSTRACT ONLY

### What we take from it
- **External focus of attention** (on the movement *effect*) produces better learning and performance than **internal focus** (on body parts). Demonstrated across balance, accuracy, force, speed tasks including basketball.
- Mechanism: the **constrained-action hypothesis** — internal focus triggers conscious control that disrupts automatic processing.
- In the free-throw study, external-focus feedback at **both** 33% and 100% frequency outperformed internal focus in retention and transfer.

### 🚨 Direct consequence for Swichy — this contradicts our current messages
`config/biomechanics.yaml` messages are almost all **internal focus**:
> ❌ "Bend knees deeper in the load" · "Tuck elbow under the ball" · "Stay taller through the torso"

The literature says these are the *less effective* form. External-focus equivalents:
> ✅ "Drive the floor away before you shoot" · "Send the ball straight up the line to the rim" · "Keep your chest pointed at the rim"

### Evidence class
`DIRECTLY SUPPORTED` (external > internal focus is one of the most replicated findings in motor learning)

### Used in
Part 46/47 feedback rewrite · `feedback/` message catalogue · AR/EN localisation

---

# PART D — CONTESTED, CORRECTED & REJECTED

## D1. ⚠️ "Release angle" means two different things

| Construct | Definition | Typical | Sources |
|---|---|---|---|
| **Ball launch angle** | Ball's initial trajectory vs horizontal (projectile physics) | 48–57° | A4, A5, A6 |
| **Arm elevation angle** | *"relative angle between the fully extended upper limb and a line parallel to the ground"* | 40–61° | A1, A2 |

Verbatim definition quoted from A1 full text. **These are different physical quantities.** Conflating them creates false agreement. Swichy must state which it computes.

**Evidence class:** `DIRECTLY SUPPORTED` (definitional, verified from full text)

---

## D2. 🚨 CORRECTION — "Elbow invariance" is NOT proven

`research.md` v1 claimed elbow invariance was supported by six comparisons. **The council rejected this, correctly.**

The supporting evidence is a set of **non-significant p-values** (0.471, 0.706, 0.228) from samples of n=10–34, **all from one research group**. Failing to reject the null hypothesis with a small sample is **not** evidence of no effect — it is evidence of low statistical power. No power analysis exists in any of these papers.

Additionally, motor-learning theory (**Bernstein's degrees-of-freedom problem**) predicts *functional variability* rather than a single invariant parameter — which cuts against the invariance model.

### Revised evidence class
`ENGINEERING DECISION` — a reasonable *tunable prior* for setting an elbow band. **Not** a validated biomechanical law.

### How Swichy must phrase it
> "Elbow set point is treated as a stable reference because published values do not vary systematically with distance. This is a tunable engineering prior, not a demonstrated invariance; the underlying studies are underpowered null results from a single laboratory."

---

## D3. 🚨 "Deeper knee load = better" is NOT SUPPORTED

| Source | Finding | Direction |
|---|---|---|
| Cabarkapa 2026 (3-point, n=24) | Proficient loaded **deeper** (94.3 vs 113.2°), p<0.001 | Deeper better |
| Cabarkapa 2022 (two-point, n=10) | **No significant** proficiency difference | Null |
| 2026 free-throw 2D study | Knee angles **not significant** after Bonferroni | Null |
| Reported association | **Lower** knee flexion in prep associated with success | **Opposite** |

Four findings, three directions.

### Evidence class
`NOT SUPPORTED`

### Swichy decision
`knee_flexion_loading` must be **demoted from a scored rule to a displayed metric**. Show the number and its trend across reps; do not assert a target and do not penalise. This also removes an unfair penalty on shorter/longer-limbed players.

---

## D4. ⚠️ Measurement-error ceiling (flagged by council peer review as a universal blind spot)

Every angle claim above assumes the pose estimator recovers the angle accurately. It does not, uniformly:

| Joint | Error vs marker-based | Source |
|---|---|---|
| Sagittal knee / hip, camera-side | bias +1.5±4.1° / −3.6±4.6°, r²=0.98/0.94 | Wade et al. 2023, *PLoS ONE* 18(11):e0293917 ✅ |
| **Elbow (basketball throws)** | **RMSD 16.68±5.03°** | Thomas et al. 2025, *Biomechanics* 5(4):100 ⚠️ |
| **Shoulder (by camera azimuth)** | **bias −16.25° to −26.07°, RMSE 24–36°** | Baldinger et al. 2025, *Sensors* 25(3):799 ✅ |
| Occluded-side limbs | SD ≈ doubles | Wade et al. 2023 ✅ |
| Multi-camera 3D markerless benchmark | segment angles < 5.5° | Kanko et al. 2021, *J Biomech* 127:110665 (**Tier 1, WoS SCIE, MEDLINE**) ⚠️ |
| OpenCap (2-camera) | MAE 3.85°, RMSE 4.34° | Uhlrich et al. 2023, *PLOS Comput Biol* 19(10):e1011462 ⚠️ |

### Consequence
No coaching target can be asserted more precisely than the measurement supports. **Shoulder angle must be excluded from scoring.** Elbow gets relative-to-baseline treatment only.

⚠️ **Conflict of interest to disclose:** Kanko et al. co-author W.S. Selbie is affiliated with Theia Markerless Inc., the system under evaluation.

---

## D5. ❌ REJECTED SOURCES

| Source | Reason |
|---|---|
| **IJERPH** (Vencúrik et al. 2021, 18(3):934) | **Discontinued from Web of Science 13 Feb 2023** (Content Relevance failure; MDPI appealed 31 Mar 2023, no confirmed reinstatement). Still in Scopus/PubMed. May be cited for entry-angle data **only with the delisting disclosed in a footnote**. Also used Xsens IMU, not optical. |
| **physicaleducationjournal.net** — *Int. J. of Physiology, Sports and Physical Education* (Sparkling Press) | No Scopus, WoS or PubMed listing found. Matches predatory-adjacent domain cluster (physicaleducationjournal .in/.com, kheljournal.com). **Excluded entirely.** |
| **iosrjournals.org** free-throw release-angle paper | Unindexed. Not used. |
| Wordpress/blog "physics of basketball" pages | Tier 5. Used only to locate the Brancazio rule; **not citable** (see B1). |

---

# PART E — SOURCE → RULE → CODE MAP

| Rule in `config/biomechanics.yaml` | Evidence | Class | **Action required** |
|---|---|---|---|
| `knee_flexion_loading` | D3 (contradictory) | `NOT SUPPORTED` | **Demote to displayed metric** |
| `hip_hinge_loading` | A1, A3 | `SUPPORTED WITH LIMITATIONS` | Keep, widen band, distance-specific |
| `elbow_slot_ball_lift` | A1, A2, D2 | `ENGINEERING DECISION` | Keep as *tunable prior*; relabel rationale |
| `shoulder_alignment_lift` | D4 (bias −16 to −26°) | `NOT SUPPORTED` (measurement) | **Remove from scoring** |
| `trunk_posture` | A2 (trunk lean), D4 | `SUPPORTED WITH LIMITATIONS` | **Currently 0% pass — Y-axis bug.** Fix first |
| `release_height` | A2 (p=0.010), A6 | `SUPPORTED WITH LIMITATIONS` | **Currently 0% pass — Y-axis bug.** Then height-normalise |
| `elbow_extension_release` | A3 (158.8–159.3°) | `SUPPORTED WITH LIMITATIONS` | Note: **not 180°** |
| `index_alignment_release`, `follow_through_*` | — | `ENGINEERING DECISION` | Label as such; no literature norm found |
| `landing_balance` | — | `ENGINEERING DECISION` | Label as such |
| `head_stability` | — | `ENGINEERING DECISION` | Label as such |

**Code map:** `angles/calculator.py` (D1, D4) · `phase_detection/detector.py` (A7 set/jump) · `analysis/engine.py` (all rules) · `feedback/` (C1 external focus) · future `player/profile` (B2, B3)

---

# PART F — CLAIMS REQUIRING FURTHER VERIFICATION

| # | Item | Needed |
|---|---|---|
| F1 | **Thomas et al. 2025** full author list | Aggregators show only "Carina Thomas and 3 others". Resolve at DOI 10.3390/biomechanics5040100 |
| F2 | **Brancazio** primary publication | Rule currently secondary-sourced (B1). **Blocking for any geometric personalisation claim** |
| F3 | Miller & Bartlett 1996 full text | Currently abstract-only |
| F4 | Tran & Silverberg 2008 full text | Currently abstract-only |
| F5 | Okazaki 2015 review full text | Currently abstract-only |
| F6 | Cabarkapa 2026 volume/pages | Confirm at Frontiers once in final issue |
| F7 | Wulf 2013 + free-throw focus study full texts | Currently abstract-only; **needed before C1 drives the feedback rewrite** |
| F8 | Okubo & Hubbard 2018 full text | 403; conceptual claim only until resolved |
| F9 | IJERPH current WoS status | Re-check Clarivate Master Journal List before submission |

---

# PART G — RESEARCH GAPS SWICHY SHOULD NAME ITSELF

1. **No peer-reviewed study has validated single-camera 2D pose estimation against marker-based capture during an actual basketball shot.** Closest is Thomas et al. 2025 (multi-camera 3D markerless, *simulated* throws, elbow RMSD 16.68°).
2. **No independent replication of the Cabarkapa group's shooting kinematics** exists. Single-lab dependency is the largest credibility risk in this corpus.
3. **"Proficient compared to whom?"** — proficiency thresholds (≥80%, ≥50%, ≥40%) are study-specific and mostly derived from *recreational* adults. Swichy must not tell a young player they differ from "proficient shooters" without stating that reference group.
4. **No study reports MediaPipe reliability during the explosive, self-occluding release motion specifically.**
5. **No evidence base for height-conditioned joint-angle targets.** Do not create them.

---

*Maintained alongside `research.md` and `docs/AUDIT_AND_PLAN.md`. Every claim in either document must resolve to an entry here.*

---

# PART H — RESOLVABLE LINKS

A link for every source above. Merged from the former `PAPER_LINKS.md`
so that one file answers both "what does this support?" and "where do
I find it?".

**A resolvable link for every source cited anywhere in this repository.**

This file is an *index*, not an evidence map. It answers one question: "where do
I actually find this paper?" For what each paper supports, how strongly, and
what it must not be used for, read [`resources.md`](resources.md) — that file
remains the single source of truth for every research claim.

Sources are spread across four files, and they do not all agree with each other.
See [Known citation problems](#known-citation-problems) before sharing this
corpus with anyone.

| File | Role |
|---|---|
| [`resources.md`](resources.md) | Curated evidence map. Authoritative. |
| [`research.md`](research.md) | Working research notes. |
| [`docs/BIOMECHANICS_RESEARCH.md`](docs/BIOMECHANICS_RESEARCH.md) | **Older, superseded.** Contains two errors, below. |
| [`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md) | Engineering prior art — other systems, not evidence. |

---

## How to read the status column

| Marker | Meaning |
|---|---|
| ✅ | DOI, PMC or PMID recorded explicitly in `resources.md`. Link is a mechanical expansion of that identifier. |
| 🔧 | **DOI constructed from the publisher's numbering pattern**, because the source file records journal + volume + article number but no DOI. Very likely correct, **not opened**. Verify before citing. |
| ❌ | No identifier recorded anywhere. Must be located. |

A second, independent axis is how much of each paper was actually read.
`resources.md` tracks this and it is not flattering: **four papers were read in
full**; most of the rest were read as abstracts only. That column is reproduced
here as *Read*.

| Read | Meaning |
|---|---|
| FULL | Numbers taken from the paper's own results tables |
| ABS | Publisher abstract only; full text never retrieved |
| — | Not recorded |

---

## A. Core biomechanics

| Ref | Paper | Link | Status | Read |
|---|---|---|---|---|
| A1 | Cabarkapa et al. 2022 — Distance & proficiency, professional males. *JFMK* 7(4):78 | https://doi.org/10.3390/jfmk7040078 · https://pmc.ncbi.nlm.nih.gov/articles/PMC9590067/ | ✅ | FULL |
| A2 | Cabarkapa et al. 2023a — Proficient free-throw shooters, 3D markerless. *Front. Sports Act. Living* 5:1208915 | https://doi.org/10.3389/fspor.2023.1208915 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10436204/ | ✅ | FULL |
| A3 | Cabarkapa et al. 2026 — Proficient 3-point shooters. *Front. Sports Act. Living* 8:1732293 | *(no DOI recorded)* | ❌ | FULL |
| A4 | Miller & Bartlett 1996 — Shooting kinematics, distance & position. *J Sports Sci* 14(3):243–253 | https://doi.org/10.1080/02640419608727708 · https://pubmed.ncbi.nlm.nih.gov/8809716/ | ✅ | ABS |
| A5 | Tran & Silverberg 2008 — Optimal free-throw release conditions. *J Sports Sci* 26(11):1147–1155 | https://doi.org/10.1080/02640410802004948 | ✅ | ABS |
| A6 | Amaro et al. 2025 — Jump & release parameters vs accuracy. *JFMK* 10(4):459 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12641682/ | ✅ | FULL |
| A7 | Okubo & Hubbard 2018 — Set-shot vs jump-shot kinematics. *Proceedings* 2(6):201 | https://doi.org/10.3390/proceedings2060201 | ✅ | ABS |
| A8 | Okazaki & Rodacki 2012 — Increased shooting distance. *J Sports Sci Med* 11:231–237 | *(no DOI recorded; open access, findable by title)* | ❌ | ABS |
| A9 | Okazaki, Rodacki & Satern 2015 — Review of the basketball jump shot. *Sports Biomech* 14(2):190–205 | https://doi.org/10.1080/14763141.2015.1052541 | ✅ | ABS |
| — | Cabarkapa et al. 2023b — Female shooters. *JFMK* 8(3):129 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10531893/ | ✅ | FULL |
| — | Li 2025 — Arm-joint coordination variability, collegiate vs recreational, 3-pt | https://pmc.ncbi.nlm.nih.gov/articles/PMC12121896/ | ✅ | FULL |
| — | Jovanović et al. 2022 — Made vs missed jump shots. *Biomechanics* 2(3):428–441 | https://doi.org/10.3390/biomechanics2030028 | ✅ | — |
| — | Shot-type definitions | https://pmc.ncbi.nlm.nih.gov/articles/PMC4454648/ | ✅ | — |

> **A3 is the priority gap.** It carries the *only* statistically significant
> knee-depth result in the entire corpus (p<0.001), and it is also the finding
> most contradicted elsewhere (see `resources.md` §D3). It needs a real DOI
> before it is defensible.

---

## B. Pose-estimation accuracy

The measurement-error ceiling. Nothing in section A can be asserted more
precisely than these papers allow, which is why shoulder angle is excluded from
scoring and elbow is treated relative-to-baseline only.

| Paper | Finding | Link | Status |
|---|---|---|---|
| BlazePose GHUM Holistic — the model MediaPipe actually runs | Metric output fitted from a statistical body model | https://arxiv.org/abs/2206.11678 | ✅ |
| Thomas et al. 2025. *Biomechanics* 5(4):100 | **Elbow RMSD 16.68 ± 5.03°** in basketball throws | https://doi.org/10.3390/biomechanics5040100 | ✅ |
| Wade et al. 2023. *PLoS ONE* 18(11):e0293917 | Sagittal knee/hip bias +1.5±4.1° / −3.6±4.6° | https://doi.org/10.1371/journal.pone.0293917 | 🔧 |
| Baldinger et al. 2025. *Sensors* 25(3):799 | **Shoulder bias −16.25° to −26.07°** by camera azimuth | https://doi.org/10.3390/s25030799 | 🔧 |
| Kanko et al. 2021. *J Biomech* 127:110665 | Multi-camera 3D markerless benchmark, <5.5° | https://doi.org/10.1016/j.jbiomech.2021.110665 | 🔧 |
| Uhlrich et al. 2023. *PLOS Comput Biol* 19(10):e1011462 | OpenCap 2-camera, MAE 3.85° | https://doi.org/10.1371/journal.pcbi.1011462 | 🔧 |

> **Conflict of interest to disclose:** Kanko et al. co-author W.S. Selbie is
> affiliated with Theia Markerless Inc., the system under evaluation.

---

## C. Motor learning & coaching feedback

Drives the external-focus rewrite of every message in `config/biomechanics.yaml`.

| Paper | Link | Status | Read |
|---|---|---|---|
| Wulf, G. (2013). *Attentional focus and motor learning: a review of 15 years.* Int. Rev. Sport Exerc. Psychol. | *(no DOI recorded)* | ❌ | ABS |
| Supporting free-throw / imagery focus study | https://pmc.ncbi.nlm.nih.gov/articles/PMC8085315/ | ✅ | ABS |

---

## D. Player height & geometry

| Source | Link | Status |
|---|---|---|
| Brancazio geometric rule — optimal angle ≈ 45° + ½ × (ball-to-basket angle) | *(primary publication never located; currently sourced from blog pages)* | ❌ |

> `resources.md` §B1 marks this **`❌ UNVERIFIED`** and **"blocking for any
> geometric personalisation claim."** Do not cite it until the primary is found.

---

## E. Engineering prior art

Other systems, cited for approach — **not** as evidence for any biomechanical claim.

| System | Link | Note |
|---|---|---|
| HoopLab — Flutter + YOLO mobile app | https://doi.org/10.5121/csit.2025.152402 | Conference proceedings |
| PoseShot — *Scientific Reports* | https://doi.org/10.1038/s41598-026-41025-0 | 75 free throws only |
| SpaceJam — 2D joint dataset | https://doi.org/10.21203/rs.3.rs-2947413/v1 | ⚠️ **Research Square preprint — not peer reviewed** |

---

## F. Excluded sources

Recorded so they are not accidentally reintroduced. See `resources.md` §D5.

| Source | Reason |
|---|---|
| Vencúrik et al. 2021, *IJERPH* 18(3):934 — https://doi.org/10.3390/ijerph18030934 | **Discontinued from Web of Science, 13 Feb 2023.** Citable for entry-angle data *only* with the delisting disclosed in a footnote. |
| *Int. J. of Physiology, Sports and Physical Education* (Sparkling Press) — `10.33545/26647710.2025.v7.i2f.191` | No Scopus, WoS or PubMed listing. Matches a predatory-adjacent domain cluster. **Excluded entirely** — but still cited in `docs/BIOMECHANICS_RESEARCH.md`, see below. |
| iosrjournals.org free-throw release-angle paper | Unindexed. |
| Wordpress "physics of basketball" pages | Used only to locate the Brancazio rule. Not citable. |

---

## Known citation problems

Two live contradictions between `docs/BIOMECHANICS_RESEARCH.md` (older) and
`resources.md` (authoritative). Both are in the repository right now.

### 1. A misattributed citation — same article, two different papers

`docs/BIOMECHANICS_RESEARCH.md:172` records:

> Jovanović, M., et al. (**2023**). Impact of Distance and Proficiency on
> Shooting Kinematics. ***Sports***, 7(4), 78. `10.3390/sports7040078`

`resources.md` §A1 records the same content as:

> **Cabarkapa** et al. (**2022**). ***Journal of Functional Morphology and
> Kinesiology***, 7(4):78. `10.3390/jfmk7040078`

Same volume, same issue, same article number — **different journal, different
authors, different year.** The `sports` vs `jfmk` DOI slug suggests the older
file is the one in error.

This matters more than the average citation slip: A1 is the source of the
set-vs-jump classifier threshold and of the distance-specific knee and hip
targets. Anyone checking that single DOI lands on a different article.

### 2. An excluded source is still cited

`docs/BIOMECHANICS_RESEARCH.md` lines 59 and 173 still cite
`10.33545/26647710.2025.v7.i2f.191`, which `resources.md` §D5 excludes entirely
as predatory-adjacent. The older document contradicts the newer one, and both
ship together.

---

## What still needs doing

Mirrors `resources.md` Part F, reduced to the items that block a citation.

| # | Item | Why it blocks |
|---|---|---|
| 1 | Locate the DOI for **Cabarkapa 2026** (A3) | Carries the only significant knee-depth result |
| 2 | Locate **Brancazio's** primary publication | Blocks every geometric personalisation claim |
| 3 | Verify the four 🔧 DOIs in section B | They set the measurement-error ceiling for the whole system |
| 4 | Resolve the two contradictions above | One is a wrong-article link on the most-used paper |
| 5 | Retrieve full text for the nine `ABS` papers | Currently cited from abstracts |

---

*Index only. Every claim must resolve to an entry in [`resources.md`](resources.md).*
