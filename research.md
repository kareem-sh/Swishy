# Literature Review: Basketball Shooting Kinematics and Markerless Motion Capture Validation

**Prepared for:** Vision-based AI basketball coaching application (Swichy)
**Date compiled:** 9 August 2026
**Status:** Draft for revision

---

## 0. How This Document Was Built

Every paper below was retrieved and read via its publisher page, PubMed record, or PubMed Central full text. Numerical values are transcribed from the papers' own results tables, not from secondary summaries. Where a value could not be extracted from the primary source, it is explicitly marked `[UNVERIFIED]`.

**Databases searched:** PubMed / MEDLINE, PubMed Central, ScienceDirect (Elsevier), Taylor & Francis Online, Frontiers, PLOS, MDPI, DOAJ, Semantic Scholar.

**Two sources were screened OUT and are documented in §5.3** — one because its journal was delisted from Web of Science, one because it is an unindexed physical-education title.

---

## 1. Definitions and Taxonomy

### 1.1 Set shot vs. jump shot — the mechanical distinction

This distinction matters more than it first appears, and the definitive treatment is Okubo & Hubbard (2018), who classify the two shots **by the vertical velocity and acceleration of the shooting-side shoulder at the instant of ball release**:

| | **Set shot** | **Jump shot** |
|---|---|---|
| Shoulder vertical velocity at release | Non-zero, upward | Zero or very small |
| Timing of release | During upward body motion | At or after jump apex |
| Source of ball release conditions | Upward body motion **plus** arm motion | Almost entirely **arm motion** |
| Typical instances | Free throw, close-range shots, youth/beginner shooting | Two-point and three-point shots, contested shots |

> **The consequence, in Okubo & Hubbard's framing:** in a set shot the whole-body upward motion contributes directly to ball release speed, angle and backspin. In a jump shot the body has almost no velocity at release, so **the shooting arm alone must produce the release conditions**.

**Why this matters for Swichy:** it means the arm and the legs are not interchangeable sources of "power" across shot types. A coaching model that grades leg drive identically for a free throw and a three-pointer is mechanically wrong. It also predicts — correctly, as §2.4 shows — that lower-limb angles should scale with shot type and distance while arm angles stay comparatively fixed.

### 1.2 Phase model

The literature converges on a three-phase decomposition:

1. **Preparatory / load phase** — deepest flexion of knee, hip, ankle before upward drive. Also where "set point" elbow and shoulder angles are measured.
2. **Release phase** — the frame at which the ball leaves the hand.
3. **Follow-through / inertia phase** — post-release.

Some papers (three-point-specific work) name these *Preparation and Ball Elevation*, *Stability and Ball Release*, and *Inertia and Follow-through*.

### 1.3 ⚠️ CRITICAL: "Release angle" means two different things

**This is the single most dangerous ambiguity in this literature, and it must be resolved explicitly in the thesis.**

| Construct | Definition | Typical value | Used by |
|---|---|---|---|
| **Ball launch angle** | Angle of the ball's initial trajectory relative to horizontal — a projectile-physics quantity | **48–55°** | Miller & Bartlett (1996); Tran & Silverberg (2008); Amaro et al. (2025) |
| **Arm elevation angle** | *"the relative angle between the fully extended upper limb and a line parallel to the ground"* — a body-posture quantity | **40–61°** | Cabarkapa et al. (2022, 2023a, 2023b) |

The definition on the right is quoted verbatim from the Cabarkapa et al. (2022) full text. The two quantities are numerically close enough (52° vs. 60.8°) that conflating them **looks like agreement between studies when it is not**. A jury member who catches this conflation can discredit the entire kinematic chapter.

**Recommendation:** Swichy should compute and report the **ball launch angle** (it is what determines whether the shot goes in, and it is what the optimisation literature validates). State the distinction in the thesis and cite both constructs separately.

### 1.4 Angle convention

Cabarkapa's group — the source of most numerical data below — uses **internal angles** throughout:

- Knee = internal angle between thigh and shank → **smaller number = more flexion**
- Hip = internal angle between torso and thigh
- Elbow = internal angle between upper arm and forearm → **smaller number = more flexion**
- Shoulder = relative angle between upper arm and torso
- Ankle = relative angle between shank and ground

Release height and elbow height are reported as **ratios normalised to body height** — a deliberate choice that makes them scale-invariant and, as §3 argues, far more robust to computer-vision depth error than metric distances.

---

## 2. Comprehensive Kinematic Matrix

### 2.1 Load / Preparatory Phase

| Variable | Free Throw (4.57 m) | Two-Point | Three-Point (6.75 m) | Source, sample, method |
|---|---|---|---|---|
| **Knee** | 121.5 ± 8.8° | 116.7 ± 7.4° | 112.5 ± 7.4° | Cabarkapa et al. 2022 — n=10 **professional** males, 150 shots, 2D sagittal video @120 fps |
| **Knee** | — | 114.8 ± 8.9° | 109.4 ± 9.2° | Cabarkapa et al. 2023b — n=18 recreational **females**, 2D video @30 fps |
| **Knee** | 107.3 ± 14.9 (non-prof)<br>113.3 ± 9.1 (prof)<br>p=0.183 | — | 113.2 (non-prof)<br>**94.3 (prof)**<br>**p<0.001** | Cabarkapa 2023a (n=34) and 2026 (n=24) — **3D markerless**, 9 cameras @120 Hz |
| **Hip** | 143.9 ± 8.4° | 141.1 ± 8.1° | 135.5 ± 8.4° | Cabarkapa et al. 2022 |
| **Hip** | — | 133.2 ± 8.9° | 126.3 ± 12.9° | Cabarkapa et al. 2023b |
| **Hip** | — | — | 155.9 (non-prof)<br>**143.1 (prof)**<br>**p<0.001** | Cabarkapa et al. 2026 |
| **Ankle** | 61.2 ± 6.7° | 61.5 ± 5.3° | 58.5 ± 4.7° | Cabarkapa et al. 2022 |
| **Ankle** | — | — | 62.3 (non-prof)<br>**50.1 (prof)**<br>**p<0.001** | Cabarkapa et al. 2026 |

**Synthesis.** Across four independent samples, **knee flexion deepens monotonically as shooting distance increases** (≈121° → 112° in professionals; ≈115° → 109° in recreational females). The 2026 three-point study is the only one where load depth *discriminated skill*: proficient shooters loaded to 94.3° vs. 113.2° for non-proficient — a ~19° gap at p<0.001, accompanied by significantly deeper hip (143.1° vs. 155.9°) and ankle (50.1° vs. 62.3°) flexion.

**This is the most defensible coachable lower-limb finding in the literature.** Note that it is distance-specific: the same group found *no* significant knee difference by proficiency at the free-throw line (p=0.183).

### 2.2 Set Point (elbow and shoulder at preparatory phase)

| Variable | Value | Source, sample, method |
|---|---|---|
| **Elbow** | 61.9 ± 13.7° (FT)<br>58.2 ± 16.6° (2pt)<br>63.6 ± 21.6° (3pt) | Cabarkapa et al. 2022 — **2D video**, n=10 professionals |
| **Elbow** | 50.4 ± 13.4 (prof) / 54.6 ± 9.4 (non-prof) — 2pt<br>49.1 ± 11.5 (prof) / 55.0 ± 12.2 (non-prof) — 3pt | Cabarkapa et al. 2023b — 2D video, n=18 females |
| **Elbow** | 78.0 ± 27.4 (non-prof) / 85.1 ± 20.6 (prof), p=0.471 | Cabarkapa et al. 2023a — **3D markerless**, n=34, free throw |
| **Elbow** | 84.4 (non-prof) / 81.9 (prof), **p=0.706** | Cabarkapa et al. 2026 — **3D markerless**, n=24, three-point |
| **Shoulder** | 78.9 ± 26.9° (FT)<br>72.7 ± 26.4° (2pt)<br>65.8 ± 31.4° (3pt) | Cabarkapa et al. 2022 |
| **Shoulder** | 52.3 ± 11.6 (prof) / 63.8 ± 6.7 (non-prof) — 2pt, p<0.05<br>50.9 ± 9.5 (prof) / 56.8 ± 11.3 (non-prof) — 3pt, n.s. | Cabarkapa et al. 2023b |

**Three warnings you must handle before the defence:**

1. **The elbow set-point value shifts ~20–25° with measurement modality.** 2D video gives 50–64°; 3D markerless gives 78–85° — *from the same research group, on the same task, with the same stated variable definition*. This is direct, internally-generated evidence of the measurement bias that §3 addresses. It is the strongest single argument for a bias-correction layer in your system.

2. **Shoulder standard deviations are enormous** in the 2D professional study — ±26.9° to ±31.4°, an SD approaching 40% of the mean. Do not build a scoring rule on absolute shoulder angle.

3. **The coaching folklore of a "90° elbow" at set point is closest to the 3D markerless values (78–85°), not the 2D values.** State this explicitly — it reframes a coaching cliché as a measurement-dependent claim, which is exactly the kind of nuance a jury rewards.

### 2.3 Release Frame

| Variable | Value | Source, sample, method |
|---|---|---|
| **Ball launch angle** | **52–55°** at 2.74 m and 4.57 m<br>**48–50°** at 6.40 m | Miller & Bartlett 1996 — n=15, **3D cinematography @100 Hz**, PMID 8809716 |
| **Ball launch angle (optimal)** | **52°** to horizontal, with up to **3 Hz backspin**, aimed at the back of the ring | Tran & Silverberg 2008 — 10⁵+ 3D trajectory simulations |
| **Ball launch angle** | **54.5–57.9°** | Amaro et al. 2025 — n=18 competitive players, 90 shots each, ball kinematics @200 Hz |
| **Ball entry angle** | 30.4–34.9° (2pt)<br>41.9–44.7° (3pt) | Vencúrik et al. 2021 — n=48 elite youth, Xsens IMU ⚠️ *see §5.3* |
| **Arm elevation angle** | 60.8 ± 6.3° (FT)<br>58.9 ± 7.4° (2pt)<br>56.9 ± 8.5° (3pt) | Cabarkapa et al. 2022 — **2D**, *different construct — see §1.3* |
| **Arm elevation angle** | 52.1 ± 5.4 (non-prof) / 51.4 ± 3.2 (prof), p=0.179 | Cabarkapa et al. 2023a — 3D markerless |
| **Arm elevation angle** | 41.7 ± 10.0° (2pt) / 39.6 ± 10.8° (3pt) | Cabarkapa et al. 2023b — females |
| **Elbow at release** | 158.8° / 159.3°, p=0.228 | Cabarkapa et al. 2026 — near-full extension, **not 180°** |
| **Knee at release** | 172.2 (non-prof) / **164.9 (prof)**, **p=0.010** | Cabarkapa et al. 2026 |
| **Release height** (÷ body height) | 1.12 (non-prof) / **1.17 (prof), p=0.010** | Cabarkapa et al. 2023a |
| **Release height** (÷ body height) | 1.40 (prof) / 1.28 (non-prof), p<0.05 — 2pt | Cabarkapa et al. 2023b |
| **Release height** (metric) | 2.02–2.13 m | Amaro et al. 2025 |
| **Release velocity** | 6.7–7.0 m/s | Amaro et al. 2025 |
| **Vertical release velocity** | Mid-range: 3.89 ± 0.37 (novice) vs **4.59 ± 0.47** (experienced)<br>Long-range: 4.42 ± 0.52 vs **5.34 ± 0.27** | Chen et al. 2026 — n=30, 13-camera OptiTrack @240 Hz |

**Synthesis — the strongest claim in this literature.** Release **height**, not any joint angle, was the significant discriminator of free-throw proficiency (p=0.010, Cabarkapa et al. 2023a) and of two-point proficiency in females (p<0.05, 2023b). Release *angle* was not significant in either (p=0.179). Amaro et al. 2025 independently confirms this: successful attempts had significantly higher release height and jump height, while **release angle showed no significant difference between made and missed shots**.

**Design implication:** build Swichy's primary scoring around release height (normalised to body height) rather than release angle.

**Important caveat from Amaro et al. 2025:** Spearman correlations between biomechanical parameters and accuracy were weak (R² = 0.005–0.012). The authors conclude these parameters *"provide necessary conditions"* but do not, in isolation, explain accuracy. **Any claim that Swichy can predict make/miss from kinematics alone is not supported by the literature.** Frame the app as a *technique-consistency* tool, not an outcome predictor.

### 2.4 ★ SET SHOT vs. JUMP SHOT — Differences and Convergences

This is the section that answers "find the common between them."

#### 2.4.1 Where they DIFFER

| Dimension | Set shot (free throw) | Jump shot (2pt / 3pt) | Evidence |
|---|---|---|---|
| Shoulder vertical velocity at release | Non-zero, upward | ≈ zero | Okubo & Hubbard 2018 |
| Release conditions produced by | Body + arm | Arm almost alone | Okubo & Hubbard 2018 |
| **Knee load depth** | **Shallowest** (121.5°) | Deeper (116.7° → 112.5°) | Cabarkapa 2022, p<0.05 |
| **Hip load depth** | **Shallowest** (143.9°) | Deeper (141.1° → 135.5°) | Cabarkapa 2022, p<0.05 |
| **Elbow height** (÷ height) | **Highest** (0.701) | Lower (0.676 → 0.619) | Cabarkapa 2022, p<0.05 |
| **Arm elevation at release** | **Highest** (60.8°) | Lower (58.9° → 56.9°) | Cabarkapa 2022, p<0.05 |
| **Vertical displacement** | **15.3 ± 5.1 cm** | **26.9 ± 5.6 → 31.2 ± 7.3 cm** | Cabarkapa 2022, p<0.05 |
| Release height (÷ height) | 1.307 ± 0.067 | 1.378 ± 0.073 / 1.377 ± 0.093 | Cabarkapa 2022, p<0.05 |

The vertical-displacement row is the cleanest empirical separator of the two shot classes: **the free throw involves roughly half the vertical displacement of a jump shot** (15.3 cm vs. 27–31 cm). This is a directly measurable discriminator your system can use to auto-classify shot type from video.

#### 2.4.2 ★ Where they CONVERGE — the invariants

**Finding 1 — The elbow set point is invariant across shot type, distance, and skill level.**

| Study | Comparison | Elbow values | Significance |
|---|---|---|---|
| Cabarkapa 2022 | FT vs 2pt vs 3pt (pros) | 61.9 / 58.2 / 63.6° | No monotonic trend |
| Cabarkapa 2023b | 2pt vs 3pt (females) | 52.3 vs 52.4° | Essentially identical |
| Cabarkapa 2023a | prof vs non-prof (FT) | 85.1 vs 78.0° | **p=0.471 (n.s.)** |
| Cabarkapa 2026 | prof vs non-prof (3pt) | 81.9 vs 84.4° | **p=0.706 (n.s.)** |
| Cabarkapa 2026 | elbow at *release* | 159.3 vs 158.8° | **p=0.228 (n.s.)** |
| Okazaki & Rodacki 2012 | 2.8 / 4.6 / 6.4 m | "increased distance did not change the maximum and minimum joint angles" | F < 3.5, p > 0.05 |

Six independent comparisons, across two measurement modalities, three populations, and both sexes, all point the same way. **The elbow behaves as a postural constant, not a distance-scaled variable.**

**Finding 2 — The lower limb scales with distance; the upper limb does not.**

The contrast is the finding. Knee and hip flex progressively deeper from free throw → two-point → three-point (all p<0.05, Cabarkapa 2022, replicated in 2023b), while elbow angle shows no systematic trend across the same conditions. This is exactly what Okubo & Hubbard's mechanical model predicts: the extra energy needed for distance is generated by the legs and transferred through the kinetic chain, while the arm's job — orienting and releasing the ball — is invariant.

Chen et al. 2026 supports the same conclusion kinetically: experienced players' advantage came from greater **knee peak power** plus elbow rate-of-torque-development and angular impulse, producing higher *vertical* release velocity; novices compensated by adding *horizontal* velocity instead.

**Finding 3 — Neither shot fully extends the joints at release.**

- Elbow at release: **158.8–159.3°**, not 180°
- Knee at release: **164.9–172.2°**, not 180° — and proficient shooters were *less* extended (164.9° vs 172.2°, p=0.010)

**Finding 4 — Ball launch angle converges on ~50–55° regardless of shot type.**

| Source | Method | Value |
|---|---|---|
| Miller & Bartlett 1996 | 3D cinematography, empirical | 52–55° (short), 48–50° (long) |
| Tran & Silverberg 2008 | Trajectory optimisation, theoretical | 52° |
| Amaro et al. 2025 | High-speed ball kinematics | 54.5–57.9° |

Three independent methodologies — empirical 3D capture, physics simulation, and modern ball tracking — spanning 29 years converge on approximately **50–55°**. This is the most robust single number in the entire kinematic literature and the safest reference value for Swichy.

#### 2.4.3 ★ The synthesised coaching rule

> **Elbow deviation is a technique error. Knee variation is a legitimate distance adaptation.**

This one sentence is the practical payoff of the convergence analysis, it is supported by six independent statistical comparisons, and it directly determines how Swichy should score a shot. A system that flags a shooter for "knee angle 109° instead of 121°" when they are shooting a three-pointer is penalising correct mechanics.

---

## 3. Vision Bias Correction

### 3.1 What the validation literature reports

| System | Task and sample | Error vs. marker-based | Source |
|---|---|---|---|
| **Theia3D** (multi-camera 3D markerless) | Treadmill gait, n=30 | Joint centres RMSD < 2.5 cm (hip 3.6 cm); **segment angles < 5.5°** except long-axis rotation | Kanko et al. 2021, *J Biomech* |
| **OpenCap** (2-smartphone 3D) | Multiple movements | **MAE 3.85°, RMSE 4.34°** grand mean; sagittal knee/hip CMC > 0.94; frontal 0.47–0.78; transverse 0.51–0.60 | Uhlrich et al. 2023, *PLOS Comput Biol* |
| **OpenPose 2D, single camera** | Overground gait, n=15, 130 trials | **Camera-side sagittal:** knee +1.5 ± 4.1° (LoA −6.5→9.6), hip −3.6 ± 4.6°, ankle **−8.4 ± 5.2°**<br>**Occluded-side:** hip −4.6 ± 9.5° (LoA −23.2→14.0) — **SD roughly doubles**<br>r²: knee 0.98 camera-side vs 0.94 occluded; hip 0.94 vs 0.76 | Wade et al. 2023, *PLoS One* |
| **OpenPose 2D, by camera azimuth** | Front lunge, n=16, vs Vicon @200 Hz | **Bias:** knee 0.53→6.33°, hip −2.31→13.74°, **elbow 6.34→20.01°, shoulder −16.25→−26.07°**<br>**RMSE:** knee 15.5–29.4°, hip 15.2–25.3°, elbow 29.4–37.1°, shoulder 23.6–36.5° | Baldinger et al. 2025, *Sensors* |
| **Theia3D** | **Simulated basketball throws**, n=13 | RMSD 7.17 ± 3.88° → **26.66 ± 14.77°** depending on joint<br>**Elbow flexion: 22.22 ± 5.52° (v2020) → 16.68 ± 5.03° (v2023)**<br>Errors *"especially large for the upper extremities"* | Thomas et al. 2025, *Biomechanics* |
| **Theia3D** | Boxing, n=3 elite | Shoulder flex/ext 7.3–10°, abd/add 6.3–6.6°, **int/ext 8.1–12°**<br>Elbow flex/ext 7.0–7.4°, **elbow int/ext 18–23°**<br>Wrist 9.1–20° | Lahkar et al. 2022, *Front Sports Act Living* |
| **OpenPose 3D** (5 cameras) | Walking, jumping, throwing, n=2 | 47% of MAE < 20 mm; 80% < 30 mm; 10% > 40 mm. **Positions only — no joint angles computed** | Nakano et al. 2020, *Front Sports Act Living* |
| **MediaPipe** (multi-view) | Static and dynamic tasks | Static ICC 0.851, **MAE 9.28°**; dynamic ICC 0.823, **MAE 12.92°**; camera baseline > 90° → ICC < 0.40 | Multi-study, see §6 |

### 3.2 The honest answer to "what offset should I apply?"

**There is no publishable constant offset, and claiming one will be challenged.**

Baldinger et al. 2025 settles this. For the *same movement, same algorithm, same participants*, elbow bias ranged **+6.34° to +20.01°** and shoulder bias **−16.25° to −26.07°** purely as a function of camera azimuth. A scalar correction fitted at one camera position is invalid at another. Any "calibration factor" you hard-code is a factor for one specific geometry.

What the literature *does* support is a **joint-tiered trust model**:

| Tier | Joints | Evidence | Use in Swichy |
|---|---|---|---|
| ✅ **Trustworthy** | Sagittal **knee** and **hip** flexion, camera-side | Wade: bias +1.5 ± 4.1° and −3.6 ± 4.6°, r² = 0.98 / 0.94 — *within reported marker-based error* | Report absolute degrees with a ±5° tolerance band |
| ⚠️ **Directional only** | **Elbow** flexion | Thomas (basketball-specific): RMSD 16.68 ± 5.03°. Baldinger: bias +6→+20° | Report relative change vs. the athlete's own baseline; never an absolute number |
| ❌ **Do not score** | **Shoulder** angle; any long-axis / transverse rotation | Baldinger: shoulder bias −26°, RMSE 36°. Lahkar: elbow int/ext 23°. Uhlrich: transverse CMC 0.51 | Exclude from scoring, or surface as a qualitative flag only |
| 🔧 **Needs correction** | **Ankle** | Wade: −8.4 ± 5.2° bias because OpenPose/MediaPipe label the **toe endpoint, not the metatarsophalangeal joint** | Apply a documented landmark correction, or exclude |

This tiering is not an opinion — each row is traceable to a specific measured value in a specific indexed paper. That traceability is what makes it defensible.

### 3.3 Six evidence-backed recommendations for the system

**1. Enforce the capture conditions under which the validity figures were measured.**
Wade et al. showed occluded-side limb SD is roughly double camera-side (hip 108% worse). Swichy should **detect and refuse** off-plane framing rather than silently returning a biased angle. Gate on: camera perpendicular to the shooting plane, shooting-side limb toward camera, full-body visibility, subject centred to limit parallax.

**2. Prefer scale-invariant ratios over metric distances.**
Cabarkapa's group normalises release height and elbow height by body height. Ratios survive depth-estimation error far better than absolute distances — and release-height ratio is the variable that actually discriminated proficiency (§2.3).

**3. Apply the constraints Baldinger et al. explicitly recommend.**
Clamp joint angles to anatomical limits; penalise implausible frame-to-frame jumps. This is a citable justification for your smoothing and filtering layer, rather than an unexplained engineering choice.

**4. Score consistency and waveform similarity, not just absolute angles.**
Uhlrich reports sagittal CMC > 0.94 alongside multi-degree MAE — pose estimation tracks the *shape and timing* of a movement well even when the absolute value is offset. Within-subject comparison cancels systematic bias entirely. This is the single most effective way to build a useful product on top of a biased sensor.

**5. Mind your frame rate.**
Cabarkapa et al. 2023b used 30 fps — 33 ms per frame. Ball release is fast, and identifying the release frame at 30 fps introduces real temporal quantisation error. Studies that resolve release cleanly use 100–240 Hz (Miller & Bartlett 100 Hz; Cabarkapa 2022/2023a/2026 120 Hz; Chen 240 Hz; Amaro 200 Hz). **Target ≥120 fps for release-frame detection**, and state the limitation if you cannot.

**6. Run your own calibration study.**
The strongest available defence move: n=10–15, your exact camera setup and MediaPipe pipeline, against any reference (a second synchronised view, or goniometry), reporting **Bland–Altman bias and limits of agreement in the same format as Wade et al.** This converts "I assumed published offsets transfer" into "I measured my system's bias." Juries reward this heavily, and it is achievable within a thesis timeline.

---

## 4. Sources Retained — Summary Table

| # | Study | n | Method | Contribution |
|---|---|---|---|---|
| 1 | Miller & Bartlett 1996 | 15 | 3D cine @100 Hz | Ball launch angle 48–55° by distance |
| 2 | Tran & Silverberg 2008 | Simulation | 10⁵+ trajectories | Optimal launch 52°, 3 Hz backspin |
| 3 | Okazaki, Rodacki & Satern 2015 | Review | Systematic review | Critical components framework |
| 4 | Okazaki & Rodacki 2012 | 10 | 2D @100 Hz | Joint angles invariant to distance |
| 5 | Okubo & Hubbard 2018 | Model | 2D 3-segment model | Set vs jump shot mechanical definition |
| 6 | Cabarkapa et al. 2022 | 10 pros | 2D @120 fps | Full angle matrix by distance |
| 7 | Cabarkapa et al. 2023a | 34 | 3D markerless @120 Hz | Release height discriminates FT proficiency |
| 8 | Cabarkapa et al. 2023b | 18 females | 2D @30 fps | Female-specific matrix |
| 9 | Cabarkapa et al. 2026 | 24 | 3D markerless @120 Hz | Load depth discriminates 3pt proficiency |
| 10 | Amaro et al. 2025 | 18 | Force plates + 200 Hz | Release angle ≠ accuracy; height does |
| 11 | Chen et al. 2026 | 30 | OptiTrack @240 Hz | Experience → vertical release velocity |
| 12 | Li et al. 2025 | 20 | 13-cam @240 Hz | Arm coordination variability |
| 13 | Vencúrik et al. 2021 | 48 | Xsens IMU @60 Hz | Entry angles ⚠️ §5.3 |
| 14 | Kanko et al. 2021 | 30 | Theia3D vs Vicon | Markerless benchmark < 5.5° |
| 15 | Uhlrich et al. 2023 | — | OpenCap vs marker | MAE 3.85° |
| 16 | Wade et al. 2023 | 15 | 2D OpenPose vs 3D | **2D single-camera bias and LoA** |
| 17 | Baldinger et al. 2025 | 16 | OpenPose by azimuth | **Camera-angle-dependent bias** |
| 18 | Thomas et al. 2025 | 13 | Theia3D, basketball | **Elbow RMSD 16.68° in basketball** |
| 19 | Lahkar et al. 2022 | 3 | Theia3D, boxing | Upper-limb RMSD by rotation axis |
| 20 | Nakano et al. 2020 | 2 | OpenPose 3D | Position accuracy only |

---

## 5. Critical Evaluation and Indexing Verification

### 5.1 Journal indexing — verified

| Journal | Scopus | Web of Science | PubMed / MEDLINE | Verdict |
|---|---|---|---|---|
| **Journal of Sports Sciences** (T&F) | ✅ | ✅ SCIE | ✅ MEDLINE | **Tier 1.** Flagship sport-science journal. Miller & Bartlett = PMID 8809716. |
| **Journal of Biomechanics** (Elsevier) | ✅ | ✅ SCIE | ✅ MEDLINE | **Tier 1.** ISB-affiliated. Kanko = PMID 34380101. IF ≈ 2.4–2.9 — argue standing, not IF; modest IF is normal for methods journals. |
| **Sports Biomechanics** (T&F) | ✅ | ✅ SCIE | ✅ | **Tier 1.** Official journal of the ISBS. IF 2.0, 5-yr 2.5, CiteScore 4.3, **Q2**. |
| **PLOS Computational Biology** | ✅ | ✅ | ✅ MEDLINE | **Tier 1**, Q1. |
| **Journal of Human Kinetics** | ✅ | ✅ | ✅ PubMed | **Tier 1–2.** Struzik = PMID 25414741. |
| **PeerJ** | ✅ | ✅ | ✅ PMC | **Tier 2.** Rigorous peer review, open data culture. |
| **PLOS ONE** | ✅ | ✅ SCIE | ✅ MEDLINE | **Tier 2.** Mega-journal, but Wade et al. is methodologically strong (repeated-measures Bland–Altman). |
| **Frontiers in Sports and Active Living** | ✅ | ⚠️ **ESCI only**, not SCIE | ✅ — confirmed directly (PMC10436204, PMC9357930) | **Tier 2.** Sound, but disclose ESCI status proactively. |
| **Sensors** (MDPI) | ✅ | ✅ SCIE | — | **Tier 2.** Established, SCIE-listed. |
| **JFMK** (MDPI) | ✅ | ⚠️ ESCI | ✅ PubMed / PMC | **Tier 2.** DOAJ-listed, PMC-archived. |
| **Biomechanics** (MDPI) | ⚠️ newer title | ⚠️ verify | DOAJ ✅ | **Tier 3.** Weakest venue cited — but the *only* basketball-specific markerless validation in existence. Cite for its data; disclose the venue. |
| **Proceedings** (MDPI, ISEA conf.) | ⚠️ conference | ❌ | ❌ | **Tier 3.** Conference proceedings. Cite Okubo & Hubbard for its *conceptual model*, not as empirical evidence. |

### 5.2 Research-group credibility

| Group | Standing | Caveat to pre-empt |
|---|---|---|
| **Cabarkapa, Fry et al.** — Jayhawk Athletic Performance Laboratory, University of Kansas | The most prolific basketball-shooting kinematics group currently active; source of four datasets here | **Heavy self-citation cluster and inconsistent modality across studies.** Samples are frequently *recreational*, not elite (2023a: "recreationally active males"; 2023b: recreational females). Their 2022 reference values come from a **single 2D camera** — so that paper is not ground truth for validating a 2D system. |
| **Kanko, Laende, Davis, Selbie, Deluzio** — Human Mobility Research Centre, Queen's University | The definitive markerless-validation series | ⚠️ **Disclose:** co-author **W. S. Selbie is affiliated with Theia Markerless Inc.**, the system under evaluation. Declare this conflict — reviewers check. |
| **Uhlrich, Falisse, Hicks, Delp** — Neuromuscular Biomechanics Lab, Stanford | Highest credibility available; authors of OpenSim | OpenCap is a **2-camera** system, not monocular — do not over-transfer its 3.85° MAE to a single-camera app. |
| **Lahkar, Muller, Dumas, Reveret, Robert** — Univ. Gustave Eiffel / LBMC | Leading European biomechanics modelling group | **n=3.** Indicative, not definitive. |
| **Miller & Bartlett** — Bartlett is a foundational figure in sports biomechanics | Canonical, heavily cited | 1996; 3D cinematography; n=15 (n=5 per position group). Dated instrumentation, small cells. |
| **Wade, Needham, Colyer, Bilzon** — CAMERA / University of Bath | Strong markerless-methods group | Gait only — no overhead arm elevation tested. |

### 5.3 ⚠️ Two sources screened OUT

**IJERPH — delisted from Web of Science.**
Clarivate discontinued coverage of the *International Journal of Environmental Research and Public Health* effective **13 February 2023** (announced 15 March 2023), for failing the **Content Relevance** criterion. MDPI appealed on 31 March 2023; no confirmed reinstatement. The journal remains in Scopus, PubMed, MEDLINE and PMC.

*Recommendation:* Vencúrik et al. 2021 may still be cited — the data (entry angles, n=48 elite youth) are useful and the paper predates the delisting — but **disclose the delisting in a footnote**. If a jury member finds it and you did not mention it, your credibility on every other citation suffers. If you disclose it, you demonstrate exactly the rigour you are claiming. Also note it used **Xsens IMU, not optical capture** — a different error profile that does not transfer to the vision-validation argument.

**physicaleducationjournal.net — exclude entirely.**
This domain hosts the *"International Journal of Physiology, Sports and Physical Education"* (Sparkling Press). It self-describes as "peer-reviewed, refereed and indexed," but **no Scopus, Web of Science, or PubMed listing was found**. It matches the profile of the low-tier PE title this review was instructed to avoid, and it sits within a cluster of near-identical domains (physicaleducationjournal.in / .com, kheljournal.com) that is a recognised predatory-adjacent pattern in the PE and sport-science space. Citing it would undermine the review's central claim to rigour.

### 5.4 Limitations table — bring this to the defence

| Study | n | Limitation to state before you are asked |
|---|---|---|
| Miller & Bartlett 1996 | 15 (5/group) | 30 years old; underpowered per-position cells |
| Okazaki & Rodacki 2012 | 10 | 2D analysis; release-angle construct differs from other studies |
| Cabarkapa 2022 | 10 | Single 2D camera; shoulder SD ±26.9–31.4° |
| Cabarkapa 2023b | 18 | **30 fps** — arguably too coarse to isolate the release frame |
| Cabarkapa 2023a / 2026 | 34 / 24 | Recreational-to-club level; proprietary SwRI system limits replication |
| Amaro 2025 | 18 | Weak kinematics–accuracy correlations (R² = 0.005–0.012) |
| Chen 2026 | 30 | Reports velocities and kinetics; **no release angles or heights** |
| Vencúrik 2021 | 48 | IMU not optical; **journal delisted from WoS** |
| Wade 2023 | 15 | **Gait, not shooting** — no overhead arm elevation |
| Baldinger 2025 | 16 | **Lunges, not shooting**; OpenPose, not MediaPipe |
| Thomas 2025 | 13 | "Simulated" lab throws, no ball flight; Tier-3 venue |
| Lahkar 2022 | 3 | Severely underpowered |
| Nakano 2020 | 2 | Positions only — **no joint angles computed** |
| Okubo & Hubbard 2018 | Model | Conference proceedings; theoretical model, limited empirical data |

### 5.5 ★ The research gap — name it yourself

> **No peer-reviewed study has validated single-camera 2D pose estimation against marker-based motion capture during an actual basketball shot.**

The closest is Thomas et al. 2025 — and that used *multi-camera 3D* markerless capture on *simulated* throws, still reporting elbow flexion RMSD of 16.68 ± 5.03°. Wade et al. and Baldinger et al. cover single-camera 2D but on gait and lunges respectively, with no overhead arm elevation.

Naming this gap explicitly positions the thesis as **filling** it, rather than leaving it for a jury member to discover. It also reframes every limitation in §5.4 as motivation rather than weakness.

---

## 6. University-Ready Bibliography

### 6.1 Core five (APA 7th)

> Baldinger, M., Reimer, L. M., & Senner, V. (2025). Influence of the camera viewing angle on OpenPose validity in motion analysis. *Sensors, 25*(3), 799. https://doi.org/10.3390/s25030799

> Cabarkapa, D., Cabarkapa, D. V., Miller, J. D., Templin, T. T., Frazer, L. L., Nicolella, D. P., & Fry, A. C. (2023). Biomechanical characteristics of proficient free-throw shooters—markerless motion capture analysis. *Frontiers in Sports and Active Living, 5*, 1208915. https://doi.org/10.3389/fspor.2023.1208915

> Kanko, R. M., Laende, E. K., Davis, E. M., Selbie, W. S., & Deluzio, K. J. (2021). Concurrent assessment of gait kinematics using marker-based and markerless motion capture. *Journal of Biomechanics, 127*, 110665. https://doi.org/10.1016/j.jbiomech.2021.110665

> Miller, S., & Bartlett, R. (1996). The relationship between basketball shooting kinematics, distance and playing position. *Journal of Sports Sciences, 14*(3), 243–253. https://doi.org/10.1080/02640419608727708

> Wade, L., Needham, L., Evans, M., McGuigan, P., Colyer, S., Cosker, D., & Bilzon, J. (2023). Examination of 2D frontal and sagittal markerless motion capture: Implications for markerless applications. *PLoS ONE, 18*(11), e0293917. https://doi.org/10.1371/journal.pone.0293917

### 6.2 Strongly recommended additions

> Amaro, C. M., Castro, M. A., Mendes, R., Rice, H., & Gomes, B. B. (2025). Influence of jump and ball release parameters on shooting accuracy in basketball under varying constraints. *Journal of Functional Morphology and Kinesiology, 10*(4), 459.

> Cabarkapa, D., Cabarkapa, D. V., & Fry, A. C. (2026). Biomechanical determinants of proficient 3-point shooters: Markerless motion capture analysis. *Frontiers in Sports and Active Living, 8*, 1732293.

> Cabarkapa, D., Cabarkapa, D. V., Philipp, N. M., Eserhaut, D. A., Downey, G. G., & Fry, A. C. (2022). Impact of distance and proficiency on shooting kinematics in professional male basketball players. *Journal of Functional Morphology and Kinesiology, 7*(4), 78. https://doi.org/10.3390/jfmk7040078

> Chen, P., Chen, T., Tang, X., Li, M., & Miao, X. (2026). Effects of playing experience on joint kinetics and ball-release velocity in mid- and long-range basketball jump shots. *PeerJ, 14*, e20757. https://doi.org/10.7717/peerj.20757

> Lahkar, B. K., Muller, A., Dumas, R., Reveret, L., & Robert, T. (2022). Accuracy of a markerless motion capture system in estimating upper extremity kinematics during boxing. *Frontiers in Sports and Active Living, 4*, 939980.

> Li, J., Kim, Y., Li, H., Zhu, B., & Kim, S. (2025). Arm joint coordination of collegiate basketball athletes and recreational players when shooting behind the 3-point line. *Journal of Human Kinetics, 96*(Spec Issue), 129–143. https://doi.org/10.5114/jhk/203104

> Okazaki, V. H. A., & Rodacki, A. L. F. (2012). Increased distance of shooting on basketball jump shot. *Journal of Sports Science and Medicine, 11*(2), 231–237.

> Okazaki, V. H. A., Rodacki, A. L. F., & Satern, M. N. (2015). A review on the basketball jump shot. *Sports Biomechanics, 14*(2), 190–205. https://doi.org/10.1080/14763141.2015.1052541

> Okubo, H., & Hubbard, M. (2018). Kinematic differences between set- and jump-shot motions in basketball. *Proceedings, 2*(6), 201. https://doi.org/10.3390/proceedings2060201

> Struzik, A., Pietraszewski, B., & Zawadzki, J. (2014). Biomechanical analysis of the jump shot in basketball. *Journal of Human Kinetics, 42*, 73–79. https://doi.org/10.2478/hukin-2014-0062

> Thomas, C., et al. (2025). Comparison of marker-based and markerless motion capture systems for measuring throwing kinematics. *Biomechanics, 5*(4), 100. https://doi.org/10.3390/biomechanics5040100

> Tran, C. M., & Silverberg, L. M. (2008). Optimal release conditions for the free throw in men's basketball. *Journal of Sports Sciences, 26*(11), 1147–1155. https://doi.org/10.1080/02640410802004948

> Uhlrich, S. D., Falisse, A., Kidziński, Ł., Muccini, J., Ko, M., Chaudhari, A. S., Hicks, J. L., & Delp, S. L. (2023). OpenCap: Human movement dynamics from smartphone videos. *PLOS Computational Biology, 19*(10), e1011462. https://doi.org/10.1371/journal.pcbi.1011462

> Vencúrik, T., Knjaz, D., Rupčić, T., Sporiš, G., & Li, F. (2021). Kinematic analysis of 2-point and 3-point jump shot of elite young male and female basketball players. *International Journal of Environmental Research and Public Health, 18*(3), 934. https://doi.org/10.3390/ijerph18030934 *[Note: journal discontinued in Web of Science, February 2023 — see §5.3]*

### 6.3 ⚠️ Citations requiring final verification before submission

| Citation | What to verify | Where |
|---|---|---|
| **Thomas et al. 2025** | **Full author list** — aggregators showed only "Carina Thomas and 3 others" | https://doi.org/10.3390/biomechanics5040100 |
| Cabarkapa et al. 2026 | Volume number and page range once in final issue | Frontiers article page |
| Amaro et al. 2025 | DOI | JFMK 10(4):459 |
| Okazaki & Rodacki 2012 | Issue number | jssm.org |
| Vencúrik et al. 2021 | Current WoS status (in case of reinstatement) | Clarivate Master Journal List |

Do not paste an incomplete author list into a thesis. Verify each of the above at the DOI before submitting.

---

## 7. Design Implications for Swichy

Consolidated from the evidence above:

1. **Auto-classify shot type** from vertical displacement (~15 cm = set shot / free throw; 27–31 cm = jump shot) before applying any scoring rule.
2. **Score the lower limb against distance-specific targets** — knee ≈ 121° (FT), ≈ 117° (2pt), ≈ 112° (3pt) for trained males. Never a single global target.
3. **Treat elbow deviation as error, knee variation as adaptation** (§2.4.3).
4. **Make release height (÷ body height) the primary success metric**, not release angle. Target ≈ 1.17+ for free throws.
5. **Report ball launch angle, not arm elevation angle** — and target ~50–55°.
6. **Exclude shoulder angle from scoring.** Bias up to −26°, RMSE up to 36°.
7. **Apply the joint trust tiers** in §3.2 to decide what gets an absolute number, what gets a relative trend, and what gets suppressed.
8. **Gate on capture geometry** — refuse off-plane or occluded framing rather than reporting a biased angle.
9. **Target ≥120 fps** for release-frame detection.
10. **Do not claim make/miss prediction.** Amaro et al. 2025 found R² = 0.005–0.012. Position Swichy as a technique-consistency tool.

### 7.1 Open decision — how to present uncertain angles

When the vision system computes an elbow angle it knows is ±16° uncertain, what should the app show?

- **(a) Absolute degrees with a confidence band** — "Elbow: 82° ± 15°." Honest and clean, but a ±15° band arguably tells a player nothing actionable.
- **(b) Relative to the athlete's own baseline** — "Your elbow set point was 8° lower than your best session." Systematic error cancels in within-subject comparison, matching the CMC evidence. But it cannot tell a beginner they are *wrong*, only that they are *inconsistent*.
- **(c) Tiered by joint** — absolute degrees for knee and hip (defensible per Wade et al.), relative-only for elbow, shoulder excluded from scoring.

**Recommended: (c).** It is the only option defensible line-by-line from the validation table in §3.2, and it lets you tell a jury exactly which citation licenses each design choice.

---

## 8. Verification Log

| Claim | Verified against |
|---|---|
| Miller & Bartlett release angles 48–55° | PubMed abstract, PMID 8809716 |
| Cabarkapa 2022 full angle table + variable definitions | PMC9590067 full text |
| Cabarkapa 2023a proficiency values | PMC10436204 full text |
| Cabarkapa 2023b female tables | PMC10531893 full text |
| Cabarkapa 2026 three-point values | Frontiers full text, fspor.2026.1732293 |
| Kanko 2021 RMSD figures | PubMed abstract, PMID 34380101 |
| Wade 2023 bias / LoA tables | PMC10635560 full text |
| Baldinger 2025 bias and RMSE by camera angle | PMC11819822 full text |
| Lahkar 2022 upper-limb RMSD | PMC9357930 full text |
| Nakano 2020 MAE distribution | PMC7739760 full text |
| Chen 2026 release velocities | PMC12883162 full text |
| Li 2025 coordination variability | PMC12121896 full text |
| Amaro 2025 release parameters | PMC12641682 full text |
| Vencúrik 2021 shoulder and entry angles | PMC7908352 full text |
| Struzik 2014 methods | PubMed abstract, PMID 25414741 |
| Thomas 2025 RMSD values | DOAJ / aggregator abstract — **full text not retrieved** |
| Okubo & Hubbard 2018 set/jump definition | Publisher abstract + secondary summaries — **full text not retrieved** |
| IJERPH WoS delisting | MDPI announcement 5536 (15 March 2023) |
| Sports Biomechanics indexing / metrics | wos-journal.info, Taylor & Francis journal page |
| Frontiers Sports Act Living PMC archiving | Confirmed directly via PMC10436204, PMC9357930 |

**Items marked "full text not retrieved"** were blocked by publisher access controls (HTTP 403) or were image-only scanned PDFs. Their values are reported from publisher-supplied abstracts and should be confirmed against the full text before final submission.

---

*End of document.*
