# Future Improvements

This file is append-only. Each enhancement includes description, difficulty, expected gain, required knowledge, research papers, prerequisites, and suggested implementation order.

---

## 1. Multi-Camera Calibration

**Description:** Use 2+ synchronized cameras with known intrinsics/extrinsics to triangulate true 3D joint positions, eliminating monocular depth ambiguity.

**Difficulty:** Hard

**Expected performance gain:** High — near ground-truth 3D accuracy for joint angles.

**Required knowledge:** Camera calibration, stereo vision, triangulation, OpenCV `calibrateCamera`, extrinsic matrix estimation.

**Research papers:**
- Zhang, "A Flexible New Technique for Camera Calibration" (2000)
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*

**Prerequisites:** Phase 1–2 complete, understanding of world vs image coordinates.

**Implementation order:** Late (after single-camera pipeline is stable).

---

## 2. Transformer-Based Movement Analysis

**Description:** Replace rule-based phase detection with a Transformer encoder that learns shot phase sequences from labeled video data.

**Difficulty:** Hard

**Expected performance gain:** High for phase classification accuracy on diverse shooters.

**Required knowledge:** Attention mechanisms, sequence modeling, PyTorch/TensorFlow, dataset labeling.

**Research papers:**
- Vaswani et al., "Attention Is All You Need" (2017)
- PoseFormer: "Pose Transformer for 3D Human Pose Estimation" (2021)

**Prerequisites:** Phase 3 rule-based FSM working, labeled dataset of 500+ shots.

**Implementation order:** Phase 6+ (after rule-based system provides baseline + training labels).

---

## 3. LSTM Sequence Classification

**Description:** LSTM network on angle time-series to classify shot phases or detect errors.

**Difficulty:** Medium–Hard

**Expected performance gain:** Medium — better than hand-crafted thresholds for varied shooting styles.

**Required knowledge:** RNNs, LSTM gates, sequence padding, loss functions for classification.

**Research papers:**
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997)
- Human activity recognition surveys using wearable/vision sensors

**Prerequisites:** Frame buffer with labeled phase sequences.

**Implementation order:** After Phase 3, before or parallel to Transformers.

---

## 4. Reinforcement Learning Coach

**Description:** RL agent learns optimal coaching feedback policy by maximizing long-term shooting improvement across sessions.

**Difficulty:** Very Hard

**Expected performance gain:** High long-term — personalized adaptive coaching.

**Required knowledge:** MDPs, Q-learning/PPO, reward shaping, simulation environments.

**Research papers:**
- Sutton & Barto, *Reinforcement Learning: An Introduction*
- RL for sports coaching surveys

**Prerequisites:** Scoring system, player session history, simulation or real feedback loop.

**Implementation order:** Late commercial feature.

---

## 5. Personalized Player Models

**Description:** Per-player calibration of acceptable angle ranges based on their body proportions, shooting style, and historical performance.

**Difficulty:** Medium

**Expected performance gain:** Medium — reduces false positives from one-size-fits-all rules.

**Required knowledge:** Online learning, statistical profiling, clustering.

**Prerequisites:** Phase 4 rule engine, session storage.

**Implementation order:** After Phase 4–5.

---

## 6. Ball-Body Synchronization

**Description:** Detect ball position (color segmentation, YOLO, or dedicated ball tracker) and synchronize with wrist/elbow kinematics for release analysis. **Full Phase 6 plan:** [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) (ball detection, timeseries outcome, make/miss).

**Difficulty:** Medium–Hard

**Expected performance gain:** High — enables release angle, ball velocity proxy, shot arc estimation.

**Required knowledge:** Object detection, multi-object tracking, temporal alignment.

**Research papers:**
- Basketball ball tracking CVPR workshop papers
- YOLOv8/v9 documentation

**Prerequisites:** Stable body pose pipeline.

**Implementation order:** Phase 5+.

---

## 7. Release Timing Prediction

**Description:** Predict release frame 2–5 frames before it happens using wrist velocity/acceleration patterns.

**Difficulty:** Medium

**Expected performance gain:** Medium — enables anticipatory coaching cues.

**Required knowledge:** Time-series forecasting, derivative features, causal filtering.

**Prerequisites:** Phase 3 phase detection, frame buffer.

**Implementation order:** After Phase 3.

---

## 8. Automatic Shot Classification

**Description:** Classify shot types: catch-and-shoot, off-dribble, fadeaway, free throw, layup.

**Difficulty:** Medium

**Expected performance gain:** Medium — context-aware coaching rules per shot type.

**Required knowledge:** Video classification, feature engineering from pose sequences.

**Prerequisites:** Phase 3 phase detection, labeled shot type dataset.

**Implementation order:** After Phase 3–4.

---

## 9. Professional Player Similarity Scoring

**Description:** Compare user's kinematic profile to reference poses of professional shooters (e.g. Curry, Durant) using DTW or embedding distance.

**Difficulty:** Medium–Hard

**Expected performance gain:** Medium — motivational and educational feedback.

**Required knowledge:** Dynamic Time Warping, pose embeddings, reference dataset curation.

**Research papers:**
- DTW for time-series similarity
- Pose embedding methods (PoseNet, etc.)

**Prerequisites:** Phase 3–4, curated pro player reference clips.

**Implementation order:** Phase 5+.

---

## 10. Kinematic Chain Optimization

**Description:** Model the kinetic chain (ankles → knees → hips → trunk → shoulder → elbow → wrist) and identify energy leaks or sequencing errors.

**Difficulty:** Hard

**Expected performance gain:** High — biomechanically meaningful coaching beyond single-joint rules.

**Required knowledge:** Biomechanics, sequential correlation, peak timing analysis.

**Research papers:**
- Basketball shooting biomechanics reviews
- Kinetic chain analysis in sports

**Prerequisites:** Phase 3–4, all joint angles stable.

**Implementation order:** After Phase 4.

---

## 11. Physics-Based Trajectory Estimation

**Description:** Estimate ball trajectory from release pose + ball detection using projectile motion equations. Builds on Phase 6 timeseries — see [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md).

**Difficulty:** Medium–Hard

**Expected performance gain:** High — predicts make/miss probability, optimal arc.

**Required knowledge:** Projectile physics, parabolic fitting, camera calibration for real-world scale.

**Prerequisites:** Ball detection (#6), release phase detection.

**Implementation order:** After #6 and Phase 3.

---

## 12. Real-Time Edge Deployment

**Description:** Deploy pipeline on mobile (TensorFlow Lite, MediaPipe Android/iOS) or edge devices (Jetson, Coral TPU).

**Difficulty:** Medium–Hard

**Expected performance gain:** High for product — no cloud dependency, lower latency.

**Required knowledge:** Model quantization, ONNX/TFLite conversion, mobile development.

**Prerequisites:** Stable Python pipeline, performance profiling.

**Implementation order:** After Phase 5.

---

## 13. 3D Skeleton Reconstruction

**Description:** Full 3D mesh or skeleton visualization in a virtual environment for immersive coaching.

**Difficulty:** Medium

**Expected performance gain:** Low accuracy gain, high UX gain.

**Required knowledge:** 3D graphics (Three.js, Unity), SMPL body model.

**Prerequisites:** World landmarks pipeline.

**Implementation order:** Optional UX enhancement.

---

## 14. Depth Estimation

**Description:** Augment RGB with learned depth maps (MiDaS, ZoeDepth) for improved Z accuracy.

**Difficulty:** Medium

**Expected performance gain:** Medium for monocular setups with poor MediaPipe depth.

**Required knowledge:** Monocular depth estimation, depth-to-point-cloud conversion.

**Research papers:**
- MiDaS, ZoeDepth papers

**Prerequisites:** Phase 1–2.

**Implementation order:** Optional enhancement before multi-camera.

---

## 15. Multi-Person Tracking

**Description:** Track multiple players in a gym setting; select active shooter.

**Difficulty:** Medium–Hard

**Expected performance gain:** Required for team practice scenarios.

**Required knowledge:** Multi-pose estimation, person Re-ID, tracking (DeepSORT).

**Prerequisites:** Single-person pipeline stable.

**Implementation order:** Commercial v2.

---

## 16. Coach Dashboard

**Description:** Web dashboard showing session history, angle trends, phase breakdowns, improvement over time.

**Difficulty:** Medium

**Expected performance gain:** High for product — coaches need aggregate views.

**Required knowledge:** Web development (React/FastAPI), database design, data visualization.

**Prerequisites:** Phase 4–5, session persistence.

**Implementation order:** After Phase 5.

---

## 17. AI-Generated Coaching Feedback

**Description:** LLM generates natural-language coaching tips from structured `RuleResult` data and player history.

**Difficulty:** Medium

**Expected performance gain:** High UX — personalized, conversational feedback.

**Required knowledge:** LLM APIs, prompt engineering, structured output.

**Prerequisites:** Phase 4–5 rule engine producing structured results.

**Implementation order:** After Phase 5.

---

## Suggested Implementation Order (Summary)

1. **Phase 3:** Temporal phase detection (FSM) — **done**
2. **Phase 4:** Biomechanical rule engine — **done**
3. **Phase 5:** Feedback + scoring — **done**
4. **Phase 5b:** PDF session reports + performance plan — **done**
5. **Phase 6a:** Ball/rim detection + tracking — **integrated**
6. **Phase 6b–6d:** Release sync + validated make/miss fusion — **in progress** — [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md)
7. **#11 Physics trajectory** — **experimental in `physics/`**
8. Release timing prediction (#7)
9. Personalized player models (#5)
10. Shot classification (#8) — use assets README to label dunk vs jump shot
11. Kinematic chain optimization (#10)
12. Coach dashboard (#16)
13. Multi-camera (#1) / depth estimation (#14)
14. ML-based phase detection (#2, #3)
15. Edge deployment (#12)

**Study and complete manually:** [MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md) · **Product + team plan:** [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md)
