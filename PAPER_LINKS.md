# Swichy — Paper Links

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
