# Final Candidate: Decision Report, Audit, Change Log and Completion Checklist

Updated 2026-10-23 after the final submission audit, for Ramesh Yadav (supervisor: Dr. Sachin Agrawal).
Manuscript: `claude/IEEE_Paper_Final_Candidate.md` (new file). `IEEE_Paper_Draft.md`, `IEEE_Paper_Revised.md` and `IEEE_Paper_Final_Handoff.md` are preserved unchanged. Phase 5 datasets, folds, hyperparameters, scripts and result tables are unchanged. All new work is in `final_candidate/`.

**Labels:** [REPRODUCED] re-derived from raw or saved data this sprint · [SAVED] read from existing saved tables, not re-derived · [EXECUTED] computed this sprint · [PROPOSED] not executed.

---

## 1. Final recommendation

**Ready for supervisor review. Not ready for submission until the author actions in §5 are done.** No unresolved issue blocks the supervisor review.

One question is for the supervisor, and it is a judgement call, not a defect that can be fixed before the deadline. The benchmark's ground truth (Rule B) breaks ties between configurations using BER margins far below 10⁻²⁰, which have no operational meaning. The manuscript discloses this (§VII-L, §IX). If the supervisor judges that a benchmark built this way cannot support the paper, the only real fix is a floor-aware ground truth. That is a new benchmark, so it is outside this sprint and not feasible by the deadline.

**Completion path.** Option A (finish from existing Phase 5 results), plus the one small Option B rerun already executed. Option C (the Phase 6 mitigation study) is dropped for this paper. No further experiment is needed. Phase 6 has not started.

---

## 2. What the final audit found (all from existing outputs, no training)

| # | Mandatory issue | What I inspected | Finding | Action |
|---|---|---|---|---|
| 1 | The "24.0% oracle ceiling" | `rule_b()` in `02_*.py`, `03_*.py` and `10_*.py` against `ml_rule_b_decision()` in `phase2b1_full_lowco_audit.py` and `select_configuration()` in `fso_physics_model.py`; all 10,433 environments | The replay uses the same threshold, tie tolerance (1e-9), fixed order and fallback as the original rule. Applied to the **raw** BER it reproduces the stored ground-truth label in 10,433 of 10,433 environments. Applied to the **clipped** BER it matches in 24.0% macro / 49.8% micro. **All 5,239 mismatches are exact ties at the −20 floor** (none in infeasible, single-wavelength or bit-rate-determined environments). With the tie-break order reversed (1550 nm first) the same replay matches in **100.0% macro / 99.9% micro**. | It is **not an upper bound**. It is the accuracy of one fixed policy (Rule B, 850-nm-first tie-break, on clipped inputs) and depends on that arbitrary tie-break. Other policies exceed it: the constant (1550, 16) policy 79.9% macro, the direct Random Forest 82.9%, and Linear Regression in 6 of 8 regimes. Renamed "clipped-truth replay (diagnostic)". The reversed-order figure is labelled a post hoc illustration chosen with knowledge of the labels. |
| 2 | Censoring-equivalent metric | `tied_set()` in `02_*.py`; `rule_b()` returning the tied set in `03_*.py` and `10_*.py` | The score is membership of the selected configuration in the set Rule B cannot distinguish on the clipped truth (same bit-rate tier, margins equal within 1e-9). The true configuration is **never outside the set: 0 of 10,433 environments** (by construction, since the true choice maximizes the raw margin). The set has 2 or more members in **5,239 environments** (3 members in 2,973), which includes every Clear, Visibility, Rain: Light and Rain: Moderate environment. It credits arbitrary choices: constant (1250, 16), never the true label, scores **70.5% macro / 40.4% micro**, and constant (850, 16) scores 53.0% / 28.6%. | **Demoted** to "tie-tolerant agreement", a lenient diagnostic. It is explicitly "not accuracy". It is removed from the abstract headline, contributions and conclusion as evidence of model quality, and is no longer used to rank models. The "all five regressors 66–79%" claim is kept only as a diagnostic with the constant-policy comparison beside it. |
| 3 | Direct classifiers | `03_lowco_rerun_and_direct_clf.py`, `10_*.py` | Label = the environment's ground-truth Rule-B configuration (6 possible classes, 4 occur). Inputs = observed V, R, L only. Scaler and `LabelEncoder` are fitted on the 7 training regimes only; one fit per held-out regime; hyperparameters copied; no tuning. When Very Dense Fog is held out, (850, 16) and (1250, 16) (7 of 2,697 environments) are absent from training and cannot be predicted. **Direct RF/XGB equal the constant policy** in the 5 single-label regimes and in Rain: Heavy, beat it by 21.5/13.3 points in Dense Fog and by 2.6/2.7 in Very Dense Fog. **Direct logistic regression is below it** in Visibility 1 and 2 (−34.2, −50.0), so its macro (70.8%) is below the constant's 79.9%. | The manuscript now states each of these. It no longer says that no model beats trivial baselines in Rain: Heavy. |
| 4 | Headline numbers | A programmatic check that every strict and tie-tolerant macro/micro value and 3-regime mean computed from `v7_rerun_decisions_all_models.csv` appears verbatim in the manuscript (0 missing), plus hand checks of the abstract, contributions, §VII-L/M tables, §VIII, §X, Fig. 1 | **Three errors in the candidate published on 2026-10-22, now fixed (a, b, d), and one caught in a draft before publication (c):** (a) "constant policy above every regression-based selector" was true only in macro; in micro the constant is 59.1% vs Linear Regression 61.0%, direct RF 61.1% and direct XGB 60.7%. (b) "In the contested regimes (Very Dense Fog, Rain: Heavy) no model beats trivial baselines" was false for Rain: Heavy, where Linear Regression is +8.6 points (paired bootstrap 95% CI [6.1, 11.1]; regime set fixed). (c) A draft's example BERs for the clipping floor compared the wrong configurations (corrected from env_000000 before publication). (d) "Every model attains close to 100% equivalent accuracy" in the single-label regimes was false (Linear Regression 66.7–79.7% in the Visibility regimes). The 83.1% infeasible-selection rate is the maximum cell, but it includes environments with no feasible configuration (37.3% of Very Dense Fog); this is now stated. | Fixed in the abstract, contributions 3–4, §VII-F, G, L, M, §VIII-B–D, §IX, §X, the figure legend and the self-audit. Macro and micro are labelled wherever a headline number appears. |
| 5 | Scope | File listing of `final_candidate/` | No training, no new data, no change to Phase 5 files. All new scripts (`10_*.py`, `11_*.py`, `12_*.py`) are read-only on inputs. Novelty remains scoped to "the sources we reviewed". No robustness claim. | None needed |

New read-only checks, all no-training: `10_submission_audit.py`, `11_audit_supplement.py` (logs in `outputs/10_*` and `11_*`).

---

## 3. Change log (this audit)

| # | File / section | Change |
|---|---|---|
| A1 | `IEEE_Paper_Final_Candidate.md`, note at top | Added item (iv) describing this audit |
| A2 | Abstract | Rewritten: macro/micro defined and stated; replay is a diagnostic, not a bound; reversed tie-break 100.0%; constant 79.9% macro but 59.1% micro; 3-regime comparison; tie-tolerant score no longer a headline |
| A3 | §I Contributions 3–4 | Rewritten to match (A2) |
| A4 | §V-F | Rewritten: "clipped-truth replay", "tie-tolerant agreement (not accuracy)", majority label verified in all 8 folds, direct-classifier labels/split/preprocessing and the 7 unpredictable Very Dense Fog labels |
| A5 | §VII-F | Infeasible-selection rate clarified (includes environments with no feasible configuration); table caveat reworded |
| A6 | §VII-G | "Censoring-equivalent" renamed "tie-tolerant agreement"; the false "close to 100%" clause replaced |
| A7 | §VII-I | "Explained by censoring" hedged to "consistent with the floor-tie effect" |
| A8 | §VII-K | Terminology updated (replay, tie-tolerant) |
| A9 | §VII-L | Rewritten: replay implementation check, mismatch decomposition, why it is not a bound, tie-tolerant definition and quantification, constants under both scores |
| A10 | §VII-M | Rewritten: macro vs micro stated, 3-regime table with means, per-regime direct-classifier comparison, paired bootstrap intervals, Fig. 1 caption |
| A11 | §VIII-B, C, D | Rewritten for tie-break interaction and the corrected contested-regime picture |
| A12 | §IX (Evaluation design) | Rewritten: replay not a bound; tie-tolerant score lenient; post hoc status; CIs condition on 8 regimes; infeasible-rate floor |
| A13 | §X Conclusion | Rewritten to match |
| A14 | Fig. 1 and "Final Self-Audit" | Legend relabelled "Clipped-truth replay (diagnostic, not a bound)"; self-audit table replaced |

The earlier change log (C1–C17), withdrawn claims (W1–W5) and corrections are unchanged.

---

## 4. Inventory and reproducibility (unchanged except as noted)

- **[REPRODUCED]:** strict LOWCO macro/micro decision accuracy for all five models (fresh rerun: LR, XGBoost, SVR, MLP 100% identical decisions; RF 99.98%, 2 of 10,433 differ); RF Very Dense Fog infeasible rate 83.1%; rank correlations (to 1e-16); ρ=0.116 (p=0.476); margin-stratified accuracy; replay, constants, tie-sets, per-regime values (this audit).
- **[SAVED]:** random-split R² (0.952) and Exp B/C regression results; dataset and physics model (not regenerated); μ=0 sensitivity strict results.
- **Missing:** original package versions; original per-environment predictions for XGBoost, SVR and MLP (replaced by the rerun); datasheet PDFs (visual check); repository URL.
- **Executed new computation:** E1 one pre-declared rerun (5 regressors + 3 untuned direct classifiers, ≈6 min CPU); all other checks no-training (`02`, `05`, `06`, `08`, `09`, `10`, `11`).

---

## 5. Remaining author actions

1. **Supervisor review** of the reframed contribution and title, and of the benchmark question above.
2. Fill the repository URL placeholder (§V-G) and upload the bundle.
3. Check datasheet values against refs [16]–[21] by hand (O5).
4. Run the novelty search in IEEE Xplore, Scopus or Web of Science (O10); resolve Springer s11082-024-06950-2 and the HAL optical-OFDM survey, or leave them uncited.
5. IEEE template, figure format, remove the top note and "Final Self-Audit", proofread.

## 6. Not verified

Datasheet PDFs (visual); full text of most §II literature (abstract/page level only); the closed-form physics against an independent simulator; the rain-coefficient diagnostics for XGBoost, SVR, MLP and the direct classifiers; the joint rain × 0.92 A/W case.
