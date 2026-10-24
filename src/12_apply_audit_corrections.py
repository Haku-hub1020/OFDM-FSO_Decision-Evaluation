"""12 -- Final submission audit corrections (2026-10-22). Post-processes the candidate written by 07_assemble_manuscript.py.
Text-only; no training; every replaced passage is asserted to occur exactly once. See Final_Candidate_Completion_Checklist.md (change log A1-A14)."""
P="/home/claude/final_candidate/manuscript/IEEE_Paper_Final_Candidate.md"
t=open(P,encoding="utf-8").read()
def rep(old,new):
    global t; assert t.count(old)==1,(t.count(old),old[:80]); t=t.replace(old,new)
def block(start,end,new,incl_end=False):
    global t; i=t.index(start); j=t.index(end,i+len(start)); assert t.count(start)==1,(t.count(start),start[:60])
    j2=j+len(end) if incl_end else j; t=t[:i]+new+t[j2:]

# --- note ---
rep("and (iii) adds §V-F/G","(iii) adds §V-F/G")
rep("with references 10–13, 22 and 23. All Phase 5","with references 10–13, 22 and 23, and (iv) incorporates a final audit (2026-10-22) that corrected the \"ceiling\" terminology (the replay of Rule B on clipped truth is a diagnostic, not an upper bound), demoted the censoring-equivalent score to a lenient diagnostic, and made every macro/micro statement explicit. All Phase 5")

rep("their reported performance should be read as a lower bound on their achievable accuracy with this feature set, not necessarily their ceiling","their reported performance should be read as a lower bound on the accuracy they could reach with this feature set, not as the best they could achieve")
# --- abstract ---
ABS="""## Abstract

Machine learning (ML) surrogates have been proposed to support adaptive wavelength and modulation selection in OFDM free-space optical (OFDM-FSO) links and are usually judged by regression accuracy on random train/test splits. We evaluate such surrogates at the decision level under atmospheric conditions absent from training. We build a physics-grounded analytical dataset of 10,433 environmental states (62,598 configuration rows; three wavelengths × two M-QAM orders) with a deterministic lexicographic decision rule (maximize bit rate subject to BER ≤ 10⁻⁹, then BER margin, then a fixed wavelength order) as ground truth, and benchmark five regressors under random-split, Leave-One-Weather-Condition-Out (LOWCO, eight regimes) and parameter-range-holdout protocols. Random Forest reaches R²≈0.952 on a random split, yet mean LOWCO R² is negative for all five models. Under strict decision matching, Linear Regression ranks first (76.4% macro, 61.0% micro) and Random Forest last (8.0%, 5.6%); macro is the mean over regimes and micro is pooled over environments. An audit shows that this ranking is largely an artifact of the benchmark. Regression targets are clipped at log₁₀ BER = −20, so in deep-feasible regimes candidate BERs tie exactly at the floor and the rule's fixed-order tie-break (850 nm first) disagrees with a ground truth computed from unclipped margins. Replaying the rule on the true clipped BERs matches the ground truth in only 24.0% of environments on a macro basis (49.8% micro), but in 100.0% (99.9%) if the tie-break order is reversed; this replay is a diagnostic, not an upper bound. A constant "1550 nm, 16-QAM" policy scores 79.9% macro, above every regression-based selector, but 59.1% micro, below Linear Regression (61.0%), because five of the eight regimes have that single label. Only Dense Fog, Rain: Heavy and Very Dense Fog involve a genuine contest. Over these three regimes mean strict accuracy is 55.0% for Linear Regression, 46.5% for the constant policy, 20–40% for the other four regressors and 50–55% for untuned direct classifiers; in Very Dense Fog no model is more than 2.7 points above the constant policy. We conclude that regression metrics alone are an unreliable proxy for decision quality under atmospheric shift, and that decision-level evaluation needs trivial baselines in both macro and micro terms, per-regime reporting and a check of how ties are broken. We do not demonstrate a reliable ML-based adaptive FSO controller or robust generalization to unseen atmospheric conditions.

"""
block("## Abstract","**Keywords:**",ABS)

# --- contributions ---
CONTR="""**Contributions:**
1. We provide a decision-level, environment-grouped LOWCO evaluation protocol (eight regimes, plus an independent parameter-range holdout) for ML surrogates of an OFDM-FSO configuration controller, with a deterministic ground-truth decision rule applied identically to true and predicted BER.
2. We show that random-split regression performance (R² up to 0.952) does not carry over to unseen atmospheric regimes: mean LOWCO R² is negative for all five models.
3. We show that strict decision-match accuracy on this benchmark is dominated by two effects unrelated to decision modelling: (a) exact ties at the −20 clipping floor interact with Rule B's fixed-order tie-break, so that replaying the rule on the true clipped BERs matches the ground truth in only 24.0% macro (49.8% micro) of environments, and in 100.0% (99.9%) with the tie-break order reversed (a post hoc illustration); this replay is a diagnostic and not an upper bound; (b) class dominance, since a constant 1550 nm / 16-QAM policy scores 79.9% macro, above every regression-based selector, but 59.1% micro, below Linear Regression (61.0%). We recommend reporting trivial baselines in both macro and micro terms, per-regime results and a tie-handling diagnostic alongside strict accuracy.
4. We identify the three regimes with a genuine decision contest (Dense Fog, Rain: Heavy, Very Dense Fog), compare all selectors there (mean strict accuracy: Linear Regression 55.0%, constant policy 46.5%, other regressors 20–40%, direct classifiers 50–55%), and show that the association between regression R² and strict decision accuracy is weak and unstable across scoring rules (ρ=0.12 strict; 0.54 under a lenient tie-tolerant diagnostic).
5. We provide a transparent, documented and sensitivity-tested physical parameterization (including a full-pipeline rain-coefficient rerun, §VII-K), a reproduction check of the saved results (§VII-N), and an explicit audit trail, as a reusable evaluation template.

"""
block("**Contributions:**","---\n\n## II. Related Work",CONTR)

# --- V-F ---
VF="""### F. Secondary Metrics, Baselines and Direct Classifiers (added post hoc)
The strict decision-match accuracy defined in §V-E is the pre-specified primary endpoint and is reported unchanged, both as a macro average (mean over the eight regimes) and as a micro average (pooled over environments). Auditing it (§VII-L) led us to add the following, all computed after the Phase 5 results were known.
*Clipped-truth replay (diagnostic):* the decisions obtained by applying Rule B unchanged, including its fixed wavelength-order tie-break, to the *true clipped target* log₁₀(BER_target) (§III-D) in place of a model's predictions, i.e. the decisions of a hypothetical predictor that reproduces the clipped target exactly. It is not an upper bound on achievable accuracy: it depends on the tie-break order, and other policies can score higher (§VII-L).
*Tie-tolerant agreement (lenient diagnostic, not accuracy):* a selection counts as agreeing if it belongs to the set of configurations that Rule B cannot distinguish on the true clipped target (same bit-rate tier, margins equal to within 10⁻⁹). By construction the ground-truth configuration is always in this set; the set can contain several configurations, so arbitrary choices earn credit (§VII-L). The −20 level that defines the ties is the ML-target clipping limit of §III-D and not a physical floor.
*Constant policies:* always selecting (1550 nm, 16-QAM), which is the most frequent ground-truth configuration in the training data of every one of the eight LOWCO folds (checked), and, as hindsight references only, other fixed configurations.
*Direct classifiers:* multinomial Logistic Regression (standardized inputs), Random Forest and XGBoost classifiers, with hyperparameters copied from their regression counterparts and no tuning. Inputs are the environment-level observed V, R and L; the label is the environment's ground-truth Rule-B configuration (six possible classes, four of which occur). Scaling and label encoding are fitted on the seven training regimes only, and one model is fitted per held-out regime. When Very Dense Fog is held out, two of its labels, (850 nm, 16-QAM) and (1250 nm, 16-QAM) (7 of 2,697 environments), do not occur in the training regimes and cannot be predicted. The classifiers are an exploratory baseline, not a proposed controller.

"""
block("### F. Secondary Metrics, Baselines and Direct Classifiers (added post hoc)","### G. Reproducibility",VF)

# --- VII-F tail ---
rep("and Random Forest's infeasible-selection rate reaches 83.1%, the worst value observed in this study.","and Random Forest's infeasible-selection rate reaches 83.1%, the worst value observed in this study (this rate counts every selected configuration that violates BER ≤ 10⁻⁹, including in the 37.3% of Very Dense Fog environments where no configuration is feasible, so it has a regime-dependent floor).")
rep("**This table must be read together with §VII-L and §VII-M:** the strict ranking is largely produced by the censoring of the regression target and is not beaten by a trivial constant policy.","**This table must be read together with §VII-L and §VII-M:** the strict ranking is largely produced by the interaction of target clipping with Rule B's tie-break and by class dominance, and in macro terms every row is below a trivial constant policy (79.9% macro; 59.1% micro).")
# --- VII-G fragments ---
rep("Under the *censoring-equivalent* accuracy of §VII-L the same correlation is ρ=0.542 (p=0.0003).","Under the lenient tie-tolerant agreement of §VII-L (a diagnostic, not accuracy) the same correlation is ρ=0.542 (p=0.0003).")
rep("for strict and ρ=0.539 (p=0.038) for censoring-equivalent accuracy.","for strict and ρ=0.539 (p=0.038) for tie-tolerant agreement.")
rep("while every model attains close to 100% equivalent accuracy.","and in which the ground truth is a single configuration in every environment.")
rep("while censoring-equivalent decision accuracy is 66–79% macro.","while the lenient tie-tolerant agreement is 66–79% macro.")
# --- VII-I ---
block("Random Forest's uniformly low accuracy is largely explained by censoring:","faithfully.","Random Forest's uniformly low accuracy is consistent with the floor-tie effect of §VII-L: among contested environments, in 99.8% of the large-margin stratum and 41.6% of the medium-margin stratum the two best configurations of the winning tier both have true BER below the −20 floor, so the clipped target carries no ordering information for them. The margin stratification therefore does not establish that Random Forest's failure is \"representational\"; it is consistent with strict matching penalizing predictions that tie or are noisy at the clipped floor.",incl_end=True)
# --- VII-K ---
rep("oracle ceiling 24.5% macro (50.5% micro)","clipped-truth replay 24.5% macro (50.5% micro)")
rep("and censoring-equivalent accuracy of 75.2% for Random Forest versus 76.3% for Linear Regression.","and tie-tolerant agreement of 75.2% for Random Forest versus 76.3% for Linear Regression.")

# --- VII-L, VII-M ---
LM="""### L. Clipped-Truth Replay and Tie-Tolerant Agreement (Diagnostics)
The regression target is clipped at log₁₀ BER = −20 (§III-D), while the ground-truth label was computed from unclipped BER margins. If all candidate configurations of the winning bit-rate tier have true BER below 10⁻²⁰, a predictor that reproduced the clipped target exactly would output −20 for all of them; their margins tie and Rule B's fixed wavelength order selects the first tied configuration (850 nm), whereas the ground truth selects 1550 nm from raw BER differences that have no operational meaning (for example, on one Clear-sky state the 16-QAM raw BERs are 10⁻¹⁶²² at 850 nm, 10⁻¹³⁹²⁶⁸ at 1250 nm and 10⁻¹⁵³⁸⁴² at 1550 nm, all clipped to 10⁻²⁰). We call the decision obtained by applying Rule B unchanged to the true clipped BERs the *clipped-truth replay*. We checked the implementation: applied to the unclipped BERs it reproduces the stored ground-truth label in 10,433 of 10,433 environments, and applied to the clipped BERs it matches it as follows.

| Regime | Clipped-truth replay (strict) | Linear Regression | Random Forest |
|---|---|---|---|
| Clear | 0.0% | 100.0% | 0.1% |
| Visibility Regime 1 | 0.0% | 79.7% | 0.0% |
| Visibility Regime 2 | 0.0% | 66.7% | 0.0% |
| Rain: Light | 0.0% | 100.0% | 0.0% |
| Rain: Moderate | 0.0% | 100.0% | 0.7% |
| Dense Fog | 52.7% | 70.7% | 54.3% |
| Rain: Heavy | 66.0% | 55.2% | 0.0% |
| Very Dense Fog | 73.6% | 39.1% | 8.7% |
| **Macro / micro** | **24.0% / 49.8%** | 76.4% / 61.0% | 8.0% / 5.6% |

All 5,239 replay mismatches are environments in which two or more configurations are exactly tied at the floor; none occurs in infeasible, single-wavelength-feasible or bit-rate-determined environments (§VI-A cases).

**The replay is a diagnostic, not an upper bound on achievable accuracy.** It is the accuracy of one fixed policy, Rule B with an 850-nm-first tie-break applied to clipped inputs, and its value depends on that arbitrary tie-break: with the order reversed (1550 nm first; an illustration chosen with knowledge of the labels, not a proposed rule) the same replay matches the ground truth in 100.0% of environments on a macro basis (99.9% micro). Other policies exceed it: the constant (1550 nm, 16-QAM) policy scores 79.9% macro, the direct Random Forest classifier 82.9%, and Linear Regression exceeds it in six of the eight regimes (the five where the replay scores 0%, and Dense Fog). The replay therefore shows how strict matching treats exact ties at the floor, not how well any model could do. In the five regimes where it scores 0%, strict matching measures whether a model's predictions break floor ties toward the wavelength preferred by the unclipped ground truth. Linear Regression's predictions are never constant across the six configurations (§VII-H), which is consistent with, but was not tested as, linear extrapolation below −20 preserving an ordering; a Random Forest cannot output values outside the range of its training targets, so near the floor its predictions tie or differ only by noise. (The replay column uses all 10,433 environments; the Random Forest and Linear Regression columns use the saved Phase 3 predictions, which reproduce the saved decisions exactly.)

*Tie-tolerant agreement.* A selection is scored as agreeing if it belongs to the set of configurations that Rule B cannot distinguish on the true clipped target (§V-F). **This is not accuracy.** By construction the ground-truth configuration is always in the set (verified for all 10,433 environments); the set contains two or more configurations in 5,239 environments (three in 2,973), namely every Clear, Visibility, Rain: Light and Rain: Moderate environment and 47.3%, 34.0% and 26.4% of Dense Fog, Rain: Heavy and Very Dense Fog environments; and the −20 level that defines the ties is the project's ML-target clipping limit, not a physical floor. It therefore credits any 16-QAM choice in deep-feasible regimes: a constant (1250 nm, 16-QAM) policy, which is never the ground-truth label, scores 70.5% macro (40.4% micro) and a constant (850 nm, 16-QAM) policy scores 53.0% (28.6%). We report it only as a lenient diagnostic of how much of the strict deficit is attributable to floor ties, not for ranking models.

| Selector | Strict macro | Tie-tolerant macro | Strict micro | Tie-tolerant micro |
|---|---|---|---|---|
| Linear Regression | 76.4% | 76.4% | 61.0% | 61.0% |
| Random Forest | 8.0% | 75.5% | 5.6% | 43.4% |
| XGBoost | 39.0% | 79.3% | 38.0% | 57.2% |
| SVR | 39.5% | 66.1% | 20.8% | 39.4% |
| MLP | 33.1% | 71.3% | 18.9% | 37.6% |
| Constant (1550 nm, 16-QAM) | 79.9% | 79.9% | 59.1% | 59.1% |
| Constant (1250 nm, 16-QAM), reference | 0.0% | 70.5% | 0.0% | 40.4% |
| Constant (850 nm, 16-QAM), reference | 0.0% | 53.0% | 0.1% | 28.6% |

Under this diagnostic the five regressors lie about 13 points apart (66.1–79.3% macro) and Random Forest is not the lowest, but none exceeds the constant (1550 nm, 16-QAM) policy on macro (XGBoost: 79.3% vs. 79.9%), and in Rain: Heavy and Very Dense Fog Random Forest remains far below it (§VII-M). Within-regime bootstrap resampling of environments gives 95% intervals of about ±0.5–1.5 points on the macro values, but this ignores regime-level variability: with only eight regimes, differences of a few points should not be interpreted. The pre-specified strict endpoint remains the primary result.

### M. Trivial Baselines and Direct Classification
Five of the eight regimes (Clear, Visibility Regimes 1 and 2, Rain: Light, Rain: Moderate; 2,744 of 10,433 environments) have (1550 nm, 16-QAM) as the ground-truth decision in every environment. In the macro average these five regimes carry 5/8 of the weight, and a constant policy that always selects (1550 nm, 16-QAM), the most frequent training label in every LOWCO fold, scores 79.9% macro, above every regression-based selector in §VII-F (best: Linear Regression, 76.4%). **In micro terms the order is reversed:** the constant policy scores 59.1% micro, below Linear Regression (61.0%) and below the direct Random Forest (61.1%) and XGBoost (60.7%) classifiers. Macro and micro answer different questions, and neither supports a general statement that selection models are better or worse than a constant policy. The informative comparison is in the three regimes with genuinely contested decisions (Dense Fog, Rain: Heavy, Very Dense Fog; 7,689 environments), where strict accuracy is as follows.

| Selector (strict accuracy) | Dense Fog | Rain: Heavy | Very Dense Fog | Mean of the 3 regimes |
|---|---|---|---|---|
| Clipped-truth replay (diagnostic, not a bound) | 52.7% | 66.0% | 73.6% | 64.1% |
| Constant (1550 nm, 16-QAM) | 53.7% | 46.6% | 39.1% | 46.5% |
| Linear Regression | 70.7% | 55.2% | 39.1% | 55.0% |
| Random Forest | 54.3% | 0.0% | 8.7% | 21.0% |
| XGBoost | 39.5% | 40.7% | 40.2% | 40.1% |
| SVR | 54.9% | 4.3% | 20.0% | 26.4% |
| MLP | 42.0% | 10.1% | 8.6% | 20.2% |
| Direct Logistic Regression | 65.4% | 46.4% | 39.1% | 50.3% |
| Direct Random Forest | 75.2% | 46.6% | 41.7% | 54.5% |
| Direct XGBoost | 66.9% | 46.6% | 41.8% | 51.8% |

Direct classifiers (§V-F) score 70.8% (logistic regression), 82.9% (Random Forest) and 81.9% (XGBoost) macro, and 55.4%, 61.1% and 60.7% micro. The Random Forest and XGBoost classifiers equal the constant policy in the five single-label regimes and in Rain: Heavy, exceed it in Dense Fog (by 21.5 and 13.3 points) and exceed it only slightly in Very Dense Fog (by 2.6 and 2.7 points; paired environment-level bootstrap 95% intervals [1.3, 3.7] and [1.5, 3.8], regime set fixed). The direct logistic regression is below the constant policy in both Visibility regimes (by 34.2 and 50.0 points), which is why its macro accuracy is below the constant policy's. Linear Regression exceeds the constant policy in Dense Fog (by 17.0 points, [10.1, 23.9]) and Rain: Heavy (by 8.6 points, [6.1, 11.1]) and has the same per-environment correctness as the constant policy in Very Dense Fog. XGBoost is below it in Rain: Heavy (by 5.9 points, [5.2, 6.6]), and Random Forest, SVR and MLP are far below it in Rain: Heavy and Very Dense Fog. A hindsight constant of (1550 nm, 4-QAM) would score 53.4% in Rain: Heavy and 60.6% in Very Dense Fog but is not available to a controller trained on the other regimes. In summary, several models clearly exceed the constant policy only in Dense Fog; in Rain: Heavy only Linear Regression does; and in Very Dense Fog no model is more than 2.7 points above it. These intervals condition on the eight regimes and treat environments within a regime as independent, and the classifiers are untuned, so the comparison is descriptive and does not show that direct classification is preferable or inadequate.

![Fig. 1](fig1_regime_accuracy.png)

**Fig. 1.** Strict decision accuracy per held-out regime (LOWCO, baseline parameterization) for the clipped-truth replay (a diagnostic, not an upper bound), the constant (1550 nm, 16-QAM) policy, and three regression-based selectors. In the five single-label regimes (left) the replay scores 0% and the constant policy 100%; the three contested regimes (right) are where selectors can differ from the constant policy. Generated from the saved decision tables by `analysis/09_make_figure.py`; Random Forest values in the five single-label regimes are 0.0–0.7%.

"""
block("### L. Censoring Ceiling and Censoring-Equivalent Scoring","### N. Reproduction of the Saved Results",LM)

# --- discussion ---
DISC="""## VIII. Discussion

### A. Why Regression Accuracy Need Not Track Decision Accuracy
Rule B's decision depends on three logically distinct sub-tasks: the relative ordering of the six candidates, the feasibility classification against a fixed threshold, and a final lexicographic optimization. A regressor trained to minimize pointwise squared error has no explicit incentive to preserve ordering or threshold placement, an observation in line with the general decision-focused-learning view that prediction error and decision error differ [22], and with work that trains surrogates to preserve the *ranking* of configurations rather than regress their performance [23]. Those works are from other domains and we do not claim that their methods would help here.

### B. Why the Strict Ranking of Models Is Misleading
The strict Random Forest/Linear Regression reversal is produced mainly by the interaction of target clipping with Rule B's fixed-order tie-break (§VII-L): replaying the rule on the true clipped BERs matches the ground truth in 24.0% of environments (macro) with the 850-nm-first tie-break and in 100.0% with the order reversed. Strict matching in the five single-label regimes therefore rewards predictions whose floor ties break toward the ground-truth wavelength (Linear Regression, plausibly through extrapolation below the floor, which we did not test) and penalizes predictions that tie or are noisy at the floor (Random Forest). The first-pass reading of the reversal as a "ranking-preservation collapse" in Random Forest is therefore withdrawn for these five regimes. In the contested regimes, Random Forest, SVR and MLP are far below the constant policy in Rain: Heavy and Very Dense Fog (§VII-M), and Random Forest's rank correlation in Very Dense Fog is 0.138 (§VII-H); the mechanism behind these deficits is not established here.

### C. Genuine Difficulty Is Concentrated in Fog and Heavy Rain
Contested decisions occur in Dense Fog, Rain: Heavy and Very Dense Fog, and small-margin contests are concentrated in the latter two. Even there, gains over a constant policy are confined to Dense Fog (several models) and Rain: Heavy (Linear Regression only), and in Very Dense Fog no model is more than 2.7 points above the constant policy (§VII-M). With only three contested regimes this is a descriptive observation and not a statistical conclusion.

### D. Implications for ML Evaluation in Adaptive FSO
Three practical rules follow. (i) Random-split regression metrics are an unsafe basis for model selection when the deployment task is configuration selection under unseen conditions. (ii) Decision-level evaluation should be accompanied by trivial constant and majority-class baselines reported in both macro and micro terms, by per-regime results with a count of the regimes that carry a genuine decision contest, and by a check of how the decision layer breaks ties created by target clipping (for example a clipped-truth replay); otherwise class imbalance and tie-break artifacts can dominate the comparison in either direction. (iii) Macro averages over regimes should be accompanied by micro averages and per-regime results.

---

"""
block("## VIII. Discussion","## IX. Limitations and Threats to Validity",DISC)

LIM2="""**Evaluation design (final candidate):** the primary strict decision metric is confounded by exact ties at the clipping floor interacting with a fixed-order tie-break, and by class dominance (§VII-L–M); the clipped-truth replay is a diagnostic and not an upper bound, and its reversed-tie-break variant was chosen with knowledge of the labels; the tie-tolerant agreement score is lenient and credits arbitrary 16-QAM choices, so it is not used to rank models; the constant-policy baselines, the direct classifiers (untuned, one run per fold, inputs limited to the observed V, R and L) and these diagnostics were added after the Phase 5 results were known and are therefore post hoc; the ground-truth Rule B itself discriminates among configurations on BER margins below 10⁻²⁰ that are analytic extrapolations of a Gaussian-tail model and have no operational meaning (a floor-aware ground truth would change the benchmark and was not attempted); only eight regimes are available, three of which carry decision content, so regime-level conclusions are low-powered, environment-level bootstrap intervals condition on the regime set, and the Spearman p-values in §VII-G are descriptive; the infeasible-selection rate includes environments in which no configuration is feasible (23.7% of all environments, 37.3% of Very Dense Fog), so it has a regime-dependent floor; within-environment rank correlation is undefined in 28.5% of environments (§VII-H); the diagnostics were repeated for the alternative rain coefficients only for Random Forest and Linear Regression and not for the 0.92 A/W responsivity case; the Random Forest reproduction differs in 2 of 10,433 decisions, and original package versions were not logged.

"""
block("**Evaluation design (final candidate):**","\n---\n\n## X. Conclusion",LIM2.rstrip("\n"))
rep("The censoring ceiling, constant-policy and equivalence checks were repeated","The clipped-truth replay, constant-policy and tie-tolerant checks were repeated")

CONC="""## X. Conclusion

We evaluated five ML models as BER surrogates for a deterministic adaptive wavelength/modulation controller for OFDM-FSO under random-split, Leave-One-Weather-Condition-Out and parameter-range-holdout protocols. Random-split performance can look strong (R² up to 0.952) while LOWCO regression R² is negative on average for every model, so random-split regression evaluation is not a safe basis for choosing a surrogate for unseen atmospheric conditions. Strict decision-match accuracy initially appeared to reverse the regression ranking (Random Forest 8.0% vs. Linear Regression 76.4% macro), but an audit showed that this ranking is largely an artifact of the benchmark: clipping the regression target at log₁₀ BER = −20 creates exact ties, and Rule B's fixed-order tie-break resolves them differently from the unclipped ground truth (replay on clipped truth: 24.0% macro, 100.0% with the tie-break order reversed). A constant 1550 nm / 16-QAM policy scores 79.9% macro, above every regression-based selector, but 59.1% micro, below Linear Regression (61.0%), because five of the eight regimes have a single label. In the three regimes with genuinely contested decisions, mean strict accuracy is 55.0% for Linear Regression, 46.5% for the constant policy and 20–40% for the other regressors; gains over the constant policy are confined to Dense Fog (several models) and Rain: Heavy (Linear Regression only), and in Very Dense Fog no model is more than 2.7 points above it. We do not demonstrate a reliable ML-based adaptive FSO controller, nor robust generalization to unseen atmospheric conditions, and we withdraw the first-pass claims that regression R² and decision accuracy are uncorrelated and that Random Forest suffers a ranking-preservation collapse across regimes. The contribution is an evaluation lesson: decision-level LOWCO evaluation of FSO surrogates should report trivial baselines in macro and micro terms, per-regime results and a tie-handling diagnostic alongside regression metrics. Extending the benchmark with a floor-aware ground truth, more regimes and measured data are the natural next steps; mitigation by ranking-aware training or calibration is premature until the evaluation issues above are resolved.

---

"""
block("## X. Conclusion","## References",CONC)

AUD="""## Final Self-Audit

*(Internal working section — remove before submission.)*

| Potential Reviewer Concern | Where Addressed | Evidence | Remaining Risk |
|---|---|---|---|
| Title accuracy | Title; §X | Describes an evaluation study and its pitfalls; no claim of robustness | Low. Author must confirm; the sprint brief used a different working title |
| Novelty claim | §I, §II | Scoped to "the sources we reviewed"; search limits stated; no claim of priority | Medium. General web search only; two leads unverified and not cited |
| Primary-endpoint integrity | §V-E, §V-F, §VII-F | Strict macro and micro accuracy reported unchanged; new metrics labelled post hoc | Low |
| Replay terminology | §V-F, §VII-L | Replay is a diagnostic, not an upper bound (reversed tie-break 100.0%; constant policy 79.9%; LR exceeds it in 6 regimes) | Low after the audit corrections |
| Tie-tolerant metric | §V-F, §VII-L | Not accuracy; true label always in set; arbitrary constants earn 53–70% macro; not used to rank | Low (demoted) |
| Trivial-baseline claims | §VII-M, abstract | Macro 79.9% vs micro 59.1%; contested-regime table and paired bootstrap intervals | Low–Medium: intervals condition on 8 regimes |
| Direct classifiers | §V-F, §VII-M | Labels, split and preprocessing audited; logistic regression below constant in Visibility regimes | Low |
| Withdrawn claims | §VII-G, §VII-I, §VIII-B, §X | ρ≈0 claim, "RF ranking collapse across regimes", "others improve monotonically with margin" withdrawn and logged | Low |
| Statistical inference | §VII-G, §VII-L, §VII-M | Spearman p-values descriptive; only 8 regimes (3 contested) | Medium |
| Data/configuration leakage | §IV-E, §V-F | Environment-level splitting; scaling and label encoding fitted on training regimes only | Low |
| Reproducibility | §V-G, §VII-N | Saved results re-derived; RF 99.98%, others 100%; versions recorded; originals not logged | Low–Medium. Repository URL placeholder must be filled |
| Physical-model limitations / hardware validation | §IX | Unchanged from Final Handoff | Medium (unavoidable here) |
| Sources not visually verified | §III, §IV, refs [16]–[21] | Manufacturer PDFs could not be downloaded in this environment | Medium. Author to check datasheet values manually (O5) |
| Rain-coefficient / responsivity sensitivity | §VII-J, §VII-K | μ=0 full rerun (strict metrics); replay/constant/tie-tolerant checked for RF and LR only (no training); joint test with 0.92 A/W not done | Medium |
"""
i=t.index("## Final Self-Audit"); t=t[:i]+AUD
open(P,"w",encoding="utf-8").write(t); print("ok",len(t))
