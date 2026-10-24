import re, sys
SRC="/home/claude/IEEE_Paper_Final_Handoff.md"; DST="/home/claude/final_candidate/manuscript/IEEE_Paper_Final_Candidate.md"
t=open(SRC,encoding="utf-8").read()

# ---- 1. renumber old refs >=10 by +4 (body cites and reference-list labels) ----
def bump(m):
    n=int(m.group(1)); return f"[{n+4 if n>=10 else n}{m.group(2) or ''}]"
t=re.sub(r"\[(\d+)((?:, (?:Sec\.|Table)[^\]]*)?)\]",bump,t)
# (reference-list labels "[n] " are already renumbered by the line-8 substitution; a second pass would double-bump them)

def block(t,start,end,new,keep_end=True):
    i=t.index(start); j=t.index(end,i+len(start))
    assert i<j,(start,end)
    return t[:i]+new+t[j:]
def rep(t,old,new):
    assert t.count(old)==1,(t.count(old),old[:70]); return t.replace(old,new)

# ---- 2. title block ----
TITLE="""# Decision-Level Evaluation of ML Surrogates for OFDM-FSO Configuration Selection Under Unseen Atmospheric Conditions: Censoring Artifacts and Trivial-Baseline Pitfalls

> **Final-candidate note (2026-10-21).** This file is a new working copy. `IEEE_Paper_Final_Handoff.md`, `IEEE_Paper_Revised.md` and `IEEE_Paper_Draft.md` are preserved unchanged. Compared with the Final Handoff version, this candidate (i) re-titles the paper, (ii) withdraws or rewrites the claims in the Abstract, Contributions, §VII-F–I, §VIII and §X that an audit of the saved predictions showed to be confounded (see the change log in `Final_Candidate_Completion_Checklist.md`), and (iii) adds §V-F/G, §VII-L–N and the related-work paragraph on adaptive-ML work in FSO, with references 10–13, 22 and 23. All Phase 5 datasets, folds, hyperparameters and result tables are unchanged. The new analyses are post hoc and are labelled as such. The working title used in the author's sprint brief (\"Robust ML-Based Wavelength Selection…\") is not used: the results do not demonstrate robustness, and the paper evaluates configuration (wavelength and modulation) selection."""
i=t.index("# When Better"); j=t.index("**Ramesh Yadav**"); t=t[:i]+TITLE+"\n\n"+t[j:]

# ---- 3. abstract ----
ABS="""## Abstract

Machine learning (ML) surrogates have been proposed to support adaptive wavelength and modulation selection in OFDM free-space optical (OFDM-FSO) links and are usually judged by regression accuracy on random train/test splits. We evaluate such surrogates at the decision level under atmospheric conditions absent from training. We build a physics-grounded analytical dataset of 10,433 environmental states (62,598 configuration rows; three wavelengths × two M-QAM orders) with a deterministic lexicographic decision rule (maximize bit rate subject to BER ≤ 10⁻⁹, then maximize BER margin) as ground truth, and benchmark five regressors under random-split, Leave-One-Weather-Condition-Out (LOWCO, eight regimes) and parameter-range-holdout protocols. Random Forest reaches R²≈0.952 on a random split, yet mean LOWCO R² is negative for all five models. Under strict decision matching, LOWCO accuracy ranks Linear Regression first (76.4% macro) and Random Forest last (8.0%). An audit shows this ranking is largely an evaluation artifact. Because regression targets are censored at log₁₀ BER = −20, a perfect predictor of the censored target reaches only 24.0% macro (49.8% micro) strict accuracy: in deep-feasible regimes the rule's margin tie-break receives tied inputs. In addition, a constant \"always 1550 nm, 16-QAM\" policy scores 79.9% macro, above every regression-based selector. Scoring censoring-equivalent decisions removes the Random Forest deficit (66–79% macro for all five regressors). In the regimes with genuinely contested decisions, no model, including direct classifiers, clearly outperforms trivial baselines in Very Dense Fog (best model 41.8% vs. 39.1% for the training-majority constant), and the best regression-based selector exceeds that constant by 8.6 points in Rain: Heavy; Random Forest, SVR and MLP are weakest there. The association between regression R² and decision accuracy depends on the scoring rule (Spearman ρ = 0.12 strict vs. 0.54 censoring-equivalent, over 40 non-independent model×regime observations). We conclude that regression metrics alone are an unreliable proxy for decision quality under atmospheric shift, and that decision-level evaluation itself requires oracle ceilings, equivalence-aware scoring and trivial baselines to avoid misleading conclusions. We do not demonstrate a reliable ML-based adaptive FSO controller or robust generalization to unseen atmospheric conditions.

"""
t=block(t,"## Abstract","**Keywords:**",ABS)
t=rep(t,"**Keywords:** free-space optics, OFDM, machine learning, adaptive modulation, wavelength selection, distribution shift, model evaluation, bit error rate prediction","**Keywords:** free-space optics, OFDM, machine learning, adaptive modulation, wavelength selection, distribution shift, model evaluation, target censoring, bit error rate prediction")

# ---- 4. intro: research question tail + contributions ----
t=rep(t,"and a Leave-One-Weather-Condition-Out (LOWCO) evaluation protocol.\n","and a Leave-One-Weather-Condition-Out (LOWCO) evaluation protocol. Auditing our first-pass answer exposed two evaluation-design pitfalls (target censoring and trivial-baseline dominance); we report these as findings in their own right.\n")
CONTR="""**Contributions:**
1. We provide a decision-level, environment-grouped LOWCO evaluation protocol (eight regimes, plus an independent parameter-range holdout) for ML surrogates of an OFDM-FSO configuration controller, with a deterministic ground-truth decision rule applied identically to true and predicted BER.
2. We show that random-split regression performance (R² up to 0.952) does not carry over to unseen atmospheric regimes: mean LOWCO R² is negative for all five models.
3. We show that strict decision-match accuracy on this benchmark is confounded by two pitfalls: (a) censoring of the regression target at log₁₀ BER = −20 caps the strict accuracy of even a perfect predictor at 24.0% macro (49.8% micro) and reverses the apparent model ranking; (b) a constant 1550 nm / 16-QAM policy attains 79.9% macro accuracy, above every regression-based selector. We propose censoring-equivalent scoring and mandatory trivial-baseline and oracle-ceiling reporting as evaluation hygiene.
4. Under corrected scoring, we characterise where decisions are genuinely hard (Dense Fog, Very Dense Fog, Rain: Heavy), report that no evaluated model, including direct classifiers, clearly beats trivial baselines in the latter two, and show that the relationship between regression R² and decision accuracy depends on the scoring rule.
5. We provide a transparent, documented and sensitivity-tested physical parameterization (including a full-pipeline rain-coefficient rerun, §VII-K), a reproduction check of the saved results (§VII-N), and an explicit audit trail, as a reusable evaluation template.

"""
t=block(t,"**Contributions:**","---\n\n## II. Related Work",CONTR)

# ---- 5. related work ----
GAP_OLD_START="**Gap addressed.**"
NEWRW="""**Adaptive ML in FSO beyond wavelength prediction.** Other ML work for FSO and optical wireless links predicts link performance or selects a transmission mode. Younes *et al.* [10] predict BER for simulated OFDM-FSO links under fog, rain and clear conditions with SVR, a decision tree and Random Forest, and report R² and MSE (the page we reviewed does not state the split). Shao *et al.* [11] use a Random Forest to classify the modulation format of a hybrid FSO/RF link from rainfall and visibility, evaluated by classification accuracy on a separate weather data set. Derouiche *et al.* [12] regress the turbulence strength Cn² and use a deep neural network to select the modulation of a mixed FSO/MIMO-mmWave link. Esmail [13] uses a Random Forest to classify modulation format and baud rate and to estimate Cn² and OSNR in simulation, with a random split on a fixed parameter grid and the wavelength fixed at 1550 nm. As described in the pages and abstracts we reviewed, none of these works evaluates held-out weather regimes or compares decision quality with regression quality. Our review of this literature used a general web search and page-level reading, not IEEE Xplore, Scopus or Web of Science, so the absence of such work from our list is not evidence that none exists.

"""
i=t.index(GAP_OLD_START); j=t.index("---",i)
GAP="""**Gap addressed.** In the sources we reviewed, we did not find an OFDM-FSO ML study that evaluates configuration-selection surrogates at the decision level under atmospheric regimes structurally absent from training, or that compares regression accuracy with decision accuracy across candidate models. We make no claim of priority beyond that scope. This paper addresses the gap and, in doing so, documents evaluation pitfalls that such a study must control for.

"""
t=t[:i]+NEWRW+GAP+t[j:]

# ---- 6. methods additions ----
VEADD="""
### F. Secondary Metrics, Baselines and Direct Classifiers (added post hoc)
The strict decision-match accuracy defined in §V-E is the pre-specified primary endpoint and is reported unchanged. Auditing it (§VII-L) led us to add the following, all computed after the Phase 5 results were known.
*Oracle ceiling:* the strict accuracy obtained by applying Rule B to the *true censored target* log₁₀(BER_target) (§III-D), i.e. the best any model trained on that target could achieve through the same decision layer.
*Censoring-equivalent accuracy:* a decision counts as correct if the selected configuration belongs to the set of configurations that Rule B cannot distinguish when applied to the true censored target (same bit-rate tier, equal margin). This follows the paper's own statement in §III-D that BER below roughly 10⁻²⁰ is operationally equivalent. It is more lenient than strict matching by construction (the true configuration always lies in that set).
*Constant policies:* always selecting (1550 nm, 16-QAM), which is the most frequent ground-truth configuration in every training fold, and, as a hindsight reference only, (1550 nm, 4-QAM).
*Direct classifiers:* multinomial Logistic Regression (standardized inputs), Random Forest and XGBoost classifiers, with hyperparameters copied from their regression counterparts and no tuning. Inputs are the environment-level observed V, R and L; the label is the ground-truth Rule-B configuration (six classes). They are an exploratory baseline, not a proposed controller.

### G. Reproducibility
All results use seed 20260906, the hyperparameters of §V-B and the environment-level splits described above. The saved Phase 2B.1 decision tables were checked against a fresh rerun in the environment listed in §VII-N. Package versions were not logged when the original experiments were run. Code and data: [REPOSITORY URL TO BE INSERTED BY THE AUTHOR BEFORE SUBMISSION].

"""
t=rep(t,"\n---\n\n## VI. Adaptive Configuration Decision Framework", "\n"+VEADD.rstrip("\n")+"\n\n---\n\n## VI. Adaptive Configuration Decision Framework")
t=rep(t,"independent of its absolute calibration.","independent of its absolute calibration. The correlation is computed against the censored target (§III-D) and is undefined for an environment in which the six true values are identical (all at the −20 floor) or the six predictions are identical; §VII-H reports how often this occurs.")

# ---- 7. results ----
t=rep(t,"(a forced choice, not a margin comparison)","(a forced choice, not a margin comparison)")  # sanity anchor
t=rep(t,"This ranking was found to be robust across the full tested eye-safety-multiplier sensitivity range","In many of the margin comparisons the compared BER values lie far below 10⁻²⁰ (§VII-L), where the analytic Gaussian-tail values have no operational meaning. This ranking was found to be robust across the full tested eye-safety-multiplier sensitivity range")

RES="""### F. LOWCO Configuration-Selection Results (strict matching)

| Model | Macro decision acc. | Micro decision acc. |
|---|---|---|
| Linear Regression | 76.4% | 61.0% |
| SVR | 39.5% | 20.8% |
| XGBoost | 39.0% | 38.0% |
| MLP | 33.1% | 18.9% |
| Random Forest | 8.0% | 5.6% |

Macro = mean over the eight regimes; micro = pooled over environments. Complete per-regime results for all five models are in the supplementary decision table. On Clear, Rain: Light, Rain: Moderate and both Visibility regimes, Random Forest's strict accuracy is at or below 0.7%, while Linear Regression achieves 100% on three of them. On Very Dense Fog every model's strict accuracy is below 41% and Random Forest's infeasible-selection rate reaches 83.1%, the worst value observed in this study. §VII-K reports that the Random Forest/Linear Regression ordering is preserved (76.3% vs. 9.5% macro) under the alternative rain coefficients. **This table must be read together with §VII-L and §VII-M:** the strict ranking is largely produced by the censoring of the regression target and is not beaten by a trivial constant policy.

### G. Regression-versus-Decision Association
Across the 40 model×regime combinations, the Spearman correlation between LOWCO regression R² and *strict* decision accuracy is ρ=0.116 (p=0.476); under the alternative rain coefficients it is ρ=0.245 (p=0.128). Under the *censoring-equivalent* accuracy of §VII-L the same correlation is ρ=0.542 (p=0.0003). Restricted to the three regimes with non-degenerate decisions (Dense Fog, Very Dense Fog, Rain: Heavy; n=15) it is ρ=0.257 (p=0.355) for strict and ρ=0.539 (p=0.038) for censoring-equivalent accuracy. These p-values treat observations as independent, which they are not (shared data and folds across models), so they are descriptive only. In addition, 25 of the 40 observations come from five regimes in which R² is zero or unstable (e.g. R²=0.000 for several models on Clear and Visibility Regime 1, and −27.5 for Linear Regression on Visibility Regime 2) while every model attains close to 100% equivalent accuracy. **We therefore do not claim that regression R² and decision accuracy are statistically uncorrelated;** the sign and strength of the association depend on the scoring rule and on which regimes are included. What the data do support is narrower: Random Forest's random-split R² (0.952) did not translate into the best LOWCO decisions under strict matching, and LOWCO R² is negative for every model while censoring-equivalent decision accuracy is 66–79% macro.

### H. Configuration-Ranking Preservation
Mean within-environment rank correlation (predicted vs. true ordering of the six configurations) is 0.848 for Linear Regression and 0.570 for Random Forest overall; in Very Dense Fog it is 0.138 for Random Forest and 0.852 for Linear Regression. These means are computed only over environments in which the correlation is defined. Because the correlation uses the censored target, it is undefined when all six true values sit at the −20 floor: this affects 28.5% of environments overall (100% of Clear and of Visibility Regime 1, 76.7% of Rain: Light, 76.3% of Visibility Regime 2, 45.7% of Rain: Moderate, and only 12.6%, 9.1% and 3.7% of Dense Fog, Rain: Heavy and Very Dense Fog respectively). Random Forest predictions are additionally constant in 6.2% of environments (22.1% of Clear). The overall means are therefore dominated by Rain: Heavy and Very Dense Fog and are not population averages. The Very Dense Fog contrast rests on about 96% of that regime's environments and remains valid descriptive evidence for that regime; §VII-K reproduces it almost exactly (0.138 and 0.852) under the alternative rain coefficients. We treat the association between rank-correlation collapse and poor decisions as descriptive: the study does not establish rank-correlation collapse as the cause.

### I. Regime-Specific Behavior and Decision Margins
Decision-margin analysis (ground truth only, §VI-C) finds 73.0% of environments contain a genuine within-tier contest, 23.7% are infeasible and 3.3% have only one wavelength feasible. Among contested environments only 6.8% have a small (<1 decade) tier-contest margin, concentrated in Very Dense Fog and Rain: Heavy. Strict decision accuracy by tier-contest margin (contested environments; small / medium / large) is:

| Model | Small (<1) | Medium (1–5) | Large (>5) |
|---|---|---|---|
| Linear Regression | 52.7% | 51.0% | 80.4% |
| Random Forest | 4.8% | 6.7% | 2.5% |
| XGBoost | 50.8% | 33.8% | 48.4% |
| SVR | 13.8% | 15.0% | 33.7% |
| MLP | 9.9% | 14.8% | 29.6% |

Only SVR and MLP increase monotonically with margin at the baseline parameterization (Linear Regression does so under the alternative rain coefficients of §VII-K; XGBoost and Random Forest do not in either). **An earlier version of this section stated that every model other than Random Forest improves monotonically with margin; that statement was incorrect and is withdrawn.** Random Forest's uniformly low accuracy is largely explained by censoring: among contested environments, in 99.8% of the large-margin stratum and 41.6% of the medium-margin stratum the two best configurations of the winning tier both have true BER below the −20 floor, so the censored target carries no ordering information for them (§VII-L). The margin stratification therefore does not show that Random Forest's failure is \"representational\"; it shows that strict matching penalizes models that reproduce the censored target faithfully.

"""
t=block(t,"### F. LOWCO Configuration-Selection Results","### J. Responsivity Sensitivity",RES)

KOLD="At the ML level, on the regenerated dataset:"
ki=t.index(KOLD); kj=t.index("\n\n",ki)
KNEW="""At the ML level, on the regenerated dataset (strict matching only): LOWCO macro decision accuracy is 76.3% for Linear Regression (vs. 76.4% at baseline) and 9.5% for Random Forest (vs. 8.0% at baseline), so the strict ordering of §VII-F is preserved in direction and approximate magnitude. The strict regression/decision Spearman correlation is ρ=0.245 (p=0.128), versus ρ=0.116 (p=0.476) at baseline; both are descriptive only (§VII-G explains why neither supports a claim of no association). The Very Dense Fog rank correlations (§VII-H) reproduce almost exactly (0.138 for Random Forest and 0.852 for Linear Regression, identical to baseline), consistent with this regime's ground truth being unaffected by rain coefficients. Random Forest's margin-stratified strict accuracy remains uniformly low (2.8%/7.7%/5.4% for large/medium/small margins, vs. 2.5%/6.7%/4.8% at baseline). Under these coefficients Linear Regression, SVR and MLP increase with margin, while XGBoost and Random Forest do not. A no-training check on the saved Random Forest and Linear Regression outputs for this dataset (10,405 environments) gives the same picture as §VII-L–M: oracle ceiling 24.5% macro (50.5% micro), constant (1550 nm, 16-QAM) policy 80.0% macro (59.3% micro), and censoring-equivalent accuracy of 75.2% for Random Forest versus 76.3% for Linear Regression. The other three models and the direct classifiers were not rerun on this dataset."""
t=t[:ki]+KNEW+t[kj:]

LMN="""
### L. Censoring Ceiling and Censoring-Equivalent Scoring
The regression target is censored at log₁₀ BER = −20 (§III-D), but Rule B's third stage compares *predicted* margins. If all candidate configurations in an environment have true BER below 10⁻²⁰, even a perfect predictor of the censored target outputs −20 for all of them; the margins tie and the fixed wavelength order selects 850 nm, whereas the ground truth selects 1550 nm from raw BER differences (for example, on one Clear-sky state the 16-QAM raw BERs are 10⁻¹⁶²² at 850 nm, 10⁻¹³⁹²⁶⁸ at 1250 nm and 10⁻¹⁵³⁸⁴² at 1550 nm, all censored to 10⁻²⁰) that have no operational meaning. Applying Rule B to the *true censored target* gives the oracle ceiling for strict matching:

| Regime | Oracle ceiling (strict) | Linear Regression | Random Forest |
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

Linear Regression exceeds the ceiling in six regimes (the five where the ceiling is 0%, and Dense Fog). Its predictions are never constant across the six configurations (§VII-H), which is consistent with linear extrapolation below −20 preserving the ordering of the configurations; we did not test this explanation directly. A Random Forest cannot output values outside the range of its training targets, so near the floor its predictions tie or differ only by noise. In the five regimes where the ceiling is 0%, strict matching therefore measures whether a model's predictions retain an ordering below the censoring floor, and not decision quality. (These per-regime and ceiling values use the saved Random Forest and Linear Regression predictions for the 0.86 A/W baseline; the Phase 3 predictions reproduce the saved Random Forest and Linear Regression decisions exactly.)

Censoring-equivalent accuracy (§V-F), computed from a fresh rerun of all five models under the identical protocol (§VII-N), is:

| Model | Strict macro | Equivalent macro | Strict micro | Equivalent micro |
|---|---|---|---|---|
| Linear Regression | 76.4% | 76.4% | 61.0% | 61.0% |
| Random Forest | 8.0% | 75.5% | 5.6% | 43.4% |
| XGBoost | 39.0% | 79.3% | 38.0% | 57.2% |
| SVR | 39.5% | 66.1% | 20.8% | 39.4% |
| MLP | 33.1% | 71.3% | 18.9% | 37.6% |

Under this scoring Random Forest is no longer the weakest model and the five regressors lie within 13 points of each other. The equivalence metric is post hoc and lenient by construction, and it was introduced because of the audit described here; it does not replace the pre-specified strict endpoint, which we continue to report. Within-regime bootstrap resampling of environments gives 95% intervals of about ±1 point on every macro value, but this ignores regime-level variability: with only eight regimes, differences of a few points between models should not be interpreted.

### M. Trivial Baselines and Direct Classification
Five of the eight regimes (Clear, Visibility Regimes 1 and 2, Rain: Light, Rain: Moderate; 2,744 of 10,433 environments) have (1550 nm, 16-QAM) as the ground-truth decision in 100% of environments. A constant policy that always selects (1550 nm, 16-QAM), the most frequent label in every training fold, therefore attains 79.9% macro and 59.1% micro strict accuracy, above every regression-based selector in §VII-F (best: Linear Regression, 76.4% macro, 61.0% micro). Untuned direct classifiers (§V-F) reach 70.8% (Logistic Regression), 82.9% (Random Forest) and 81.9% (XGBoost) macro strict accuracy, and 55.4%, 61.1% and 60.7% micro. The three regimes with genuinely contested decisions are:

| Strict accuracy (equivalent accuracy in parentheses where different) | Dense Fog | Rain: Heavy | Very Dense Fog |
|---|---|---|---|
| Oracle ceiling (censored target) | 52.7% | 66.0% | 73.6% |
| Constant (1550, 16) | 53.7% | 46.6% | 39.1% |
| Linear Regression | 70.7% | 55.2% | 39.1% |
| Random Forest | 54.3% (75.2%) | 0.0% (21.3%) | 8.7% (14.8%) |
| XGBoost | 39.5% (53.4%) | 40.7% (41.2%) | 40.2% (40.6%) |
| SVR | 54.9% | 4.3% (15.3%) | 20.0% (25.0%) |
| MLP | 42.0% (58.3%) | 10.1% | 8.6% (15.9%) |
| Direct Logistic Regression | 65.4% | 46.4% | 39.1% |
| Direct Random Forest | 75.2% | 46.6% | 41.7% |
| Direct XGBoost | 66.9% | 46.6% | 41.8% |

In Very Dense Fog the best model (41.8%) is within 3 points of the training-majority constant (39.1%), and in Rain: Heavy the best (Linear Regression, 55.2%) exceeds it by 8.6 points; a hindsight constant of (1550 nm, 4-QAM) would score 60.6% and 53.4% in these regimes but is not available to a controller trained on the other regimes. Only in Dense Fog do several models (Linear Regression, equivalence-scored Random Forest, and the direct classifiers) clearly exceed the constant policy. The Random Forest and XGBoost direct classifiers achieve their high macro values mainly by predicting the dominant class (100% strict accuracy in each of the five single-label regimes); no direct classifier solves the contested regimes under LOWCO either. Because only 3 of 8 regimes carry decision content and these classifiers were untuned, these results indicate neither that direct classification is preferable nor that it is inadequate.

![Fig. 1](fig1_regime_accuracy.png)

**Fig. 1.** Strict decision accuracy per held-out regime (LOWCO, baseline parameterization) for the censoring-oracle ceiling (Rule B applied to the true censored target), the constant (1550 nm, 16-QAM) policy, and three regression-based selectors. In the five single-label regimes (left) the oracle ceiling is 0% and the constant policy scores 100%; the contested regimes (right) are where models can differ from trivial baselines. Generated from the saved decision tables by `analysis/09_make_figure.py`; Random Forest values in the five single-label regimes are 0.0–0.7%.

### N. Reproduction of the Saved Results
Every strict decision in the saved Phase 2B.1 detail table (52,165 model-environment decisions) was re-derived from fresh LOWCO fits of all five models with the original seeds and hyperparameters: Linear Regression, XGBoost, SVR and MLP reproduced 100% of decisions, and Random Forest 99.98% (2 of 10,433 decisions differ, one in Very Dense Fog and one in Visibility Regime 1). The saved Phase 3 per-configuration predictions reproduce the saved Random Forest and Linear Regression decisions in 20,866 of 20,866 cases, and the saved rank correlations to within 10⁻¹⁶. The macro/micro strict accuracies, the infeasible-selection rate in Very Dense Fog (83.1%), the margin-stratified accuracies and the ρ=0.116 (p=0.476) correlation were recomputed from the saved files and match the values reported above. The rain-coefficient sensitivity results of §VII-K come from a separate full-pipeline rerun on the regenerated dataset (Phase 5 closure); their headline values were checked against the saved outputs at that time and were not re-derived again in this final-candidate audit. The environment was Python 3.13.16 with numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, XGBoost 3.4.1 and SciPy 1.18.1; the versions used for the original runs were not logged, so exact bit-level equality is not guaranteed (as the two Random Forest differences show).

"""
t=rep(t,"\n---\n\n## VIII. Discussion","\n"+LMN.rstrip("\n")+"\n\n---\n\n## VIII. Discussion")

# ---- 8. discussion ----
DISC="""## VIII. Discussion

### A. Why Regression Accuracy Need Not Track Decision Accuracy
Rule B's decision depends on three logically distinct sub-tasks: the relative ordering of the six candidates, the feasibility classification against a fixed threshold, and a final lexicographic optimization. A regressor trained to minimize pointwise squared error has no explicit incentive to preserve ordering or threshold placement, an observation in line with the general decision-focused-learning view that prediction error and decision error differ [22], and with work that trains surrogates to preserve the *ranking* of configurations rather than regress their performance [23]. Those works are from other domains and we do not claim that their methods would help here.

### B. Why the Strict Ranking of Models Is Misleading
The strict Random Forest/Linear Regression reversal is produced mainly by the interaction of target censoring with the tie-break (§VII-L). A model that reproduces the censored target faithfully outputs (near-)identical values at the floor for deep-feasible configurations; the margins tie, or differ only by noise, and the selected wavelength is arbitrary relative to the raw-BER ground truth. Linear Regression's predictions retain an ordering across the six configurations in these regimes (§VII-H) and it is scored correct; this is consistent with, but not proof of, linear extrapolation below the floor, and in any case reflects a property of the predictions and not demonstrated decision modelling. The first-pass reading of the reversal as a \"ranking-preservation collapse\" in Random Forest is therefore withdrawn for the five regimes in which the oracle ceiling is 0%. In the contested regimes, Random Forest, SVR and MLP are genuinely weaker than Linear Regression in Rain: Heavy and, to a lesser degree, in Very Dense Fog (§VII-M), and Random Forest's rank correlation in Very Dense Fog is 0.138 (§VII-H); the mechanism behind these deficits is not established here.

### C. Genuine Difficulty Is Concentrated in Fog and Heavy Rain
Contested, small-margin decisions are concentrated in Very Dense Fog and Rain: Heavy, the regimes in which regression generalization is also poorest and in which no model clearly beats a constant policy (§VII-M). The hardest-to-generalize conditions are thus also those with the hardest decisions. With only eight regimes (and only three with decision content), this is a descriptive observation and not a statistical conclusion.

### D. Implications for ML Evaluation in Adaptive FSO
Three practical rules follow. (i) Random-split regression metrics are an unsafe basis for model selection when the deployment task is configuration selection under unseen conditions. (ii) Decision-level evaluation must be accompanied by an oracle ceiling for the chosen target and decision layer, by an equivalence-aware score when the target is censored, and by trivial (constant, majority-class) baselines, otherwise class imbalance across regimes and censoring artifacts can dominate the comparison, in either direction. (iii) Macro averages over regimes should be accompanied by per-regime results and by an explicit count of regimes that carry decision content.

---

"""
t=block(t,"## VIII. Discussion","## IX. Limitations and Threats to Validity",DISC)

# ---- 9. limitations ----
t=rep(t,"the headline findings (the RF/LR reversal, the null regression-decision correlation, the Very Dense Fog rank-correlation collapse, and the margin-stratified Random Forest pattern) all held in direction and approximate magnitude, though exact point estimates shifted (e.g., the correlation moved from ρ=0.116 to ρ=0.245, both non-significant).","the strict-matching results (the Random Forest/Linear Regression ordering, the Very Dense Fog rank-correlation contrast and the margin-stratified Random Forest pattern) held in direction and approximate magnitude, while exact point estimates shifted (e.g., the strict regression/decision correlation moved from ρ=0.116 to ρ=0.245, both descriptive only). The censoring ceiling, constant-policy and equivalence checks were repeated under these coefficients for Random Forest and Linear Regression only (§VII-K); the other models and the direct classifiers were not.")
t=rep(t,"the ranking-collapse mechanism is demonstrated in full depth only for the Random Forest/Linear Regression comparison, not XGBoost, SVR, or MLP individually; the 0.92 A/W responsivity sensitivity was not tested at the full LOWCO decision-mechanism level.","per-environment predictions were saved in the original runs only for Random Forest and Linear Regression (all five models were rerun for §VII-L–N); the 0.92 A/W responsivity sensitivity was not tested at the full LOWCO decision-mechanism level.")
LIM2="""**Evaluation design (final candidate):** the primary strict decision metric is confounded by target censoring and class dominance (§VII-L–M); the censoring-equivalent metric, the constant-policy baselines and the direct classifiers were added after the Phase 5 results were known and are therefore post hoc (the direct classifiers and constant policies were run once, with no tuning); the ground-truth Rule B itself chooses among configurations on BER margins below 10⁻²⁰ that are analytic extrapolations of a Gaussian-tail model and not physically meaningful (a floor-aware ground truth would change the benchmark and was not attempted); only eight regimes are available, three of which carry decision content, so regime-level conclusions are low-powered and the Spearman p-values in §VII-G are descriptive; within-environment rank correlation is undefined in 28.5% of environments (§VII-H); the censoring-equivalent and baseline analyses were not repeated for the 0.92 A/W responsivity case, and for the alternative rain coefficients only for Random Forest and Linear Regression (no-training check); the Random Forest reproduction differs in 2 of 10,433 decisions, and original package versions were not logged.

"""
t=rep(t,"\n---\n\n## X. Conclusion","\n"+LIM2.rstrip("\n")+"\n\n---\n\n## X. Conclusion")

# ---- 10. conclusion ----
CONC="""## X. Conclusion

We evaluated five ML models as BER surrogates for a deterministic adaptive wavelength/modulation controller for OFDM-FSO under random-split, Leave-One-Weather-Condition-Out and parameter-range-holdout protocols. Random-split performance can look strong (R² up to 0.952) while LOWCO regression R² is negative on average for every model, so random-split regression evaluation is not a safe basis for choosing a surrogate for unseen atmospheric conditions. Strict decision-match accuracy initially appeared to reverse the regression ranking (Random Forest 8.0% vs. Linear Regression 76.4% macro), but an audit showed that this ranking is largely an artifact of censoring the regression target at log₁₀ BER = −20 (a perfect predictor of the censored target reaches only 24.0% macro) and that a constant 1550 nm / 16-QAM policy (79.9% macro) outperforms every regression-based selector. Under censoring-equivalent scoring all five regressors reach 66–79% macro accuracy, and in the regimes with genuinely contested decisions (Very Dense Fog, Rain: Heavy) no evaluated model, including direct classifiers, clearly outperforms trivial baselines. We do not demonstrate a reliable ML-based adaptive FSO controller, nor robust generalization to unseen atmospheric conditions, and we withdraw the first-pass claims that regression R² and decision accuracy are uncorrelated and that Random Forest suffers a ranking-preservation collapse across regimes. The contribution is an evaluation lesson: decision-level LOWCO evaluation of FSO surrogates should report oracle ceilings, equivalence-aware scores, trivial baselines and per-regime results alongside regression metrics. Extending the benchmark with a floor-aware ground truth, more regimes and measured data are the natural next steps; mitigation by ranking-aware training or calibration is premature until the evaluation issues above are resolved.

---

"""
t=block(t,"## X. Conclusion","## References",CONC)

# ---- 11. references ----
NEWREF="""[10] R. Younes, F. Ghosna, M. Nassr, M. Anbar, and H. A. A. Alasadi, "Predicting BER value in OFDM-FSO systems using Machine Learning techniques," Spanish Optical Society (SEDOPTICA), vol. 55, no. 4, 2022, article 51114, doi: 10.7149/OPA.55.4.51114. [Online]. Available: https://opa.sedoptica.es/en/article/1836

[11] J. Shao, Y. Liu, X. Du, and T. Xie, "Adaptive modulation scheme for soft-switching hybrid FSO/RF links based on machine learning," *Photonics*, vol. 11, no. 5, Art. no. 404, Apr. 2024, doi: 10.3390/photonics11050404.

[12] S. Derouiche, S. Kameche, and H. E. Adardour, "Adaptive modulation selection based on deep neural networks for optimizing mixed FSO/MIMO-MMW systems in 5G and 6G networks," *Telecommun. Syst.*, vol. 88, no. 3, 2025, doi: 10.1007/s11235-025-01316-9.

[13] M. A. Esmail, "Autonomous self-adaptive and self-aware optical wireless communication systems," *Sensors*, vol. 23, no. 9, Art. no. 4331, Apr. 2023, doi: 10.3390/s23094331.

"""
t=rep(t,"[14] A. G. Alkholidi",NEWREF+"[14] A. G. Alkholidi")
ENDREF="""[22] A. N. Elmachtoub and P. Grigas, "Smart 'predict, then optimize'," arXiv:1710.08005, v5, Nov. 2020. [Online]. Available: https://arxiv.org/abs/1710.08005

[23] A. S. Khazi, S. Pineda Arango, and J. Grabocka, "Deep ranking ensembles for hyperparameter optimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023, arXiv:2303.15212. [Online]. Available: https://arxiv.org/abs/2303.15212

"""
t=rep(t,"\n---\n\n## Final Self-Audit","\n"+ENDREF.rstrip("\n")+"\n\n---\n\n## Final Self-Audit")

# ---- 12. self-audit ----
AUD="""## Final Self-Audit

*(Internal working section — remove before submission.)*

| Potential Reviewer Concern | Where Addressed | Evidence | Remaining Risk |
|---|---|---|---|
| Title accuracy | Title; §X | Title describes an evaluation study with its two pitfalls; no claim of robustness | Low. The author must confirm; the sprint brief used a different working title |
| Novelty claim | §I, §II | Scoped to \"the sources we reviewed\"; limits of the search stated; no claim of priority | Medium. Search used general web search and page-level reading, not IEEE Xplore, Scopus or Web of Science; two leads unverified (Springer s11082-024-06950-2; HAL survey) and not cited |
| Primary-endpoint integrity | §V-E, §V-F, §VII-F | Strict macro accuracy reported unchanged; new metrics labelled post hoc | Low |
| Censoring confound | §VII-L, §VIII-B | Oracle ceiling 24.0% macro; equivalence scoring; margin-censoring link (99.8% / 41.6%) | Medium. Ground-truth Rule B itself uses sub-1e-20 margins; floor-aware ground truth not attempted |
| Trivial-baseline dominance | §VII-M | Constant policy 79.9% macro; per-regime table | Low |
| Withdrawn claims | §VII-G, §VII-I, §VIII-B, §X | ρ≈0 claim, \"RF ranking collapse across regimes\" and \"others improve monotonically with margin\" withdrawn and logged | Low |
| Statistical inference | §VII-G, §VII-L | Spearman p-values flagged descriptive; only 8 regimes; bootstrap ignores regime variability | Medium |
| Data/configuration leakage | §IV-E, §V-F | Environment-level splitting; direct classifiers use env-level inputs; held-out regime never used in fitting | Low |
| Reproducibility | §V-G, §VII-N | Saved results re-derived; RF 99.98%, others 100%; versions recorded; original versions not logged | Low–Medium. Repository URL placeholder must be filled |
| Physical-model limitations / hardware validation | §IX | Unchanged from Final Handoff | Medium (unavoidable here) |
| Sources not visually verified | §III, §IV, refs [16]–[21] | Manufacturer PDFs could not be downloaded in this environment | Medium. Author to check the datasheet values manually (open item O5) |
| Rain-coefficient / responsivity sensitivity | §VII-J, §VII-K | μ=0 full rerun (strict metrics); ceiling/constant/equivalence checked for RF and LR only (no training); joint test with 0.92 A/W not done | Medium |
"""
i=t.index("## Final Self-Audit"); t=t[:i]+AUD
open(DST,"w",encoding="utf-8").write(t); print("written",len(t))
