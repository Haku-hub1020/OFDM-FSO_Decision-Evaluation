# Protocol for the single new experiment (declared BEFORE execution, 2026-10-21)

Purpose: (1) make the censoring-aware secondary decision metric available for all five models (saved predictions exist only for RF and LR);
(2) reproduce the saved Phase 2B.1 decisions as a check on the current software environment; (3) add a direct-classification baseline.

Fixed (identical to Phase 2B.1; nothing tuned): dataset baseline_data/dataset_primary_R1250_0.86.parquet; seed 20260906; features V_km_observed, R_mmh_observed, L_m, wavelength_nm, modulation_M;
target log10_BER_target; the five regressors with their Phase 2B hyperparameters; SVR/MLP 15,000-row cap via default_rng(0); the 8 LOWCO folds (env-level);
Rule B exactly as in phase2b1_full_lowco_audit.py.

Primary endpoint: UNCHANGED = strict Rule-B decision match, macro over the 8 regimes (the value reported in Phase 5).
New SECONDARY metric (post hoc, motivated by an observed confound -- see report): "censoring-equivalent" accuracy = the ML-chosen configuration is in the set of
configurations that Rule B cannot distinguish when applied to the TRUE censored target log10_BER_target (same tier, equal predicted margin).
Also reported: oracle ceiling = strict accuracy of Rule B applied to the true censored target.

Direct classifiers (exploratory baseline, hyperparameters copied from the regressor counterparts, no tuning): multinomial LogisticRegression (scaled, max_iter=2000),
RandomForestClassifier (200 trees, depth 14, min leaf 2), XGBClassifier (300 trees, depth 6, lr 0.1, subsample 0.8). Inputs: env-level V_obs, R_obs, L. Label: ground-truth Rule-B config.

Leakage safeguards: held-out regime is never used for fitting, scaling, early stopping or label encoding. No hyperparameter choice depends on any held-out result.
Stopping criteria: ONE run per model per fold. No reruns with changed settings. If a reproduction check fails (strict-decision agreement with the saved Phase 2B.1 detail < 99.9% for a model),
the discrepancy is reported and the new run is labeled a new run; saved Phase 5 numbers are NOT overwritten.
Compute: 2 CPU cores; expected tens of minutes.
Outputs only in final_candidate/outputs/. Phase 5 files are read-only inputs.
