# OFDM-FSO Decision-Level Evaluation — Curated Research Package

This repository supports the manuscript:

**Decision-Level Evaluation of ML Surrogates for OFDM-FSO Configuration Selection Under Unseen Atmospheric Conditions: Censoring Artifacts and Trivial-Baseline Pitfalls**

## Scope and interpretation

This is a curated research package assembled from the supplied research archive and the latest available final-candidate audit bundle. It is not a claim that every result can already be reproduced from a clean checkout with one command. The author must still verify the scripts' path assumptions, dependencies, exact data lineage, and end-to-end execution.

The paper is an evaluation-methodology study. It does not establish a robust adaptive OFDM-FSO controller.

## Contents

- `manuscript/`: final-candidate Markdown manuscript and its main figure.
- `src/final_audit/`: analysis and audit scripts from the final-candidate sprint.
- `src/physics_model/`: physics-model source files preserved from the Phase 5 reproducibility bundle.
- `data/`: primary and robust Parquet datasets and environmental-state CSV.
- `results/final_candidate/`: final-candidate derived tables and prediction artifact.
- `results/phase3_baseline/`: saved Phase 3 baseline tables.
- `docs/`: declared protocol, completion checklist, and audit/run logs.

## Important metric definitions

- **Strict decision accuracy** measures agreement with the original Rule-B ground-truth label; macro (mean over regimes) and micro (pooled over environments) are different and should be reported separately.
- **Clipped-truth replay (diagnostic)** applies Rule B to clipped BER targets and compares the resulting selections with labels derived from original decisions. It is not a general upper bound.
- **Tie-tolerant agreement** checks whether a selected configuration belongs to the set of configurations indistinguishable under Rule B on the clipped target. It is a diagnostic, not ordinary accuracy and not a model-ranking metric.

The analysis identifies a mismatch between clipped regression targets and ground-truth decisions based on unclipped BER margins. Interpret strict decision scores in light of this limitation.

## Reproduction status

**Reproduction is not yet certified.** Before describing the repository as fully reproducible, the author should:

1. Inspect scripts for hard-coded paths and confirm every referenced input is included.
2. Confirm that the final-candidate scripts use the included versions of datasets and physics models, rather than files from another workspace.
3. Run the workflow from a clean clone and record the exact commands and output.
4. Record Python and package versions, including relevant versions of NumPy, pandas, scikit-learn, XGBoost, Matplotlib, and PyArrow.
5. Verify all reported numbers against the manuscript, tables, and figure.
6. Add a data dictionary, source-data provenance, and checksums for datasets.
7. Confirm redistribution rights for all files before making the repository public.

No generic one-command reproduction command is provided because the final scripts' data paths and dependencies have not yet been validated from a clean checkout.

## Publication and repository metadata

Replace the manuscript's repository URL placeholder only after this package is published at the final repository URL. Do not add a license until the author has confirmed rights to license the included code and data. Manufacturer datasheet values and novelty claims remain subject to the manual verification listed in the completion checklist.

## Citation

Add the final bibliographic citation for the paper after acceptance/publication.
