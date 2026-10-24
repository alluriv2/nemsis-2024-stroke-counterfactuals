# Counterfactual Intervention Analysis on EMS Stroke Response Times
### NEMSIS 2024 National Dataset

**Author:** Vaibhav Alluri  
**Institution:** Clarkson University  
**Date:** 2026

---

## Abstract

Prolonged EMS scene times in stroke/CVA calls are associated with worse patient outcomes, yet the specific operational factors driving delays remain poorly understood at a national scale. This project applies a counterfactual machine learning framework to the 2024 NEMSIS National EMS Dataset to identify actionable interventions — combinations of procedures, medications, and response patterns — that are predicted to reduce scene times exceeding 10 minutes. Binary classifiers are trained to predict long scene time, and counterfactual substitution is used to simulate the effect of replacing high-risk feature values with empirically low-risk alternatives, quantified by delta, efficiency, and harm rate metrics.

---

## Background / Motivation

Stroke is a leading cause of death and long-term disability in the United States, and time-to-treatment is the single most important determinant of outcomes. EMS scene time — the interval between unit arrival and patient departure — is a modifiable component of the pre-hospital timeline, yet national patterns in what drives prolonged scene times for stroke calls are not well characterized. Existing literature largely focuses on transport time or hospital door-to-needle time, leaving a gap in understanding which EMS operational variables are most amenable to intervention. This project addresses that gap by leveraging the full 2024 NEMSIS national dataset and framing the problem as a counterfactual policy simulation: given a patient encounter predicted to have a long scene time, what changes to EMS practice would a model predict would have changed that outcome?

---

## Methodology

### Stage 1 — Data Acquisition & Conversion

**Script:** `Scripts/ascii64_to_parquet_batch_processing.py`

The NEMSIS 2024 national dataset is distributed as a collection of ASCII-64 pipe-delimited (` ~|~`) flat files totalling approximately 300 GB across 51 tables. This script converts each file to Parquet format using PyArrow with Snappy compression, processing files in descending size order and writing in 1-million-row chunks to manage memory. The conversion is incremental — already-converted files are skipped on re-run.

- **Input:** `ASCII_Files/*.txt`
- **Output:** `Main_Parquet_files/*.parquet`

---

### Stage 2 — Dataset Construction

**Script:** `Scripts/DB_Creation_from_parquet.ipynb`

A DuckDB in-process database is built from the 51 source parquets. Stroke and CVA cases are isolated by filtering on the EMS dispatch complaint field (`eDispatch_01`) for the NEMSIS stroke/CVA code (`2301067`). The filtered cohort is then enriched through a series of LEFT JOINs to fact tables covering vitals, medications, procedures, diagnoses, symptoms, response times, and outcomes.

**Numeric multi-valued features** (vitals such as blood glucose, GCS, blood pressure, heart rate, respiratory rate, and SpO2) can have multiple recorded values per encounter. These are aggregated to MIN, MAX, and AVERAGE per `PcrKey`, producing three scalar columns per vital sign. Records where no valid numeric value was recorded are excluded.

**Categorical multi-valued features** (medications administered, procedures performed, impression codes, symptoms, protocols, etc.) can also have multiple values per encounter. Rather than one-hot encoding at this stage, all distinct values for a given field and `PcrKey` are concatenated into a single delimited string (e.g., `"medication_A__medication_B"`). This preserves the combination as a single categorical identity. Records where all values were not-a-code markers (`'.'`, empty string) are excluded entirely.

**Lookup tables** (`Data_Files/2024_lookups_parquet/`) — one parquet per categorical field — store the mapping from each unique delimited-string combination to a surrogate integer key. Schema: `enc_id` (int) → `combo_key` (string). These are the decoding reference: given a surrogate integer from the model or results, the corresponding `combo_key` is the original concatenated NEMSIS code string. They are never joined to the master dataset — they are used exclusively for post-hoc interpretation of coded values.

**Coded tables** (`Data_Files/2024_coded_parquet/`) — one parquet per categorical field — store the per-`PcrKey` surrogate key assignment. Schema: `PcrKey` → `feature_ID` (int). These map each encounter to the surrogate integer assigned to whatever combination of values was observed for that field. They carry no descriptions and cannot be used for decoding; the corresponding lookup table is required for that. Coded tables are joined to the master dataset to replace raw delimited strings with compact integer codes for modeling.

The final master dataset is assembled by progressively LEFT JOINing each coded table to the stroke cohort on `PcrKey`, then applying a final cleaning pass to remove records with invalid or missing `PcrKey` values. Demographic and geographic fields (age, census region, urbanicity) are attached from the `ComputedElements` table.

- **Input:** `Main_Parquet_files/*.parquet`
- **Output:** `Data_Files/stroke_master_final_cleaned.parquet`, `Data_Files/2024_lookups_parquet/`, `Data_Files/2024_coded_parquet/`

---

### Stage 3 — Feature Selection

**Script:** `Scripts/Stat_Analysis.Rmd`

Statistical feature selection is applied to the stroke master dataset through a sequential pipeline. The input dataset contains **1,050,785 encounters across 131 columns**. No train/test split is performed at this stage — filtering uses the full dataset to maximise statistical power for identifying irrelevant features. The pipeline reduces this to **160,566 complete-case encounters and 33 features** passed to the modeling stage.

#### Feature Type Classification

Before any filtering, features are partitioned into three groups:

**Manual drops (`cols_to_drop`)** — excluded by design before any statistical test:

| Column | Reason |
|--------|--------|
| `PcrKey_1` | Duplicate encounter identifier |
| `USCensusRegion`, `USCensusDivision`, `NasemsoRegion`, `Urbanicity` | Geographic stratifiers — retained as subgrouping variables in Stage 5, not predictors |
| `eSituation_01`, `eSituation_18`, `eScene_09` | Excluded by design |
| `ageinyear` | Raw age string; replaced by coded `ePatient_15` |
| `min_eVitals_18`, `max_eVitals_18` | Min/max blood glucose excluded; average (`avg_eVitals_18`) retained |
| `PcrMedicationKey_ID`, `PcrVitalKey_ID`, `PcrProcedureKey_ID` | Internal join keys with no predictive meaning |

**Numeric features (`num_feat`)** — EMS call timing intervals, patient age, and mean vital signs:
`EMSDispatchCenterTimeSec`, `EMSChuteTimeMin`, `EMSSystemResponseTimeMin`, `EMSSceneResponseTimeMin`, `EMSSceneTimeMin`, `EMSSceneToPatientTimeMin`, `EMSTransportTimeMin`, `EMSTotalCallTimeMin`, `ePatient_15`, `avg_eVitals_06`, `avg_eVitals_10`, `avg_eVitals_12`, `avg_eVitals_14`, `avg_eVitals_27`, `avg_eVitals_18`

**Categorical features (`cat_feat`)** — everything remaining after removing `cols_to_drop` and `num_feat`.

#### Data Preparation

**Missingness filter** — a per-column NA rate is computed across the full dataset. Any column with ≥ 35% missing values is dropped. **62 of 131 columns** are removed at this step, leaving **69 columns**.

**Complete-case analysis** — after the missingness filter, `drop_na()` removes all rows containing any remaining NA value. This reduces **1,050,785 rows to 160,566** complete-case encounters used for all downstream modeling.

**Binary target** — `scene_time_binary = if_else(EMSSceneTimeMin <= 10, 0L, 1L)`. The raw `EMSSceneTimeMin` and `PcrKey` columns are then removed from the working dataset to prevent leakage. The target is cast to a two-level factor `{0, 1}`.

#### Statistical Filtering Pipeline

All four filters operate on the complete-case dataset with `α = 0.05` and Benjamini-Hochberg multiple testing correction. Removed features are tracked per-technique.

**1. Spearman correlation filter (numeric features)**

A pairwise Spearman correlation matrix is computed via `Hmisc::rcorr` on all numeric predictors. `caret::findCorrelation(cor_r, cutoff = 0.8)` identifies the minimal set of columns to remove such that no remaining pair exceeds r = 0.8. Removed columns are recorded in `removed$correlation`.

**2. Near-zero variance filter (all predictors)**

`caret::nearZeroVar(saveMetrics = TRUE)` is applied to all remaining predictors. Columns flagged as `nzv == TRUE` — those with very low frequency ratios or near-constant distributions — are dropped. Removed columns are recorded in `removed$nzv`.

**3. Chi-square test (categorical features)**

Each surviving categorical feature is tested for independence from `scene_time_binary` using a contingency table chi-square test. When any expected cell count is below 5, a Monte Carlo simulated p-value is used (`simulate.p.value = TRUE, B = 2000`) to handle sparse tables. BH-adjusted p-values below α are retained; the rest are recorded in `removed$chi_square`.

**4. Kruskal-Wallis test (numeric features)**

Each surviving numeric feature (from the post-correlation set) is tested for distributional differences across target classes using `kruskal.test`. BH-adjusted p-values below α are retained; the rest are recorded in `removed$kruskal`.

In total, **98 of 131 columns are removed** across all steps (14 manual drops + 62 high-missingness + statistical filters), leaving **33 features** passed to the modeling stage alongside the binary target.

#### Outputs

Three CSVs are written for use by the Python modeling pipeline:

| File | Contents |
|------|----------|
| `removed_features.csv` | Union of all 98 removed columns: manual drops + high-missingness drops + all four filter removals |
| `cols_to_drop.csv` | Manual design drops only (`cols_to_drop`) |
| `num_feat.csv` | Numeric feature list — used by Python to route features to `StandardScaler` vs `OneHotEncoder` |

- **Input:** `Data_Files/stroke_master_final_cleaned.parquet`
- **Output:** `Stat_Outputs/cols_to_drop.csv`, `Stat_Outputs/num_feat.csv`, `Stat_Outputs/removed_features.csv`

---

### Stage 4 — Problem Formulation & Baseline Modeling

**Script:** `Scripts/baseline_script.ipynb`

#### Problem Formulation

The task is framed as binary classification on 160,566 complete-case stroke encounters (no imputation). The target variable is defined as:

- **y = 1:** EMS scene time > 10 minutes (86.8% of encounters)
- **y = 0:** EMS scene time ≤ 10 minutes

Stage 4 is purely predictive — its goal is to understand which features are most discriminative for long scene time and to establish benchmark classification performance. The feature importance rankings produced at this stage directly inform which features are selected for intervention analysis in Stage 5.

#### Model Development Pipeline

Four classifiers are evaluated:

| Model | Hyperparameters Searched |
|-------|--------------------------|
| Logistic Regression | `C: [0.01, 0.1, 1, 10]` |
| Decision Tree | `max_depth: [5, 10, 15, None]`, `min_samples_split: [2, 5, 10]` |
| Random Forest | `n_estimators: [100, 200]`, `max_depth: [10, 15, 20]` |
| XGBoost | `n_estimators: [100, 200]`, `max_depth: [3, 6, 10]`, `learning_rate: [0.01, 0.1, 0.2]` |

**Preprocessing:** Categorical features → `OneHotEncoder(handle_unknown="ignore")`; numeric features → `StandardScaler()`, assembled via `ColumnTransformer`.

**Class imbalance:** Three workflows — unbalanced, class-weighted (`compute_class_weight("balanced")`; XGBoost uses `scale_pos_weight = n_neg/n_pos`), and undersampled (`RandomUnderSampler`).

The baseline runs both TVT and OOF pipelines under the macro F1 objective, then calls `pick_winner_between_tvt_and_oof` to select the single best model across both strategies. Feature importances are extracted from the winner (`feature_importances_` for tree-based models, `|coef_|` for Logistic Regression). OHE features are collapsed back to parent feature names and importances summed. The top-ranked features are grouped into five categories:

| Category | Features | Role in Stage 5 |
|----------|----------|----------------|
| **Clinical** | `eProcedures_03`, `eSituation_09`, `eSituation_11`, `eProcedures_06`, `eSituation_13` | Counterfactual substitution candidates |
| **Operational** | `eDisposition_20`, `eResponse_24`, `eDisposition_17`, `eResponse_07`, `eScene_01`, `eDisposition_32`, `eProcedures_10` | Counterfactual substitution candidates |
| **Timing** | `EMSTotalCallTimeMin`, `EMSSceneToPatientTimeMin`, `EMSTransportTimeMin`, `EMSSceneResponseTimeMin`, `EMSChuteTimeMin` | Contextual / excluded from substitution |
| **Patient** | `eVitals_20`, `eVitals_02`, `ePatient_14`, `ePatient_15`, `avg_eVitals_*` | Subgrouping strata |
| **Geographic** | `eScene_09`, `USCensusDivision`, `USCensusRegion`, `Urbanicity`, `NasemsoRegion` | Subgrouping strata |

- **Input:** `Data_Files/stroke_master_final_cleaned.parquet`, `Stat_Outputs/*.csv`
- **Output:** `Model_Outputs/out_*/candidates_*.csv`, `Model_Outputs/out_*/best_model_*.csv`

---

### Stage 5 — Counterfactual Tuning

**Scripts:** `Scripts/delta_tuned_oof.ipynb`, `Scripts/delta_tuned_tvt.ipynb`, `Scripts/macro_f1_tuned_oof.ipynb`, `Scripts/macro_f1_tuned_tvt.ipynb`

---

#### Stage 5.1 — Concept & Framing

The central question in this stage is: *given a stroke encounter that a model predicts will have a long scene time, what changes to EMS operational practice would the model predict would have prevented that outcome?*

This is addressed through counterfactual substitution. A subset of actionable operational features — ones that reflect EMS decision-making rather than fixed patient or geographic characteristics — is designated as the counterfactual feature set: `CF_FEATURES = ["eResponse_07", "eProcedures_10_ID", "eDisposition_32"]`. These represent EMS response type, procedures performed, and disposition. For each encounter in the test set, the values of these features are replaced with the combination observed to carry the lowest empirical risk of long scene time in the training data. The model is re-scored on this modified dataset, and the shift in predicted outcomes is measured.

This stage goes further than applying a fixed intervention post-hoc. The tuning objective itself — for the delta pipelines — is defined in terms of this intervention signal: models are selected not because they classify well, but because their predictions are most responsive to the substitution. This makes the counterfactual analysis integral to model selection, not just an evaluation step applied after the fact.

**Pipeline Overview**

All four notebooks follow the same high-level flow:

1. **Data loading** — the complete-case stroke master dataset is loaded and split into training and test sets (with an additional validation split in TVT variants). Feature metadata from Stage 3 (`num_feat.csv`, `cols_to_drop.csv`, `removed_features.csv`) is used to partition features into numeric and categorical routing for the preprocessor.

2. **Modeling** — four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) are evaluated under a class-weighted workflow. Preprocessing applies `OneHotEncoder` to categorical features and `StandardScaler` to numeric features via `ColumnTransformer`. Hyperparameter search is performed via `ParameterGrid` with inner K-fold CV.

3. **Model selection** — the objective used to rank hyperparameter configurations and select the decision threshold differs by pipeline: macro F1 pipelines optimise classification balance; delta pipelines optimise intervention responsiveness. The final model is retrained on the full training set before test evaluation.

4. **Counterfactual analysis** — the selected model is used to simulate the effect of the intervention on the test set globally (all encounters) and by urbanicity subgroup (`Urbanicity_ID`). The replacement policy is always derived from training data only. Coded feature values in results are decoded back to human-readable NEMSIS strings using the lookup parquets.

---

#### Stage 5.2 — Data Splitting

Two validation strategies are implemented across the four notebooks.

**Train–Validation–Test (TVT)** — `macro_f1_tuned_tvt.ipynb`, `delta_tuned_tvt.ipynb`

The dataset is split 80/10/10 into training, validation, and test sets. Hyperparameters are selected via inner K-fold CV on the training set; the decision threshold is tuned on the held-out validation set; the model is evaluated once on the test set. This provides a strict three-way separation between parameter learning, threshold selection, and final evaluation — no information from the validation or test sets influences the hyperparameter search.

**Out-of-Fold (OOF)** — `macro_f1_tuned_oof.ipynb`, `delta_tuned_oof.ipynb`

The dataset is split 80/20 into training and test sets. Within the training set, 5-fold stratified CV generates out-of-fold predictions covering all training samples. These OOF predictions serve as a unified surrogate validation set for both hyperparameter ranking and threshold tuning. The final model is retrained on the full training set before test evaluation. OOF is more data-efficient than TVT — all training samples contribute to model selection — and produces a more stable estimate of model behaviour by averaging across five different held-out partitions.

> TVT separates training and validation explicitly with a dedicated held-out set; OOF uses cross-validated predictions to serve both roles from the training data alone.

---

#### Stage 5.3 — Model Selection

**Tuning Objectives**

**Macro F1–Based Tuning** — `macro_f1_tuned_tvt.ipynb`, `macro_f1_tuned_oof.ipynb`

Macro F1 tuning follows a standard classification paradigm. Hyperparameters are selected to maximise macro F1 — the unweighted average of per-class F1 scores — and the decision threshold is swept to maximise the same metric on the validation set (TVT) or OOF predictions (OOF). Using macro F1 rather than accuracy or class-1 F1 ensures the minority class (short scene time, ~13% of encounters) is not ignored during model selection. This objective treats the task purely as a prediction problem — it is used here as the reference baseline against which the intervention-driven delta objective is compared.

**Delta-Based Tuning** — `delta_tuned_tvt.ipynb`, `delta_tuned_oof.ipynb`

Delta-based tuning is a novel model selection strategy developed for this research. Rather than selecting a model based on how well it classifies, it selects based on how strongly its predictions respond to the counterfactual substitution. The central premise is that in a policy simulation setting, a model's value lies in its sensitivity to the intervention — a model that classifies well but whose predictions do not change under substitution provides no actionable signal. Both hyperparameter selection and threshold tuning are driven entirely by the delta objective, with no classification metric involved.

**The Delta Tuning Mechanism**

Delta tuning centres on three functions that together define the counterfactual policy and measure its effect.

**Step 1 — Build the empirical risk table (`scene_time_rates`)**

For every unique combination of `CF_FEATURES` observed in the training data, `scene_time_rates` computes the empirical P(scene_time_binary=1) as the group mean of the binary target. The result is a table of feature combinations ranked by their observed long-scene-time rate. Combinations with fewer than 50 observations (`MIN_COUNT_CF = 50`) are excluded to avoid spurious candidates from rare patterns. This table is always derived from training data only — it is a lookup, not a learned parameter, so using it to evaluate held-out data does not constitute leakage.

**Step 2 — Select the replacement policy (`lowest_prob_combinations_with_dict`)**

`lowest_prob_combinations_with_dict` sorts the risk table ascending and retains the top 20 lowest-risk combinations (`TOP_K_CF = 20`). The single lowest-risk combination — the specific triplet of `eResponse_07`, `eProcedures_10_ID`, and `eDisposition_32` values associated with the lowest empirical P(long scene time) in the training data — is extracted as `replacement_dict`. This dictionary represents the counterfactual policy: it encodes what the operational and procedural profile of the lowest-risk EMS encounters looks like, and is the value set that every test encounter is hypothetically moved to.

**Step 3 — Apply the intervention and compute delta (`counterfactual_replace`)**

`counterfactual_replace` applies the policy to the evaluation set and quantifies the prediction shift:

1. The model is scored on the unmodified evaluation set → baseline predictions `yhat_base`.
2. The CF feature columns are overwritten with `replacement_dict` values for every row → counterfactual dataset.
3. The model is re-scored on the counterfactual dataset → `yhat_cf`.
4. Three metrics quantify the shift:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Delta (Δ)** | `cf_pred_rate_1 − base_pred_rate_1` | Net change in predicted long-scene-time rate — more negative is a stronger intervention signal |
| **Efficiency** | `flip_1→0 / baseline_pred_1_count` | Fraction of predicted-long encounters resolved by the intervention |
| **Harm rate** | `flip_0→1 / baseline_pred_0_count` | Fraction of predicted on-time encounters worsened by the intervention — should remain near zero |

**Step 4 — Model selection ranking**

Candidates are ranked by minimising Δ first, then harm rate, then maximising efficiency:

```
sort_values(["delta", "harm_rate", "efficiency"], ascending=[True, True, False])
```

This priority order ensures the selected model produces the largest predicted reduction in adverse outcomes while keeping unintended worsening of on-time cases minimal and resolving the largest share of predicted delays.

**TVT Variant — Delta (`delta_tuned_tvt.ipynb`)**

Hyperparameter selection uses `compute_delta_inner_cv`, which runs stratified K-fold CV on the training set. In each fold: the model is fit on the fold-train subset; `lowest_prob_combinations_with_dict` is called on the fold-train subset only to derive the replacement dict (no leakage from fold-validation); the intervention is applied to the fold-validation set and delta is computed. The mean delta across all folds ranks the hyperparameter configuration.

After retraining on the full training set, `pick_threshold_by_valid_delta` sweeps the threshold grid on the held-out validation set — for each candidate threshold, `counterfactual_replace` is called and the resulting delta recorded. The threshold with the most negative delta is selected. This produces a clean three-way separation: training set → model parameters, inner-CV delta → hyperparameters, validation set → decision threshold.

**OOF Variant — Delta (`delta_tuned_oof.ipynb`)**

The OOF variant uses the same three core functions but replaces the held-out validation set with out-of-fold predictions. In each CV fold, the replacement dict is learned from the fold-train portion only, and delta is computed on the fold-validation set. Fold-level deltas are accumulated and averaged across all five folds. The aggregated OOF delta ranks hyperparameter configurations and also drives threshold selection, making all training data contribute to the decision without a dedicated validation split.

---

#### Stage 5.4 — Post-Selection Analysis

All four notebooks share the same analysis steps after model selection.

**Feature importance** — OHE feature importances are extracted from the winning pipeline and collapsed to parent feature names. This reveals which features the selected model relied on most, independently of the tuning objective used to select it.

**Global counterfactual analysis** — The replacement dict is learned from the full training set and substituted into every test encounter. The model is re-scored and Δ, efficiency, and harm rate are reported for the full test population.

**Subgroup counterfactual analysis** — The same intervention is repeated independently for each urbanicity level (`Urbanicity_ID`). For each subgroup, a subgroup-specific replacement dict is learned from the corresponding training subset, and delta/efficiency/harm are computed on the matching test rows. This reveals whether the intervention effect is consistent across rural, suburban, and urban settings or whether different replacement policies emerge by geography.

**Decoded results** — Surrogate integer codes in the subgroup results are resolved back to human-readable NEMSIS strings using the lookup parquets (`enc_id → combo_key`). Raw NEMSIS code columns (`eResponse_07`, `eDisposition_32`) pass through unchanged. The `replacement_dict` column is decoded in the same pass.

**Comparative Framework**

The four pipelines are designed to enable direct cross-comparison:

| Objective | Strategy | Notebook |
|-----------|----------|----------|
| Macro F1 | TVT | `macro_f1_tuned_tvt.ipynb` |
| Macro F1 | OOF | `macro_f1_tuned_oof.ipynb` |
| Delta | TVT | `delta_tuned_tvt.ipynb` |
| Delta | OOF | `delta_tuned_oof.ipynb` |

Comparing macro F1 vs delta pipelines on the same split strategy reveals whether optimising for intervention responsiveness produces a different model or just a different threshold. Comparing TVT vs OOF on the same objective reveals how sensitive model selection is to the choice of validation strategy. Together, the four pipelines answer whether a prediction-optimal model and an intervention-optimal model agree on what the best EMS policy looks like.

- **Input:** `Data_Files/stroke_master_final_cleaned.parquet`, `Stat_Outputs/*.csv`
- **Output:** `Model_Outputs/out_*/candidates_*.csv`, `Model_Outputs/out_*/best_model_*.csv`

---

## Results

All four Stage 5 pipelines were executed on **160,566 complete-case encounters** with **33 features**, reduced from 1,050,785 encounters and 131 columns in the raw stroke cohort. The core question is not which pipeline classifies best — it is which pipeline produces the most reliable and actionable intervention recommendation. To answer that, the evaluation is structured as a three-tier comparison: intra-TVT (macro F1 versus delta on the same split), intra-OOF (macro F1 versus delta on the same split), and inter-strategy (the best TVT pipeline against the best OOF pipeline).

---

### Classification Performance (All Pipelines)

Before comparing intervention quality, it is useful to establish the classification baseline. The macro F1 and baseline OOF pipelines produce strong discrimination (macro F1 ≈ 0.685–0.687, ROC-AUC ≈ 0.827–0.828). The delta pipelines deliberately operate at conservative thresholds (0.61/0.69), which concentrates predictions on the highest-confidence positives and compresses classification metrics — but this is by design, not a failure.

Test metrics are from the held-out test set. For TVT pipelines, ROC-AUC is available only from the saved candidates CSV; Macro F1 TVT's test macro F1 is computed from the confusion matrix in the output (LR, t=0.33 on 16,057 test encounters).

| Pipeline | Model | Threshold | Test Macro F1 | Test ROC-AUC | Test PR-AUC |
|----------|-------|-----------|---------------|--------------|-------------|
| Baseline OOF (Stage 4) | XGBoost, unbalanced | 0.73 | 0.687 | 0.828 | 0.721 |
| Macro F1 OOF | XGBoost, weighted | 0.35 | 0.685 | 0.827 | 0.710 |
| Macro F1 TVT | Logistic Regression, weighted | 0.33 | ~0.668 | — | 0.684 |
| Delta TVT | Random Forest, weighted | 0.61 | 0.214 | 0.740 | 0.627 |
| Delta OOF | Random Forest, weighted | 0.69 | 0.190 | 0.753 | 0.634 |

Strong classification performance, however, does not translate directly into a strong intervention signal. A model that flags 83–86% of encounters as positive leaves almost no room for the substitution to produce negative flips. The comparisons below show that the macro F1 pipelines — despite their classification quality — fail as policy instruments, while the delta pipelines succeed precisely because their conservative thresholds make the intervention's effect sharp and interpretable.

---

### Intervention Metrics

Two metrics are used throughout the evaluation to quantify intervention quality:

- **Delta (Δ)** = `cf_pred_rate_1 − base_pred_rate_1` — the absolute change in predicted long-scene-time rate after substituting CF features. A negative value means the intervention reduces predicted adverse outcomes.
- **Relative Δ** = `Δ / base_pred_rate_1` — the percentage reduction relative to the baseline predicted rate. This is the primary comparability metric across subgroups, because absolute delta is sensitive to how many encounters the model flags as positive in the first place. A subgroup with a small absolute Δ may still show a strong relative Δ if the model's baseline positive rate for that group is low.

Two further metrics accompany each result: **efficiency** (fraction of predicted-long encounters resolved by the intervention) and **harm rate** (fraction of predicted on-time encounters inadvertently worsened — should remain near zero).

---

### Pipeline Model Selection

Each pipeline selects the model and threshold whose best configuration scored highest on its objective. For TVT pipelines, selection additionally penalises a large train-to-validation metric gap — a model that overfits its training fold is demoted even if its raw validation score is higher.

| Pipeline | Selected Model | Hyperparameters | Threshold | Selection Score |
|----------|----------------|-----------------|-----------|-----------------|
| Macro F1 TVT | Logistic Regression, weighted | `C=0.1` | 0.33 | valid macro F1: 0.669, gap: 0.013 |
| Macro F1 OOF | XGBoost, weighted | `lr=0.1, max_depth=6, n=100` | 0.35 | OOF macro F1: 0.695 |
| Delta TVT | Random Forest, weighted | `max_depth=15, n=100` | 0.61 | valid Δ: −0.059 |
| Delta OOF | Random Forest, weighted | `max_depth=20, n=100` | 0.69 | OOF Δ: −0.047 |

**Why Logistic Regression is selected for Macro F1 TVT over XGBoost:** XGBoost achieved a higher raw valid macro F1 (0.678 vs 0.669), but its train-to-validation gap was 0.147 — over 10× larger than LR's 0.013. The gap-penalised criterion demotes XGBoost as an overfit candidate; LR's stable generalisation gives it the higher adjusted score despite lower absolute performance.

Both delta pipelines independently converge on Random Forest at a conservative threshold, narrowing the positive prediction pool to 6–9% of encounters. This is the mechanism that makes the CF flip signal actionable — the intervention affects a small, well-defined set of high-confidence predictions rather than an indiscriminate majority.

---

### Intra-TVT Comparison: Macro F1 versus Delta

With models selected, the first comparison tests whether tuning for intervention responsiveness (delta) produces a better policy instrument than tuning for classification balance (macro F1), holding the TVT split strategy fixed.

| Metric | Macro F1 TVT (LR, t=0.33) | Delta TVT (RF, t=0.61) |
|--------|--------------------------|------------------------|
| Valid macro F1 | **0.669** | 0.213 |
| Global Δ (pp) | +4.68 ↑ (worsens) | **−5.75** ↓ (improves) |
| Global relative Δ | +5.6% | **−66.6%** |
| Efficiency | 1.0% | **69.7%** |
| Harm rate | 33.0% | **0.30%** |

The macro F1 TVT model classifies ~83% of test encounters as long scene time. At this low threshold, the model is too indiscriminate to respond to the intervention — substituting CF features increases the predicted positive rate (+4.68 pp) because the replacement combination does not push LR's probability estimate below 0.33. The delta TVT model, operating on only 8.6% of high-confidence positives, resolves 69.7% of them with near-zero harm. **Delta TVT outperforms Macro F1 TVT on every intervention metric.**

---

### Intra-OOF Comparison: Macro F1 versus Delta

The same comparison repeated on the OOF split strategy confirms the pattern — and widens the margin.

| Metric | Macro F1 OOF (XGBoost, t=0.35) | Delta OOF (RF, t=0.69) |
|--------|--------------------------------|------------------------|
| OOF macro F1 | **0.695** | 0.190 |
| Global Δ (pp) | +1.54 ↑ (worsens) | **−5.55** ↓ (improves) |
| Global relative Δ | +1.8% | **−88.7%** |
| Efficiency | 0.8% | **89.1%** |
| Harm rate | 16.0% | **0.03%** |

The macro F1 OOF model achieves the best raw classification performance of all four pipelines (OOF macro F1: 0.695, ROC-AUC: 0.827) but again fails as a policy instrument — global CF delta is marginally positive (+1.8%), efficiency is under 1%, and harm rate is 16%. The delta OOF model sacrifices classification quality for a dramatically cleaner intervention profile: −88.7% relative reduction, 89.1% resolved, and 0.03% harm. **Delta OOF outperforms Macro F1 OOF by an even wider margin than the corresponding TVT comparison.**

---

### Inter-Strategy Comparison: Delta TVT versus Delta OOF

Both intra-strategy comparisons favour the delta pipeline, so the final question is whether the TVT or OOF validation strategy produces the stronger delta model.

| Metric | Delta TVT (RF, t=0.61) | Delta OOF (RF, t=0.69) |
|--------|------------------------|------------------------|
| Global Δ (pp) | −5.75 | **−5.55** |
| Global relative Δ | −66.6% | **−88.7%** |
| Efficiency | 69.7% | **89.1%** |
| Harm rate | 0.30% | **0.03%** |
| Test set size | 16,057 | **32,114** |

Both pipelines select the same model family (Random Forest, class-weighted) and converge on the same global replacement policy. The OOF variant operates on a larger test set (80/20 versus 80/10/10), uses all training data for model selection, and produces a substantially cleaner intervention profile — higher relative reduction, higher efficiency, and ten times lower harm rate. **Delta OOF is selected as the final pipeline.**

---

### Global Counterfactual Results

With Delta OOF as the selected pipeline, the next question is what intervention it actually recommends. The replacement policy is derived from the full training set and is identical across all four pipelines:

| Feature | Replacement Value |
|---------|------------------|
| `eResponse_07` | `2207015` |
| `eProcedures_10_ID` | `3413` |
| `eDisposition_32` | `4232005` |

Applying this substitution to all test encounters produces the following global results:

| Pipeline | Baseline pred rate | CF pred rate | Δ (pp) | Relative Δ | Efficiency | Harm rate |
|----------|--------------------|--------------|--------|------------|------------|-----------|
| Macro F1 TVT | 83.2% | 87.9% | +4.68 | +5.6% | 1.0% | 33.0% |
| Macro F1 OOF | 86.1% | 87.6% | +1.54 | +1.8% | 0.8% | 16.0% |
| Delta TVT | 8.6% | 2.9% | −5.75 | −66.6% | 69.7% | 0.30% |
| Delta OOF | 6.3% | 0.7% | −5.55 | −88.7% | 89.1% | 0.03% |

The macro F1 pipelines predict long scene time for the vast majority of test encounters (83–86% baseline positive rate), leaving little room for negative flips. The delta pipelines at high thresholds operate on a much smaller, more confident positive set (6–9%), and the intervention resolves nearly all of those encounters. The selected pipeline (Delta OOF) drives the predicted long-scene-time rate from 6.3% to 0.7% — a global relative reduction of 88.7%.

---

### Subgroup Counterfactual Results

The global result establishes that the intervention works — the subgroup analysis asks whether it works uniformly across settings or whether different urbanicity levels require different operational profiles. The same substitution is repeated independently per urbanicity level, with each subgroup's replacement policy derived from its own training partition.

Across all subgroups in both pipelines, `eProcedures_10_ID = 3413` and `eDisposition_32 = 4232005` are constant — every urbanicity level converges on the same procedure combination and disposition code as the lowest-risk profile. Only `eResponse_07` varies by setting, so it is the sole column shown in the tables below.

#### Delta TVT — Subgroup Results (t = 0.61)

| Urbanicity | Test size | Baseline pred rate | CF pred rate | Δ (pp) | Relative Δ | Efficiency | Harm rate | `eResponse_07` |
|------------|-----------|-------------------|--------------|--------|------------|------------|-----------|----------------|
| Urban | 13,221 | 7.4% | 2.5% | −4.86 | −65.8% | 69.8% | 0.32% | `2207015` |
| Frontier | 282 | 16.7% | 11.3% | −5.32 | −31.9% | 36.2% | 0.85% | `2207017` |
| Suburban | 1,157 | 10.4% | 5.1% | −5.27 | −50.8% | 56.7% | 0.68% | `2207017` |
| Rural | 1,397 | 17.5% | 12.6% | −4.94 | −28.2% | 37.9% | 2.08% | `2207017` |

#### Delta OOF — Subgroup Results (t = 0.69)

| Urbanicity | Test size | Baseline pred rate | CF pred rate | Δ (pp) | Relative Δ | Efficiency | Harm rate | `eResponse_07` |
|------------|-----------|-------------------|--------------|--------|------------|------------|-----------|----------------|
| Urban | 26,377 | 5.3% | 0.4% | −4.86 | −92.2% | 92.8% | 0.03% | `2207015` |
| Frontier | 545 | 13.8% | 8.1% | −5.69 | −41.3% | 44.0% | 0.43% | `2207017` |
| Suburban | 2,276 | 7.6% | 2.8% | −4.79 | −63.0% | 64.7% | 0.14% | `2207017` |
| Rural | 2,916 | 12.7% | 7.9% | −4.80 | −37.8% | 40.3% | 0.35% | `2207017` |

**Interpreting absolute vs relative delta across subgroups:** Frontier and Rural show larger absolute Δ (−5.3 to −5.7 pp) than Urban (−4.9 pp), yet their relative reductions are lower (−28% to −41% vs −65% to −92%). This is because the model's baseline positive rate is higher in Frontier and Rural settings (12–17% vs 5–7% Urban), meaning there are more predicted-long encounters to begin with — the absolute flip count is larger but so is the denominator, compressing the relative figure. Urban shows the strongest relative signal because the model is most conservative there, and the intervention resolves nearly all of its high-confidence predictions. For small subgroups like Frontier, relative Δ is the more informative metric: a small absolute swing can represent a substantial proportional reduction given the limited pool of flagged encounters.

---

### Conclusion — Suggested Intervention

Taken together, the results establish a clear hierarchy: delta tuning consistently outperforms macro F1 tuning as a model selection objective for policy simulation, and OOF is the more data-efficient and stable split strategy. The suggested intervention, drawn from the selected pipeline (Delta OOF) subgroup results and sorted by most negative absolute Δ, is:

| Urbanicity | Suggested `eResponse_07` | `eProcedures_10_ID` | `eDisposition_32` | Δ (pp) | Relative Δ |
|------------|--------------------------|---------------------|-------------------|--------|------------|
| Frontier | `2207017` | `3413` | `4232005` | −5.69 | −41.3% |
| Urban | `2207015` | `3413` | `4232005` | −4.86 | −92.2% |
| Rural | `2207017` | `3413` | `4232005` | −4.80 | −37.8% |
| Suburban | `2207017` | `3413` | `4232005` | −4.79 | −63.0% |

The procedure combination (`eProcedures_10_ID = 3413`) and disposition code (`eDisposition_32 = 4232005`) are consistent across all urbanicity settings — the model converges on the same operational and clinical profile regardless of geography. The only divergence is in response type (`eResponse_07`): Urban encounters favour `2207015` while all non-urban settings (Frontier, Rural, Suburban) favour `2207017`. Frontier ranks first on absolute Δ (−5.69 pp) but lags on relative Δ (−41.3%) because its baseline positive rate is already elevated (13.8%); Urban leads on relative Δ (−92.2%) because the model is most conservative there — only 5.3% of encounters are flagged, and nearly all flip under the intervention.

Human-readable decoding of NEMSIS code values for `eProcedures_10_ID` is available via `Data_Files/2024_lookups_parquet/lookup_eProcedures_10.parquet` (`enc_id → combo_key`). The `eResponse_07` and `eDisposition_32` values are raw NEMSIS codes and can be looked up in the NEMSIS data dictionary (`Documentations/`).

---

## Repository Structure

```
NEMSIS_2024_Counterfactuals/
├── ASCII_Files/                        # Raw NEMSIS ASCII-64 source files (not tracked)
├── Main_Parquet_files/                 # Stage 1 output — converted parquets (not tracked)
├── 2024_duckdb/                        # DuckDB database — Stage 2 output (not tracked)
├── Data_Files/
│   ├── stroke_master_final_cleaned.parquet   # Stage 2 master dataset
│   ├── 2024_lookups_parquet/           # enc_id → combo_key decode tables (one per field)
│   └── 2024_coded_parquet/             # PcrKey → feature_ID assignment tables (one per field)
├── Stat_Outputs/                       # Stage 3 outputs — feature selection CSVs
├── Model_Outputs/
│   ├── out_oof_80_20_macro_f1/            # Baseline Stage 4 OOF results
│   ├── out_tvt_macro_f1/                  # Stage 5 TVT results (macro F1 + delta — shared dir)
│   ├── out_oof_macro_f1_universal_split/  # Stage 5 OOF macro F1 results
│   └── out_oof_delta_macro_f1/            # Stage 5 OOF delta results
├── Documentations/                     # (local only, not tracked) NEMSIS data dictionary and user manual: download from nemsis.org
└── Scripts/
    ├── ascii64_to_parquet_batch_processing.py   # Stage 1
    ├── DB_Creation_from_parquet.ipynb           # Stage 2
    ├── Stat_Analysis.Rmd                        # Stage 3
    ├── baseline_script.ipynb                    # Stage 4
    ├── macro_f1_tuned_tvt.ipynb                 # Stage 5 — TVT macro F1
    ├── macro_f1_tuned_oof.ipynb                 # Stage 5 — OOF macro F1
    ├── delta_tuned_tvt.ipynb                    # Stage 5 — TVT delta
    └── delta_tuned_oof.ipynb                    # Stage 5 — OOF delta
```

---

## How to Reproduce

**1. Obtain the data**
Download the NEMSIS 2024 National EMS Dataset from [nemsis.org](https://nemsis.org) and place all `.txt` files in `ASCII_Files/`.

**2. Convert to Parquet**
```bash
python Scripts/ascii64_to_parquet_batch_processing.py
```

**3. Build the stroke dataset**
Open and run all cells in `Scripts/DB_Creation_from_parquet.ipynb`.

**4. Run feature selection**
Knit `Scripts/Stat_Analysis.Rmd` in RStudio or render via:
```bash
Rscript -e "rmarkdown::render('Scripts/Stat_Analysis.Rmd')"
```

**5. Train models and generate counterfactuals**
Run any of the four model notebooks in `Scripts/`. To reproduce all results, run all four.

---

## Dependencies

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install R dependencies:
```r
install.packages(readLines("requirements_r.txt") |> sub("==.*", "", x = _))
```

---

## References

- NEMSIS Technical Assistance Center. *National EMS Information System 2024 Public-Release Research Dataset.* [https://nemsis.org](https://nemsis.org)
- NEMSIS Data Dictionary v3.5.0

---

## Data Source and Credit

National Highway Traffic Safety Administration, National Emergency Medical Services Information System. The content reproduced from the NEMSIS Database remains the property of the National Highway Traffic Safety Administration. The National Highway Traffic Safety Administration is not responsible for any claims arising from works based on the original Data, Text, Tables, or Figures.

This repository contains code and aggregate results only. It does not include NEMSIS data. The NEMSIS data dictionary and user manual are available from [nemsis.org](https://nemsis.org); download the Public-Release Research Dataset from there under its terms of use.
