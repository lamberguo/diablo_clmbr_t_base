#Head-only vs Adapter FT

Standalone briefing version for management update.

## 1) Experiment scope and data

- Evaluation setup: compare-eval pipeline (A/B model comparison)
- Split: `val`
- Task count: 15
- Per-task eval cap: `max_examples=3000` (where applicable)
- Compared models:
  - Model 1: **Head-only**
  - Model 2: **Adapter FT model**

Benchmark column is computed from benchmark predictions filtered by `split=val`.
For `chexpert`, benchmark is macro-mean over 14 subtasks.

## 2) Core method used in our pipeline

### Backbone encoder

- Base model: `StanfordShahLab/clmbr-t-base`
- Input: patient timeline truncated at prediction time
- Representation: pooled hidden state at each patient sequence end

### Prediction head structure (our head)

For both head-only and adapter FT workflows, classification head is:

- `classifier = nn.Linear(hidden_size, num_labels)`
- Output:
  - binary tasks: `num_labels=2`
  - lab multiclass tasks: `num_labels=4`
  - chexpert: `num_labels=14` (multi-label)

### Loss function

- `single_label_classification` (binary + multiclass): **CrossEntropyLoss**
- `multi_label_classification` (`chexpert`): **BCEWithLogitsLoss**

## 3) How Head-only and Adapter FT are trained

### A) Head-only training

- Freeze all CLMBR backbone parameters.
- Train only the linear classifier head (`classifier.weight`, `classifier.bias`).
- Objective: supervised task loss (CE or BCEWithLogits as above).

Interpretation: this is linear probing on frozen CLMBR features (with task labels on truncated timelines).

### B) Adapter FT training

- Inject Block/Diablo adapter modules into selected transformer linear layers (`input_proj`, `output_proj`, `final_layer`).
- Keep original backbone frozen.
- Train:
  - adapter parameters (e.g., BlockLinear adapter tensors, number of block 8),
  - classifier head.
- Objective: same supervised task loss (CE/BCEWithLogits).

Interpretation: compared with head-only, Adapter FT adds trainable low-rank/block adaptation capacity in backbone pathways while still avoiding full-model fine-tuning.

## 4) Updated checkpoint comparison tables (Section 3.11.1-3.11.3)

### 4.1 Accuracy comparison

| Task group | Task | Val n_examples | Val benchmark accuracy | Val adapter FT accuracy | Val delta accuracy (adapter - benchmark) | Test n_examples | Test benchmark accuracy | Test adapter FT accuracy | Test delta accuracy (adapter - benchmark) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chexpert | chexpert | 9366 | 0.904543 | 0.909551 | +0.005008 | 9428 | 0.901228 | 0.905191 | +0.003963 |
| guo_* | guo_los | 2231 | 0.795909 | 0.851188 | +0.055279 | 2195 | 0.740488 | 0.854670 | +0.114182 |
| guo_* | guo_readmission | 2206 | 0.885621 | 0.909338 | +0.023717 | 2189 | 0.867723 | 0.902695 | +0.034972 |
| guo_* | guo_icu | 2052 | 0.956016 | 0.962476 | +0.006460 | 2037 | 0.952312 | 0.965636 | +0.013324 |
| new_* | new_hypertension | 1247 | 0.868313 | 0.890136 | +0.021823 | 1258 | 0.861232 | 0.886328 | +0.025096 |
| new_* | new_hyperlipidemia | 1441 | 0.867951 | 0.879944 | +0.011993 | 1317 | 0.873119 | 0.880030 | +0.006911 |
| new_* | new_pancan | 2215 | 0.971256 | 0.982393 | +0.011137 | 2220 | 0.951448 | 0.981982 | +0.030534 |
| new_* | new_acutemi | 2176 | 0.938149 | 0.933824 | -0.004325 | 2127 | 0.934163 | 0.934650 | +0.000487 |
| new_* | new_celiac | 2284 | 0.970270 | 0.995622 | +0.025352 | 2222 | 0.962302 | 0.991899 | +0.029597 |
| new_* | new_lupus | 2225 | 0.986143 | 0.985169 | -0.000974 | 2243 | 0.976971 | 0.991529 | +0.014558 |
| lab_* | lab_thrombocytopenia | 54504 | 0.823022 | nan | nan | 56338 | 0.725490 | nan | nan |
| lab_* | lab_hyperkalemia | 60168 | 0.977233 | nan | nan | 63653 | 0.976120 | nan | nan |
| lab_* | lab_hyponatremia | 64473 | 0.768584 | nan | nan | 67028 | 0.714505 | nan | nan |
| lab_* | lab_anemia | 56224 | 0.901548 | nan | nan | 58155 | 0.778550 | nan | nan |
| lab_* | lab_hypoglycemia | 95488 | 0.985316 | nan | nan | 100568 | 0.985586 | 0.986397 | +0.000811 |

### 4.2 AUROC comparison

| Task group | Task | Val n_examples | Val benchmark AUROC | Val adapter FT AUROC | Val delta AUROC (adapter - benchmark) | Test n_examples | Test benchmark AUROC | Test adapter FT AUROC | Test delta AUROC (adapter - benchmark) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chexpert | chexpert | 9366 | 0.722400 | 0.753598 | +0.031198 | 9428 | 0.713152 | 0.752018 | +0.038866 |
| guo_* | guo_los | 2231 | 0.809493 | 0.898029 | +0.088536 | 2195 | 0.694267 | 0.905016 | +0.210749 |
| guo_* | guo_readmission | 2206 | 0.796393 | 0.861217 | +0.064824 | 2189 | 0.689890 | 0.868545 | +0.178655 |
| guo_* | guo_icu | 2052 | 0.888600 | 0.945741 | +0.057141 | 2037 | 0.727001 | 0.923367 | +0.196366 |
| new_* | new_hypertension | 1247 | 0.816378 | 0.896732 | +0.080354 | 1258 | 0.676761 | 0.851609 | +0.174848 |
| new_* | new_hyperlipidemia | 1441 | 0.674862 | 0.840704 | +0.165842 | 1317 | 0.617191 | 0.835031 | +0.217840 |
| new_* | new_pancan | 2215 | 0.856023 | 0.937802 | +0.081779 | 2220 | 0.618486 | 0.927367 | +0.308881 |
| new_* | new_acutemi | 2176 | 0.728798 | 0.877071 | +0.148273 | 2127 | 0.643679 | 0.877147 | +0.233468 |
| new_* | new_celiac | 2284 | 0.517511 | 0.936328 | +0.418817 | 2222 | 0.474157 | 0.975444 | +0.501287 |
| new_* | new_lupus | 2225 | 0.674835 | 0.945891 | +0.271056 | 2243 | 0.544928 | 0.943500 | +0.398572 |
| lab_* | lab_thrombocytopenia | 54504 | 0.866109 | nan | nan | 56338 | 0.722924 | nan | nan |
| lab_* | lab_hyperkalemia | 60168 | 0.772773 | nan | nan | 63653 | 0.673188 | nan | nan |
| lab_* | lab_hyponatremia | 64473 | 0.762735 | nan | nan | 67028 | 0.660048 | nan | nan |
| lab_* | lab_anemia | 56224 | 0.961882 | nan | nan | 58155 | 0.788758 | nan | nan |
| lab_* | lab_hypoglycemia | 95488 | 0.798876 | nan | nan | 100568 | 0.676055 | 0.779527 | +0.103472 |

### 4.3 AUPRC comparison

| Task group | Task | Val n_examples | Val benchmark AUPRC | Val adapter FT AUPRC | Val delta AUPRC (adapter - benchmark) | Test n_examples | Test benchmark AUPRC | Test adapter FT AUPRC | Test delta AUPRC (adapter - benchmark) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chexpert | chexpert | 9366 | 0.209561 | 0.252176 | +0.042615 | 9428 | 0.215457 | 0.251196 | +0.035739 |
| guo_* | guo_los | 2231 | 0.544947 | 0.742150 | +0.197203 | 2195 | 0.577472 | 0.775330 | +0.197858 |
| guo_* | guo_readmission | 2206 | 0.446972 | 0.616610 | +0.169638 | 2189 | 0.413115 | 0.582707 | +0.169592 |
| guo_* | guo_icu | 2052 | 0.325635 | 0.555514 | +0.229879 | 2037 | 0.315017 | 0.542306 | +0.227290 |
| new_* | new_hypertension | 1247 | 0.461576 | 0.650527 | +0.188951 | 1258 | 0.253742 | 0.512725 | +0.258983 |
| new_* | new_hyperlipidemia | 1441 | 0.243783 | 0.552587 | +0.308804 | 1317 | 0.216421 | 0.500021 | +0.283600 |
| new_* | new_pancan | 2215 | 0.183032 | 0.656615 | +0.473583 | 2220 | 0.220871 | 0.644522 | +0.423652 |
| new_* | new_acutemi | 2176 | 0.153826 | 0.469514 | +0.315688 | 2127 | 0.183275 | 0.532277 | +0.349001 |
| new_* | new_celiac | 2284 | 0.006175 | 0.543099 | +0.536924 | 2222 | 0.017657 | 0.710860 | +0.693203 |
| new_* | new_lupus | 2225 | 0.069690 | 0.587575 | +0.517885 | 2243 | 0.028202 | 0.588055 | +0.559853 |
| lab_* | lab_thrombocytopenia | 54504 | nan | nan | nan | 56338 | nan | nan | nan |
| lab_* | lab_hyperkalemia | 60168 | nan | nan | nan | 63653 | nan | nan | nan |
| lab_* | lab_hyponatremia | 64473 | nan | nan | nan | 67028 | nan | nan | nan |
| lab_* | lab_anemia | 56224 | nan | nan | nan | 58155 | nan | nan | nan |
| lab_* | lab_hypoglycemia | 95488 | nan | nan | nan | 100568 | 0.082184 | 0.268363 | +0.186179 |

Comment: update date = 2026-05-02 (aligned with report Section 3.11.1-3.11.3).
Average uplift summary (computed from non-`nan` tasks only):
- Accuracy: val average uplift = `+0.015547` (average percentage uplift `+1.79%`, n=10); test average uplift = `+0.024949` (average percentage uplift `+2.99%`, n=11).
- AUROC: val average uplift = `+0.140782` (average percentage uplift `+21.52%`, n=10); test average uplift = `+0.233000` (average percentage uplift `+39.11%`, n=11).
- AUPRC: val average uplift = `+0.298117` (average percentage uplift `+1023.49%`, n=10); test average uplift = `+0.307723` (average percentage uplift `+628.82%`, n=11).

## 5) Best train/val loss table (all tasks)

Loss summary aligned to updated runs.  
- `best_train_loss`: minimum logged training loss in run history (`loss`)  
- `best_val_loss`: minimum validation loss (`eval_loss`)  

If a task has no available value yet, it is shown as `nan`.

| Task group | Task | best_train_loss | best_val_loss |
|---|---|---:|---:|
| chexpert | chexpert | 0.186400 | 0.233077 |
| guo_* | guo_los | 0.226600 | 0.330727 |
| guo_* | guo_readmission | 0.108800 | 0.266855 |
| guo_* | guo_icu | 0.005100 | 0.102040 |
| new_* | new_hypertension | 0.120900 | 0.265241 |
| new_* | new_hyperlipidemia | 0.131200 | 0.292368 |
| new_* | new_pancan | 0.002700 | 0.082110 |
| new_* | new_acutemi | 0.016700 | 0.204943 |
| new_* | new_celiac | 0.000900 | 0.046117 |
| new_* | new_lupus | 0.001300 | 0.067478 |
| lab_* | lab_thrombocytopenia | nan | nan |
| lab_* | lab_hyperkalemia | nan | nan |
| lab_* | lab_hyponatremia | nan | nan |
| lab_* | lab_anemia | nan | nan |
| lab_* | lab_hypoglycemia | nan | nan |


