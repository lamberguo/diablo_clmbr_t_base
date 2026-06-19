# diablo_clmbr_t_base

Self-contained inference project for CLMBR + Diablo adapters.

User flow:
1) prepare dataset
2) set prediction time
3) predict task value/label

## Project Structure

- `inference/predict_next_value.py`: main inference script
- `diablo/`: local BlockLinear implementation
- `weights/adapters/<task>_b8/`: task-specific adapter weights (sync from FT via `bash_scripts/run_sync_adapters_from_ft.sh`)
- `weights/adapters/manifest.json`: last sync time and source `best_checkpoint` paths
- `dataset/`: user input data and templates
- `dataset_pt/`: optional `.pt` cache output
- `data_preprocessing/make_dataset_parquet.py`: CSV -> `dataset/data/data.parquet`
- `data_preprocessing/build_patient_cache.py`: parquet -> `.pt`
- `bash_scripts/`: one-command shell entrypoints

## Install

```bash
pip install -r requirements.txt
```

## Adapter Files

Expected files per task folder:

- `weights/adapters/<task>_b8/adapter_config.pt`
- `weights/adapters/<task>_b8/adapter_model.bin`
- `weights/adapters/<task>_b8/classifier_head.bin`

Adapter weights (~22 MB per task) are **included in this repo** under `weights/adapters/`. Large `.bin` files are tracked with **Git LFS** (see below).

### Publish / clone from GitHub

Install [Git LFS](https://git-lfs.com/) before clone or push:

```bash
git lfs install
git clone <your-repo-url>
cd diablo_clmbr_t_base
git lfs pull   # if weights look missing after clone
```

When pushing a new repo:

```bash
git lfs install
git add .gitattributes weights/
git commit -m "Add adapter weights"
git push
```

### Sync adapter weights from FT training (maintainers)

After FT pipeline selects `best_checkpoint` under `FT/checkpoint/ft/<task>_b8/`, copy them into this repo:

```bash
bash bash_scripts/run_sync_adapters_from_ft.sh
```

Optional override:

```bash
FT_CHECKPOINT_ROOT=/path/to/FT/checkpoint/ft bash bash_scripts/run_sync_adapters_from_ft.sh
```

Tasks without a finished `best_checkpoint` (e.g. `lab_hypoglycemia` while training) are skipped; copy the trainer-best step manually or re-run sync after the FT pipeline completes.

## Data Preprocessing

### Step 0: CSV -> Parquet

Start template:
- `dataset/raw_events_template.csv`

Run:

```bash
bash bash_scripts/run_make_dataset_parquet.sh
```

### Step 1: Parquet -> PT (optional cache)

Run:

```bash
bash bash_scripts/run_build_patient_cache.sh
```

## Inference at Prediction Time

List available tasks:

```bash
bash bash_scripts/run_list_tasks.sh
```

Describe one task (label names, problem type):

```bash
bash bash_scripts/run_describe_task.sh guo_los
```

Run prediction:

```bash
bash bash_scripts/run_predict_next_value.sh
```

Edit variables in `bash_scripts/run_predict_next_value.sh`:
- `TASK`
- `PATIENT_JSON`
- `PREDICTION_TIME`
- optional `HF_TOKEN`

Patient JSON template:
- `dataset/patient_template.json`

## Output Format

Prediction output JSON includes:
- `task`
- `patient_id`
- `prediction_time`
- `num_events_used`
- `task_head_prediction`

`task_head_prediction` comes from task-specific `classifier_head` in `weights/adapters/<task>_b8/`:
- single-label tasks: `predicted_label` + class probabilities + `positive_probability` (binary tasks)
- multi-label tasks: `predicted_labels` + per-label probabilities + `predicted_findings` (CheXpert label names)
- lab tasks (`lab_*`): 4-class severity + `abnormal_probability` (= `1 - P(normal)`) and `predicted_class_name`

## License (this repository)

The **code and documentation** in this repository (excluding third-party pretrained weights and any material governed by upstream licenses) are released under the **MIT License** — see [`LICENSE`](LICENSE).

The pretrained **CLMBR-T-base** weights and related obligations are **not** MIT-licensed; they are covered by Stanford / Hugging Face terms as summarized under [Third-party: CLMBR-T-Base](#third-party-clmbr-t-base) below.

## Third-party: CLMBR-T-Base

Inference uses the pretrained model **`StanfordShahLab/clmbr-t-base`** ([Hugging Face model card](https://huggingface.co/StanfordShahLab/clmbr-t-base)).

Upstream terms (see the model card for the authoritative wording):

- **Model license:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Hugging Face / EHRSHOT:** access to weights may require accepting the **EHRSHOT Credentialed Health Data License** — [full license text](https://shahlab.stanford.edu/ehrshot_license) — and any CITI / credentialing steps described on the model page.

Your obligations when using or redistributing those weights follow Stanford’s and Hugging Face’s terms, not this repository alone.

## Citation

If you use CLMBR-T-base or EHRSHOT, cite (at minimum):

```bibtex
@article{wornow2023ehrshot,
  title={EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models},
  author={Michael Wornow and Rahul Thapa and Ethan Steinberg and Jason Fries and Nigam Shah},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

Additional BibTeX entries may appear on the Hugging Face model card.

## Upstream intended use

Per the CLMBR-T-base model card: **research use**; not for real-world clinical or hospital operations without appropriate validation and governance. See the model card for bias, risks, and limitations.

