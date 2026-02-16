# priprava_podatkov

Repozitorij je postavljen kot DVC-first ML pipeline za imputacijo kakovosti zraka:

- `dvc repro` je glavni entrypoint.
- DVC verzionira `data/processed`, `artifacts/models`, `reports/*`.
- MLflow (npr. DagsHub endpoint) sledi train/eval runom in artefaktom (opcijsko).

## Pipeline

Stagi v `dvc.yaml`:

1. `prepare_data`
2. `train_models`
3. `evaluate_models`

Implementacija stage-ov:

- `src/air_quality_imputer/pipeline/prepare_data.py`
- `src/air_quality_imputer/pipeline/train_models.py`
- `src/air_quality_imputer/pipeline/evaluate_models.py`

## Parametri

Enotni vir parametrov: `configs/pipeline/params.yaml`.

Glavni bloki:

- `experiment`
- `training`
- `evaluation`
- `models`
- `paths`
- `tracking`

Priporocena delitev odgovornosti:

- `experiment`: dataset/model setup (`stations`, `models`, `features`, `seed`, windowing)
- `training.train_mask`: per-model train-time masking (`training.train_mask.transformer`, `training.train_mask.saits`)
- `training.shared_validation_mask`: skupno maskiranje za pripravo validacijskega seta (`X_val_masked`) za vse modele
- `evaluation`: maskiranje za test/eval + opcijski `mlflow_model_refs`

Opomba: SAITS trenutno podpira samo `training.train_mask.saits.mode: random` (MCAR).

## Uporaba

### 1) Namestitev

```bash
pip install -e .
```

### 1b) SAITS benchmark dataseti (arXiv:2202.08516v5)

SAITS paper uporablja 4 datasete in specifičen preprocessing (Section 4.1). Ta repo zdaj podpira enake split/window
nastavitve preko `dataset.definitions.<name>.split` in `dataset.definitions.<name>.window` v `configs/pipeline/params.yaml`.

- `PhysioNet-2012`: 37 featurejev, 48 korakov (1h bin), random split 80/20 + 20% val iz train
- `Air-Quality` (Beijing multi-site): 132 featurejev (12 site x 11), 24 korakov (1h), date-range split
- `Electricity`: 370 featurejev, 100 korakov (15-min), date-range split
- `ETT`: 7 featurejev, 24 korakov (15-min), date-range split, sliding step 12

CSV-ji so v repozitoriju pripravljeni v:

- `data/datasets/beijing_air_quality/raw/combined.csv`
- `data/datasets/electricity/raw/electricity.csv`
- `data/datasets/ett/raw/ett.csv`
- `data/datasets/physionet2012/raw/physionet2012.csv`

### 2) Lokalni zagon pipeline-a

```bash
dvc repro
```

### 3) Eksperimenti

```bash
dvc exp run -S configs/pipeline/params.yaml:training.lr=0.0005 -S configs/pipeline/params.yaml:evaluation.missing_rate=0.25
dvc exp show
```

### 4) DVC podatki/modeli poročila

```bash
dvc push
dvc pull
```

### 5) DVC metrics/plots

```bash
dvc metrics show
dvc plots show
```

### 6) Dataset switch v enem `params.yaml`

Dataset se zdaj izbira v YAML preko `experiment.dataset`.
`prepare_data` uporablja izključno CSV datasete (brez NPZ adapterja).  
Feature-ji se privzeto inferajo iz CSV in uporabijo se vsi stolpci razen `datetime`/`station`/`series_id`.

Skupna struktura map:

- `data/datasets/<dataset>/raw/`

Potem zaženi isti `params.yaml` z override-i:

```bash
dvc exp run \
  -S configs/pipeline/params.yaml:experiment.dataset=electricity \
  -S configs/pipeline/params.yaml:experiment.stations='[electricity]' \
  -S configs/pipeline/params.yaml:experiment.models='[transformer]' \
  -S configs/pipeline/params.yaml:tracking.enabled=false
```

Ali pa direktno v `configs/pipeline/params.yaml` spremeni:

```yaml
experiment:
  dataset: electricity # ali air_quality / physionet2012 / ett
```

### 7) Beijing PRSA (multi-site)

Dataset `beijing_air_quality` je v `data/datasets/beijing_air_quality/raw/combined.csv` (12 postaj v “wide” formatu).

Če želiš SAITS-style “wide” format (postaje kot del feature dimenzije), uporabi postajo `combined`:

- `experiment.dataset: beijing_air_quality`
- `experiment.stations: [combined]`
- `experiment.features: []` (samodejno razširi v `PM2.5_Aotizhongxin`, …)

Opomba: ta repo predpostavlja, da so postaje že poravnane na isti `datetime` indeks (v PRSA so).

## MLflow / DagsHub

`tracking` konfiguracija je v `configs/pipeline/params.yaml`.

Ime MLflow eksperimenta se nastavi avtomatsko iz `experiment.dataset`:

- `air` / `air_quality` -> `air-quality-imputer`
- ostalo -> `<dataset>-quality-imputer` (npr. `electricity-quality-imputer`)

Če imaš nameščen `dagshub` in nastaviš `tracking.repo_owner`/`tracking.repo_name`, tracker poskusi narediti
`dagshub.init(..., mlflow=True)`. Če `dagshub` ni na voljo, lahko MLflow nastaviš klasično (npr. preko `MLFLOW_TRACKING_URI`).

Train run naming:

- `train/<model>/<station>/seed-<seed>`

Eval run naming:

- `eval/<model>/<station>/seed-<seed>`

Eval logira `evaluation.*` nastavitve posebej, train run pa ne logira eval maskiranja.

Za eval lahko opcijsko nastaviš checkpoint direktno iz MLflow:

```yaml
evaluation:
  mlflow_model_refs:
    transformer: runs:/.../model/files/transformer.pt
    saits: runs:/.../model/files/saits.pt
```

Logirane metrike:

- `train.loss.best`
- `eval.mae`
- `eval.rmse`
- `eval.n_eval`

## DVC remote (DagsHub S3)

Primer setupa:

```bash
dvc remote add -d origin s3://<bucket-or-repo-path>
dvc remote modify origin endpointurl https://dagshub.com/api/v1/repos/<owner>/<repo>/s3
dvc remote modify --local origin access_key_id <username>
dvc remote modify --local origin secret_access_key <token>
```

Credentiali naj ostanejo v `.dvc/config.local` (ne commitaj).

## Izhodi

Generirani artefakti:

- `data/processed/splits/*`
- `artifacts/models/<model>/<station>/*.pt`
- `reports/model_eval/<model>/test_metrics_overall.csv`
- `reports/model_eval/summary_overall.csv`
- `reports/metrics/model_eval_metrics.json`

## CLI skripte

- `aqi-prepare`
- `aqi-train-models`
- `aqi-evaluate-models`
