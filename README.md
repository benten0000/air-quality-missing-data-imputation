# priprava_podatkov

Repozitorij je postavljen kot DVC-first ML pipeline za imputacijo kakovosti zraka:

- `dvc repro` je glavni entrypoint.
- DVC verzionira `data/processed`, `artifacts/models`, `reports/*`.
- MLflow (npr. DagsHub endpoint) sledi train/eval runom in artefaktom.

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
- `evaluation`: maskiranje za test/eval + opcijski `model_ids`

Opomba: SAITS trenutno podpira samo `training.train_mask.saits.mode: random` (MCAR).

## Uporaba

### 1) Namestitev

```bash
pip install -e .
```

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

## MLflow / DagsHub

`tracking` konfiguracija je v `configs/pipeline/params.yaml`.

MLflow tracking je nastavljen na DagsHub-only način:

- `tracking.repo_owner`
- `tracking.repo_name`

Tracker vedno pokliče `dagshub.init(..., mlflow=True)` (brez fallbacka na klasični URI način).

Train run naming:

- `train/<model>/<station>/seed-<seed>`

Eval run naming:

- `eval/<model>/<station>/seed-<seed>`

Eval logira `evaluation.*` nastavitve posebej, train run pa ne logira eval maskiranja.

Za izbor checkpointa po ID v eval lahko nastaviš:

```yaml
evaluation:
  model_ids:
    transformer: transformer-all_stations-seed-2189050817
    saits:
      all_stations: saits-all_stations-seed-2789381912
```

Pri treningu se samodejno gradi `artifacts/models/model_index.json`, iz katerega eval poišče model po ID.

Logirane metrike:

- `train.loss.best`
- `eval.mae`
- `eval.rmse`
- `eval.n_eval`
- `eval.feature.<feature>.mae`
- `eval.feature.<feature>.rmse`

Za train rune poskusimo logirati tudi MLflow Torch model (`model/mlflow`), zato se v UI napolni stolpec **Models** tam, kjer je model uspešno logiran.

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
- `data/processed/scalers/*`
- `artifacts/models/<model>/<station>/*.pt`
- `reports/model_eval/<model>/test_metrics_overall.csv`
- `reports/model_eval/<model>/test_metrics_by_feature.csv`
- `reports/model_eval/summary_overall.csv`
- `reports/model_eval/summary_by_feature.csv`
- `reports/metrics/model_eval_metrics.json`
- `reports/plots/model_eval/*.png`

## CLI skripte

- `aqi-prepare`
- `aqi-train-models`
- `aqi-evaluate-models`
