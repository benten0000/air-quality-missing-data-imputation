# priprava_podatkov

## Struktura
- `data/raw/`: vhodni CSV-ji po postajah (pri tebi symlink na `combined/`)
- `data/processed/splits/`: train/val/test CSV-ji + `windows.npz` po postajah
- `data/processed/scalers/`: scalerji po postajah
- `artifacts/models/`: checkpointi modelov po tipu in postaji
- `reports/`: porocila in grafi
- `notebooks/`: analiticni notebooki
- `src/air_quality_imputer/models/`: implementacije modelov
- `src/air_quality_imputer/training/`: skupen train/eval pipeline + registry

## Modularni model setup
Model se izbira prek Hydra config group `model`:
- `model=classic_transformer`
- `model=diffusion_transformer`

Model-specifični parametri:
- `src/air_quality_imputer/training/conf/model/classic_transformer.yaml`
- `src/air_quality_imputer/training/conf/model/diffusion_transformer.yaml`

Dodajanje novega modela:
1. Dodas model datoteko v `src/air_quality_imputer/models/`
2. Dodas vnos v `src/air_quality_imputer/training/model_registry.py`
3. Dodas `conf/model/<new_model>.yaml`

## Trening
Kanonični entrypoint:
- `src/air_quality_imputer/training/train_imputer.py`

Zagon:

```bash
aqi-train
```

Alternativno (brez install):
```bash
PYTHONPATH=src python -m air_quality_imputer.training.train_imputer
```

Več modelov hkrati:

```bash
aqi-train \
  experiment.models=[classic_transformer,diffusion_transformer]
```

## Evaluacija
Kanonični entrypoint:
- `src/air_quality_imputer/training/evaluate_imputer.py`

Rezultati:
- `reports/model_eval/<model_type>/test_metrics_overall.csv`
- `reports/model_eval/<model_type>/test_metrics_by_feature.csv`

Zagon:

```bash
aqi-eval
```

Alternativno (brez install):
```bash
PYTHONPATH=src python -m air_quality_imputer.training.evaluate_imputer
```

Več modelov hkrati:

```bash
aqi-eval \
  experiment.models=[classic_transformer,diffusion_transformer]
```

## Backward compatibility
Stari entrypointi se še vedno podpirajo kot wrapperji:
- `air_quality_imputer.training.train_transformer_imputer`
- `air_quality_imputer.training.evaluate_transformer_imputer`

Opomba: CUDA je za trening/evaluacijo bistveno hitrejsa.
