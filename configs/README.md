# Config Layout

Aktivne YAML konfiguracije so centralizirane v `configs/`.

- `configs/pipeline/params.yaml`
  Enotni parameter source za DVC pipeline (`dvc repro`, `dvc exp run`).

- `configs/legacy/hydra/`
  Arhiv stare Hydra postavitve za referenco/migracijo.
  Ni več primarni source-of-truth za trenutno pipeline izvedbo.
