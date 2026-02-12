from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from air_quality_imputer.pipeline.common import (
    build_parser,
    derive_seed,
    get_model_cfg_from_params,
    load_params,
    to_plain_dict,
)
from air_quality_imputer.tracking.mlflow_utils import MLflowTracker
from air_quality_imputer.training.data_utils import mask_windows, mask_windows_block_feature, mask_windows_blocks
from air_quality_imputer.training.model_registry import build_model_from_checkpoint


SUPPORTED_MODELS = {"transformer", "saits"}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return mae, rmse


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_np = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights_np = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values_np) & np.isfinite(weights_np) & (weights_np > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values_np[valid], weights=weights_np[valid]))


def build_eval_summaries(
    per_model_overall: dict[str, pd.DataFrame],
    per_model_feature: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_overall_rows: list[dict[str, Any]] = []
    summary_feature_rows: list[dict[str, Any]] = []

    for model_type, overall_df in per_model_overall.items():
        if overall_df.empty:
            summary_overall_rows.append(
                {
                    "model_type": model_type,
                    "n_stations": 0,
                    "n_eval": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                }
            )
            continue

        n_eval = int(pd.to_numeric(overall_df["n_eval"], errors="coerce").fillna(0).sum())
        summary_overall_rows.append(
            {
                "model_type": model_type,
                "n_stations": int(overall_df["station"].nunique()),
                "n_eval": n_eval,
                "mae": weighted_mean(overall_df["mae"], overall_df["n_eval"]),
                "rmse": weighted_mean(overall_df["rmse"], overall_df["n_eval"]),
            }
        )

    for model_type, feature_df in per_model_feature.items():
        if feature_df.empty:
            continue
        for feature, group in feature_df.groupby("feature"):
            group_weights = cast(pd.Series, group["n_eval"])
            summary_feature_rows.append(
                {
                    "model_type": model_type,
                    "feature": str(feature),
                    "n_eval": int(pd.to_numeric(group_weights, errors="coerce").fillna(0).sum()),
                    "mae": weighted_mean(cast(pd.Series, group["mae"]), group_weights),
                    "rmse": weighted_mean(cast(pd.Series, group["rmse"]), group_weights),
                }
            )

    summary_overall_df = pd.DataFrame(summary_overall_rows)
    if not summary_overall_df.empty:
        summary_overall_df = summary_overall_df.sort_values(["mae", "rmse"], na_position="last").reset_index(drop=True)

    summary_feature_df = pd.DataFrame(summary_feature_rows)
    if not summary_feature_df.empty:
        summary_feature_df = summary_feature_df.sort_values(["feature", "mae"], na_position="last").reset_index(drop=True)

    return summary_overall_df, summary_feature_df


def save_eval_plots(summary_overall_df: pd.DataFrame, summary_feature_df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    if summary_overall_df.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib, skipping eval plots: {exc}")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    ordered = summary_overall_df.sort_values("mae", na_position="last").reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ordered["mae"], width=width, label="MAE")
    ax.bar(x + width / 2, ordered["rmse"], width=width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["model_type"], rotation=25, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison (Weighted by n_eval)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    overall_plot_path = plots_dir / "overall_metrics.png"
    fig.savefig(overall_plot_path, dpi=160)
    plt.close(fig)
    created_paths.append(overall_plot_path)

    if not summary_feature_df.empty:
        pivot = summary_feature_df.pivot(index="feature", columns="model_type", values="mae")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(pivot.columns)), max(4, 0.45 * len(pivot.index))))
            mat = pivot.to_numpy(dtype=float)
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("MAE by Feature and Model")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("MAE")
            fig.tight_layout()
            heatmap_path = plots_dir / "feature_mae_heatmap.png"
            fig.savefig(heatmap_path, dpi=160)
            plt.close(fig)
            created_paths.append(heatmap_path)

    return created_paths


def evaluate_station(
    station: str,
    model_type: str,
    checkpoint_name: str,
    models_dir: Path,
    processed_dir: Path,
    missing_rate: float,
    seed: int,
    mask_mode: str,
    block_min_len: int,
    block_max_len: int,
    block_missing_prob: float | None,
    feature_block_prob: float,
    block_no_overlap: bool,
    never_mask_features: list[str] | None,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_path = models_dir / model_type / station / checkpoint_name
    windows_path = processed_dir / station / "windows.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing windows dataset: {windows_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_model_type = str(checkpoint["model_type"])
    config = checkpoint["config_dict"]
    features = checkpoint["features"]
    never_mask_features_set = set(never_mask_features or [])
    never_mask_feature_indices = [index for index, feature in enumerate(features) if feature in never_mask_features_set]

    model = build_model_from_checkpoint(ckpt_model_type, config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    windows = np.load(windows_path)
    X_test_ori = windows["X_test_ori"].astype(np.float32)
    S_test = windows["S_test"].astype(np.int64) if "S_test" in windows.files else None
    if mask_mode == "block":
        X_test_masked = mask_windows_blocks(
            X_test_ori,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
        )
    elif mask_mode == "block_feature":
        X_test_masked = mask_windows_block_feature(
            X_test_ori,
            missing_rate=missing_rate,
            seed=seed,
            min_block_len=block_min_len,
            max_block_len=block_max_len,
            block_missing_prob=block_missing_prob,
            feature_missing_prob=feature_block_prob,
            no_overlap=block_no_overlap,
            never_mask_feature_indices=never_mask_feature_indices,
        )
    elif mask_mode == "random":
        X_test_masked = mask_windows(X_test_ori, missing_rate=missing_rate, seed=seed)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")
    if never_mask_feature_indices:
        valid_indices = [idx for idx in never_mask_feature_indices if 0 <= idx < X_test_masked.shape[2]]
        if valid_indices:
            X_test_masked = X_test_masked.copy()
            X_test_masked[:, :, valid_indices] = X_test_ori[:, :, valid_indices]

    eval_mask = np.isnan(X_test_masked) & ~np.isnan(X_test_ori)
    if not eval_mask.any():
        raise ValueError(f"No masked values for station {station}; increase missing_rate")

    impute_dataset: dict[str, Any] = {"X": X_test_masked}
    if S_test is not None:
        impute_dataset["station_ids"] = S_test
    X_imputed = model.impute(impute_dataset)
    if X_imputed is None:
        raise RuntimeError(f"Imputation failed for station {station}")

    overall_mae, overall_rmse = compute_metrics(X_test_ori[eval_mask], X_imputed[eval_mask])
    per_feature_rows: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        feature_mask = eval_mask[:, :, index]
        if feature_mask.any():
            mae, rmse = compute_metrics(X_test_ori[:, :, index][feature_mask], X_imputed[:, :, index][feature_mask])
            n_eval = int(feature_mask.sum())
        else:
            mae, rmse, n_eval = np.nan, np.nan, 0
        per_feature_rows.append(
            {
                "station": station,
                "feature": feature,
                "n_eval": n_eval,
                "mae": mae,
                "rmse": rmse,
            }
        )

    overall_row = {
        "station": station,
        "n_eval": int(eval_mask.sum()),
        "mae": overall_mae,
        "rmse": overall_rmse,
    }
    return overall_row, per_feature_rows


def _to_metrics_json(summary_overall_df: pd.DataFrame) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for _, row in summary_overall_df.iterrows():
        models[str(row["model_type"])] = {
            "mae": float(row["mae"]) if pd.notna(row["mae"]) else None,
            "rmse": float(row["rmse"]) if pd.notna(row["rmse"]) else None,
            "n_eval": int(row["n_eval"]) if pd.notna(row["n_eval"]) else 0,
            "n_stations": int(row["n_stations"]) if pd.notna(row["n_stations"]) else 0,
        }
    best_model = None
    if not summary_overall_df.empty:
        valid = summary_overall_df[pd.to_numeric(summary_overall_df["mae"], errors="coerce").notna()]
        if not valid.empty:
            best_model = str(valid.sort_values("mae").iloc[0]["model_type"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model_by_mae": best_model,
        "models": models,
    }


def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir = Path(cfg.paths.models_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    reports_root = Path(cfg.paths.reports_dir)
    plots_dir = Path(cfg.paths.plots_dir)
    metrics_dir = Path(cfg.paths.metrics_dir)

    stations = list(cfg.experiment.stations)
    model_names = list(cfg.experiment.models)
    unsupported = sorted(set(model_names) - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unsupported models for V1 pipeline: {unsupported}")

    missing_rate = float(cfg.experiment.missing_rate)
    base_seed = int(cfg.experiment.seed)
    mask_mode = str(cfg.experiment.mask_mode)
    block_min_len = int(cfg.experiment.block_min_len)
    block_max_len = int(cfg.experiment.block_max_len)
    block_missing_prob_raw = cfg.experiment.block_missing_prob
    block_missing_prob = None if block_missing_prob_raw is None else float(block_missing_prob_raw)
    feature_block_prob = float(cfg.experiment.feature_block_prob)
    block_no_overlap = bool(cfg.experiment.block_no_overlap)
    never_mask_features = list(cfg.experiment.never_mask_features)

    tracker = MLflowTracker(to_plain_dict(cfg.get("tracking")))

    per_model_overall: dict[str, pd.DataFrame] = {}
    per_model_feature: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        model_cfg = get_model_cfg_from_params(cfg, model_name)
        model_type = str(model_cfg.type)
        checkpoint_name = str(model_cfg.checkpoint_name)
        reports_dir = reports_root / model_type
        station_reports_dir = reports_dir / "stations"
        reports_dir.mkdir(parents=True, exist_ok=True)
        station_reports_dir.mkdir(parents=True, exist_ok=True)

        overall_rows: list[dict[str, Any]] = []
        per_feature_rows: list[dict[str, Any]] = []

        for station in stations:
            run_seed = derive_seed(base_seed, station=station, model_name="eval")
            overall_row, feature_rows = evaluate_station(
                station=station,
                model_type=model_type,
                checkpoint_name=checkpoint_name,
                models_dir=models_dir,
                processed_dir=processed_dir,
                missing_rate=missing_rate,
                seed=run_seed,
                mask_mode=mask_mode,
                block_min_len=block_min_len,
                block_max_len=block_max_len,
                block_missing_prob=block_missing_prob,
                feature_block_prob=feature_block_prob,
                block_no_overlap=block_no_overlap,
                never_mask_features=never_mask_features,
                device=device,
            )
            overall_rows.append(overall_row)
            per_feature_rows.extend(feature_rows)

            station_overall_path = station_reports_dir / f"{station}_overall.csv"
            station_feature_path = station_reports_dir / f"{station}_by_feature.csv"
            pd.DataFrame([overall_row]).to_csv(station_overall_path, index=False)
            pd.DataFrame(feature_rows).to_csv(station_feature_path, index=False)

            run_name = f"eval/{model_name}/{station}/seed-{run_seed}"
            tags = {
                "stage": "eval",
                "model": model_name,
                "station": station,
                "seed": run_seed,
                "mask_mode": mask_mode,
            }
            feature_metrics: dict[str, Any] = {}
            for row in feature_rows:
                feature = str(row["feature"])
                feature_metrics[f"eval.feature.{feature}.mae"] = row["mae"]
                feature_metrics[f"eval.feature.{feature}.rmse"] = row["rmse"]

            with tracker.start_run(run_name=run_name, tags=tags):
                tracker.log_input_dataset(
                    name=f"prepared-{station}",
                    source=str(processed_dir / station / "windows.npz"),
                    context="evaluation",
                    preview={
                        "station": station,
                        "model": model_name,
                        "mask_mode": mask_mode,
                    },
                )
                tracker.log_params(to_plain_dict(cfg.experiment), prefix="experiment")
                tracker.log_params(to_plain_dict(model_cfg), prefix=f"models.{model_name}")
                tracker.log_metrics(
                    {
                        "eval.mae": overall_row["mae"],
                        "eval.rmse": overall_row["rmse"],
                        "eval.n_eval": overall_row["n_eval"],
                    }
                )
                tracker.log_metrics(feature_metrics)
                tracker.log_artifact(station_overall_path, artifact_path=f"evaluation/{station}")
                tracker.log_artifact(station_feature_path, artifact_path=f"evaluation/{station}")

            print(
                f"[eval] {model_name}/{station}: "
                f"MAE={overall_row['mae']:.6f}, RMSE={overall_row['rmse']:.6f}, n_eval={overall_row['n_eval']}"
            )

        overall_df = pd.DataFrame(overall_rows)
        per_feature_df = pd.DataFrame(per_feature_rows)
        overall_path = reports_dir / "test_metrics_overall.csv"
        per_feature_path = reports_dir / "test_metrics_by_feature.csv"
        overall_df.to_csv(overall_path, index=False)
        per_feature_df.to_csv(per_feature_path, index=False)
        per_model_overall[model_type] = overall_df
        per_model_feature[model_type] = per_feature_df

    summary_overall_df, summary_feature_df = build_eval_summaries(per_model_overall, per_model_feature)
    summary_overall_path = reports_root / "summary_overall.csv"
    summary_feature_path = reports_root / "summary_by_feature.csv"
    summary_overall_df.to_csv(summary_overall_path, index=False)
    summary_feature_df.to_csv(summary_feature_path, index=False)

    plot_paths = save_eval_plots(summary_overall_df, summary_feature_df, plots_dir=plots_dir)

    metrics_payload = _to_metrics_json(summary_overall_df)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "model_eval_metrics.json"
    plots_source_path = metrics_dir / "summary_by_feature_for_dvc_plots.csv"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    summary_feature_df.to_csv(plots_source_path, index=False)

    # Attach summary artifacts to a single compact run.
    with tracker.start_run(
        run_name=f"eval/summary/all/seed-{base_seed}",
        tags={
            "stage": "eval_summary",
            "model": "all",
            "station": "all",
            "seed": base_seed,
            "mask_mode": mask_mode,
        },
    ):
        tracker.log_artifact(summary_overall_path, artifact_path="summary/files")
        tracker.log_artifact(summary_feature_path, artifact_path="summary/files")
        tracker.log_artifact(metrics_path, artifact_path="summary/files")
        tracker.log_artifact(plots_source_path, artifact_path="summary/files")
        for plot_path in plot_paths:
            tracker.log_artifact(plot_path, artifact_path="plots")

    print(f"[eval] Saved summary: {summary_overall_path}")
    print(f"[eval] Saved summary: {summary_feature_path}")
    print(f"[eval] Saved metrics: {metrics_path}")
    print(f"[eval] Saved DVC plots source: {plots_source_path}")
    for path in plot_paths:
        print(f"[eval] Saved plot: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Evaluate imputation models and emit DVC metrics.")
    args = parser.parse_args(argv)
    cfg = load_params(args.params)
    run(cfg)


if __name__ == "__main__":
    main()
