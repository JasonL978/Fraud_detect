"""Train the XGBoost baseline on PaySim and log results.

Run (needs data/raw/paysim.csv):
    python models/train_baseline.py
    python models/train_baseline.py --csv data/raw/paysim_synth_demo.csv --no-mlflow

Outputs:
  models/artifacts/pr_curve.png      (committed)
  models/artifacts/baseline_metrics.json
  ./mlruns/...                       (MLflow run, gitignored)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from fraud_detect.baseline import BaselineParams, train_and_evaluate  # noqa: E402

_ARTIFACTS = _REPO_ROOT / "models" / "artifacts"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the XGBoost fraud baseline")
    p.add_argument("--csv", default=str(_REPO_ROOT / "data" / "raw" / "paysim.csv"))
    p.add_argument("--sample", type=int, default=None, help="train on first N rows (quick runs)")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--synthetic", action="store_true",
                   help="mark outputs as synthetic placeholder numbers")
    return p.parse_args(argv)


def save_pr_curve(result, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(result.recall, result.precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}\nDownload PaySim (see data/README.md) "
              f"or pass --csv to a synthetic file.", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path, nrows=args.sample)
    print(f"loaded {len(df):,} rows from {csv_path.name}")

    result = train_and_evaluate(df, BaselineParams())
    m = result.metrics

    print("\n=== Baseline metrics (time-split test set) ===")
    print(f"  PR-AUC              : {m['pr_auc']:.4f}")
    print(f"  recall@1%FPR        : {m['recall_at_1pct_fpr']:.4f}   <-- headline")
    print(f"  ROC-AUC (secondary) : {m['roc_auc']:.4f}")
    print(f"  base rate           : {m['base_rate']:.5f}  "
          f"(pos={m['n_pos']:,} / neg={m['n_neg']:,})")
    print(f"  scale_pos_weight    : {result.scale_pos_weight:.1f}")
    print(f"  split sizes         : {result.split_sizes}")

    tag = "SYNTHETIC PLACEHOLDER" if args.synthetic else "PaySim"
    save_pr_curve(result, _ARTIFACTS / "pr_curve.png",
                  f"Baseline PR curve ({tag}) — PR-AUC={m['pr_auc']:.3f}")

    payload = {
        "source": tag,
        "metrics": m,
        "scale_pos_weight": result.scale_pos_weight,
        "split_sizes": result.split_sizes,
        "feature_columns": result.feature_columns,
    }
    (_ARTIFACTS / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {_ARTIFACTS / 'pr_curve.png'} and baseline_metrics.json")

    if not args.no_mlflow:
        import mlflow
        import mlflow.xgboost
        mlflow.set_tracking_uri(f"file:{(_REPO_ROOT / 'mlruns').as_posix()}")
        mlflow.set_experiment("baseline")
        with mlflow.start_run(run_name=f"xgb_baseline_{tag.split()[0].lower()}"):
            mlflow.log_params({
                "scale_pos_weight": result.scale_pos_weight,
                "n_features": len(result.feature_columns),
                "n_train": result.split_sizes["train"],
                "n_test": result.split_sizes["test"],
                "source": tag,
            })
            mlflow.log_metrics({k: v for k, v in m.items()})
            mlflow.log_artifact(str(_ARTIFACTS / "pr_curve.png"))
            mlflow.log_artifact(str(_ARTIFACTS / "baseline_metrics.json"))
            mlflow.xgboost.log_model(result.model, artifact_path="model")
        print("logged MLflow run under ./mlruns (experiment: baseline)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
