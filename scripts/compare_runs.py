from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def summarize_class_report(report_csv: Path) -> Dict[str, Any]:
    """
    Expects sklearn classification_report saved as CSV (transpose of report dict).
    We will extract macro avg, weighted avg, and identify best/worst class by f1-score.
    """
    df = pd.read_csv(report_csv, index_col=0)

    out: Dict[str, Any] = {}

    # macro/weighted if available
    if "macro avg" in df.index:
        out["macro_f1_report"] = float(df.loc["macro avg", "f1-score"])
        out["macro_recall_report"] = float(df.loc["macro avg", "recall"])
        out["macro_precision_report"] = float(df.loc["macro avg", "precision"])
    else:
        out["macro_f1_report"] = None

    if "weighted avg" in df.index:
        out["weighted_f1_report"] = float(df.loc["weighted avg", "f1-score"])
    else:
        out["weighted_f1_report"] = None

    # identify classes: exclude summary rows if present
    exclude = {"accuracy", "macro avg", "weighted avg"}
    class_rows = [idx for idx in df.index if idx not in exclude]

    if class_rows:
        f1s = df.loc[class_rows, "f1-score"].astype(float)
        out["best_class"] = str(f1s.idxmax())
        out["best_class_f1"] = float(f1s.max())
        out["worst_class"] = str(f1s.idxmin())
        out["worst_class_f1"] = float(f1s.min())
    else:
        out["best_class"] = None
        out["worst_class"] = None

    return out


def find_run_dirs(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    return sorted([p for p in models_dir.iterdir() if p.is_dir() and (p / "config.json").exists()])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare trained runs using reports/*/metrics.json")
    ap.add_argument("--models-dir", default="models", help="Directory containing run folders")
    ap.add_argument("--reports-dir", default="reports", help="Directory containing evaluation outputs per run")
    ap.add_argument("--out", default="reports/model_comparison.csv", help="CSV output path")
    ap.add_argument("--topk", type=int, default=10, help="Show top-k runs by F1 macro")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    out_path = Path(args.out)

    rows = []
    for run_dir in find_run_dirs(models_dir):
        run_id = run_dir.name

        cfg_path = run_dir / "config.json"
        labels_path = run_dir / "labels.json"

        cfg = load_json(cfg_path)
        arch = str(safe_get(cfg, "arch", "unknown"))
        img_size = safe_get(cfg, "img_size", None)
        batch_size = safe_get(cfg, "batch_size", None)
        epochs = safe_get(cfg, "epochs", None)
        lr = safe_get(cfg, "lr", None)

        num_classes: Optional[int] = None
        if labels_path.exists():
            labels = load_json(labels_path)
            # keys are indices
            num_classes = len(labels)

        # Evaluation artifacts
        metrics_path = reports_dir / run_id / "metrics.json"
        report_csv = reports_dir / run_id / "classification_report.csv"
        cm_png = reports_dir / run_id / "confusion_matrix.png"

        if not metrics_path.exists():
            # If you haven't evaluated this run, skip it (or include as NaN)
            continue

        metrics = load_json(metrics_path)
        acc = safe_get(metrics, "accuracy", None)

        # Suporta ambos formatos:
        # - antigo: f1_macro
        # - novo: f1_macro_present / f1_macro_all
        f1_macro_all = safe_get(metrics, "f1_macro_all", None)
        f1_macro_present = safe_get(metrics, "f1_macro_present", None)
        f1_macro_legacy = safe_get(metrics, "f1_macro", None)

        # Regra: preferir f1_macro_all; se não existir, usar present; se não, legacy
        f1_for_rank = (
            f1_macro_all
            if f1_macro_all is not None
            else (f1_macro_present if f1_macro_present is not None else f1_macro_legacy)
        )

        rep_summary = {}
        if report_csv.exists():
            rep_summary = summarize_class_report(report_csv)

        rows.append(
            {
                "run_id": run_id,
                "arch": arch,
                "num_classes": num_classes,
                "img_size": img_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "test_accuracy": acc,
                "test_f1_for_rank": f1_for_rank,
                "test_f1_macro_all": f1_macro_all,
                "test_f1_macro_present": f1_macro_present,
                "test_f1_macro_legacy": f1_macro_legacy,
                "macro_f1_report": rep_summary.get("macro_f1_report"),
                "weighted_f1_report": rep_summary.get("weighted_f1_report"),
                "best_class": rep_summary.get("best_class"),
                "best_class_f1": rep_summary.get("best_class_f1"),
                "worst_class": rep_summary.get("worst_class"),
                "worst_class_f1": rep_summary.get("worst_class_f1"),
                "has_confusion_matrix_png": cm_png.exists(),
                "model_dir": str(run_dir.as_posix()),
                "report_dir": str((reports_dir / run_id).as_posix()),
            }
        )

    if not rows:
        raise SystemExit(
            "No evaluated runs found. Ensure you have:\n"
            "1) models/<run_id>/config.json\n"
            "2) reports/<run_id>/metrics.json (run scripts/evaluate.py)\n"
        )

    df = pd.DataFrame(rows)

    # Sort: best F1 macro (primary), then accuracy
    df_sorted = df.sort_values(["test_f1_for_rank", "test_accuracy"], ascending=[False, False]).reset_index(drop=True)


    # Save CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(out_path, index=False, encoding="utf-8")

    # Print summary
    best = df_sorted.iloc[0]
    print("\n=== Model comparison (sorted by test_f1_macro, then test_accuracy) ===")
    print(df_sorted.head(args.topk)[["run_id","arch","test_f1_for_rank","test_accuracy","test_f1_macro_all","test_f1_macro_present","test_f1_macro_legacy","worst_class","worst_class_f1"]].to_string(index=False))

    print("\n=== Recommended model (highest test_f1_for_rank) ===")

    print(f"run_id: {best['run_id']}")
    print(f"arch: {best['arch']}")
    print(f"test_f1_for_rank: {best['test_f1_for_rank']}")
    print(f"test_f1_macro_all: {best.get('test_f1_macro_all')}")
    print(f"test_f1_macro_present: {best.get('test_f1_macro_present')}")
    print(f"test_f1_macro_legacy: {best.get('test_f1_macro_legacy')}")

    print(f"test_accuracy: {best['test_accuracy']}")
    print(f"worst_class: {best.get('worst_class')} (f1={best.get('worst_class_f1')})")
    print(f"report_dir: {best['report_dir']}")
    print(f"model_dir: {best['model_dir']}")
    print(f"\nSaved full comparison CSV to: {out_path}")


if __name__ == "__main__":
    main()
