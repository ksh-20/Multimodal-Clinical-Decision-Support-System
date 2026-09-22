"""
generate_metrics_report.py
==========================
One-time system performance metrics report for the Multimodal Clinical
Knowledge Layer. Run this script once after training all models to produce
a fixed, reproducible metrics report suitable for research paper inclusion.

Usage:
    python generate_metrics_report.py
    python generate_metrics_report.py --save        # also saves report to outputs/metrics_report.txt

The report reads ONLY from:
  - model_metrics.json   : training-time evaluation metrics (static)
  - kg_relationships.json: knowledge graph structure    (static)

It does NOT depend on any patient input or inference run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_METRICS_PATH = BASE_DIR / "model_metrics.json"
KG_JSON_PATH       = BASE_DIR / "kg_relationships.json"

BORDER  = "=" * 74
DIVIDER = "-" * 74


def make_bar(val: float, total_width: int = 30) -> str:
    """Creates a visual progress bar using block characters."""
    filled = int(round(val * total_width))
    filled = max(0, min(total_width, filled))
    return "\u2588" * filled + "\u2591" * (total_width - filled)


def load_model_metrics() -> dict:
    if not MODEL_METRICS_PATH.exists():
        print(f"[ERROR] model_metrics.json not found at {MODEL_METRICS_PATH}")
        sys.exit(1)
    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_kg_metrics() -> dict:
    if not KG_JSON_PATH.exists():
        return {"error": f"kg_relationships.json not found at {KG_JSON_PATH}"}
    with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])
    rules = raw.get("clinical_rules", [])

    node_type_map = {n["id"]: n.get("type", "other") for n in nodes}
    cross_modal = sum(
        1 for e in edges
        if node_type_map.get(e.get("source", ""), "") !=
           node_type_map.get(e.get("target", ""), "")
        and node_type_map.get(e.get("source", ""), "")
        and node_type_map.get(e.get("target", ""), "")
    )

    connected_ids: set[str] = set()
    for e in edges:
        connected_ids.add(e.get("source", ""))
        connected_ids.add(e.get("target", ""))
    nodes_connected = sum(1 for n in nodes if n["id"] in connected_ids)

    type_counts: dict[str, int] = {}
    for n in nodes:
        t = n.get("type", "other")
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_rules": len(rules),
        "cross_modal_edges": cross_modal,
        "nodes_connected": nodes_connected,
        "coverage_pct": round(nodes_connected / max(len(nodes), 1) * 100, 2),
        "node_type_distribution": type_counts,
    }


def format_report(models: dict, kg: dict) -> str:
    lines: list[str] = []

    lines.append(BORDER)
    lines.append("  MULTIMODAL CLINICAL AI \u2014 SYSTEM PERFORMANCE METRICS REPORT")
    lines.append("  Source: model_metrics.json + kg_relationships.json (training-time, static)")
    lines.append(BORDER)

    # ------------------------------------------------------------------ KG
    lines.append("")
    lines.append("[1] KNOWLEDGE GRAPH STRUCTURE")
    lines.append(DIVIDER)
    if "error" in kg:
        lines.append(f"  {kg['error']}")
    else:
        lines.append(f"  Total Nodes              : {kg['total_nodes']}")
        lines.append(f"  Total Edges              : {kg['total_edges']}")
        lines.append(f"  Clinical Decision Rules  : {kg['total_rules']}")
        lines.append(f"  Cross-modal Edges        : {kg['cross_modal_edges']}")
        lines.append(f"  Node Coverage            : {kg['coverage_pct']}%  ({kg['nodes_connected']}/{kg['total_nodes']} nodes connected)")
        dist = ", ".join(f"{k}={v}" for k, v in sorted(kg["node_type_distribution"].items()))
        lines.append(f"  Node Type Distribution   : {dist}")

    # ------------------------------------------------ Individual models
    lines.append("")
    lines.append("[2] INDIVIDUAL MODEL EVALUATION METRICS")
    lines.append(DIVIDER)

    model_defs = [
        {
            "key": "diabetes_model",
            "label": "Diabetes Risk Classifier",
            "keys": [
                ("Accuracy", "accuracy"),
                ("AUC-ROC", "auc_roc"),
                ("F1-Score", "f1_score"),
                ("Precision", "precision"),
                ("Recall / Sensitivity", "recall"),
                ("Specificity", "specificity"),
            ],
            "cm_key": "confusion_matrix",
        },
        {
            "key": "diabetic_retinopathy_model",
            "label": "Diabetic Retinopathy Grader",
            "keys": [
                ("Accuracy", "accuracy"),
                ("AUC-ROC (multi-class OvR)", "auc_roc_multiclass"),
                ("Quadratic Weighted Kappa", "quadratic_weighted_kappa"),
                ("F1-Macro", "f1_macro"),
                ("Precision-Macro", "precision_macro"),
                ("Recall-Macro", "recall_macro"),
            ],
            "cm_key": "confusion_matrix_5class",
        },
        {
            "key": "skin_disease_model",
            "label": "Skin Lesion Classifier",
            "keys": [
                ("Accuracy", "accuracy"),
                ("AUC-ROC (multi-class OvR)", "auc_roc_multiclass"),
                ("F1-Macro", "f1_macro"),
                ("Precision-Macro", "precision_macro"),
                ("Recall-Macro", "recall_macro"),
                ("Malignant Sensitivity", "sensitivity_malignant"),
            ],
            "cm_key": "confusion_matrix_summary",
        },
    ]

    for md in model_defs:
        data = models.get(md["key"], {})
        arch = data.get("architecture", "")
        dataset = data.get("dataset", "")
        tr_s = data.get("training_samples", "?")
        te_s = data.get("test_samples", "?")
        m = data.get("metrics", {})

        lines.append("")
        lines.append(f"  {md['label']}")
        lines.append(f"  Architecture : {arch}")
        lines.append(f"  Dataset      : {dataset}")
        lines.append(f"  Samples      : {tr_s} train / {te_s} test")
        lines.append("")

        for display_name, field in md["keys"]:
            val = m.get(field)
            if val is not None:
                bar = make_bar(val, 30)
                lines.append(f"    {display_name:<32} {bar}  {val * 100:.2f}%")

        # Per-class breakdown if available
        cm = data.get(md["cm_key"])
        if isinstance(cm, dict):
            classes = cm.get("classes", [])
            per_class_f1 = cm.get("per_class_f1", cm.get("per_class_accuracy", []))
            if classes and per_class_f1:
                lines.append("")
                lines.append("    Per-Class Breakdown:")
                for cls, val in zip(classes, per_class_f1):
                    bar = make_bar(val, 20)
                    lines.append(f"      {cls:<36} {bar}  {val * 100:.1f}%")
            # Binary CM
            if all(k in cm for k in ("true_positive", "true_negative", "false_positive", "false_negative")):
                tp = cm["true_positive"]
                tn = cm["true_negative"]
                fp = cm["false_positive"]
                fn = cm["false_negative"]
                lines.append("")
                lines.append("    Confusion Matrix (Test Set):")
                lines.append(f"      Predicted Positive | TP={tp:>4}  FP={fp:>4}")
                lines.append(f"      Predicted Negative | FN={fn:>4}  TN={tn:>4}")

        if data.get("notes"):
            lines.append(f"\n    NOTE: {data['notes']}")

        lines.append("")
        lines.append("  " + DIVIDER[2:])

    # ------------------------------------------------ Combined metrics
    auc_vals, f1_vals, acc_vals = [], [], []
    for md in model_defs:
        m = models.get(md["key"], {}).get("metrics", {})
        auc = m.get("auc_roc") or m.get("auc_roc_multiclass")
        f1 = m.get("f1_score") or m.get("f1_macro")
        acc = m.get("accuracy")
        if auc is not None:
            auc_vals.append(auc)
        if f1 is not None:
            f1_vals.append(f1)
        if acc is not None:
            acc_vals.append(acc)

    lines.append("")
    lines.append("[3] COMBINED SYSTEM METRICS  (macro-average across all sub-models)")
    lines.append(DIVIDER)
    if auc_vals:
        v = sum(auc_vals) / len(auc_vals)
        bar = make_bar(v, 30)
        lines.append(f"  System AUC-ROC (macro avg)          {bar}  {v*100:.2f}%  [{len(auc_vals)} models]")
    if f1_vals:
        v = sum(f1_vals) / len(f1_vals)
        bar = make_bar(v, 30)
        lines.append(f"  System F1-Macro  (macro avg)        {bar}  {v*100:.2f}%  [{len(f1_vals)} models]")
    if acc_vals:
        v = sum(acc_vals) / len(acc_vals)
        bar = make_bar(v, 30)
        lines.append(f"  System Accuracy  (macro avg)        {bar}  {v*100:.2f}%  [{len(acc_vals)} models]")

    sep_name = '-' * 30
    sep_val  = '-' * 10
    lines.append("")
    lines.append("  RESEARCH PAPER SUMMARY TABLE (for LaTeX / Paper inclusion)")
    lines.append("  " + DIVIDER[2:])
    lines.append(f"  {'Model / Component':<32} {'Accuracy':<12} {'AUC-ROC':<12} {'F1-Macro':<12}")
    lines.append(f"  {sep_name:<32} {sep_val:<12} {sep_val:<12} {sep_val:<12}")

    for md in model_defs:
        m = models.get(md["key"], {}).get("metrics", {})
        acc_s = f"{m.get('accuracy', 0)*100:.2f}%" if m.get('accuracy') is not None else "N/A"
        auc_raw = m.get('auc_roc') or m.get('auc_roc_multiclass')
        auc_s = f"{auc_raw*100:.2f}%" if auc_raw is not None else "N/A"
        f1_raw = m.get('f1_score') or m.get('f1_macro')
        f1_s = f"{f1_raw*100:.2f}%" if f1_raw is not None else "N/A"
        lines.append(f"  {md['label']:<32} {acc_s:<12} {auc_s:<12} {f1_s:<12}")

    lines.append(f"  {sep_name:<32} {sep_val:<12} {sep_val:<12} {sep_val:<12}")
    comb_acc = f"{(sum(acc_vals)/len(acc_vals))*100:.2f}%" if acc_vals else "N/A"
    comb_auc = f"{(sum(auc_vals)/len(auc_vals))*100:.2f}%" if auc_vals else "N/A"
    comb_f1  = f"{(sum(f1_vals)/len(f1_vals))*100:.2f}%" if f1_vals else "N/A"
    lines.append(f"  {'Overall Combined System':<32} {comb_acc:<12} {comb_auc:<12} {comb_f1:<12}")

    lines.append("")
    lines.append("  METHODOLOGY NOTE")
    lines.append("  " + DIVIDER[2:])
    lines.append("  - Individual metrics are computed on held-out test sets during offline training.")
    lines.append("  - Combined metrics represent unweighted macro-averages across all sub-models.")
    lines.append("  - AUC-ROC is the primary screening indicator (threshold-independent).")
    lines.append("  - Values are FIXED and independent of single patient inference runs.")
    lines.append("  - To update metrics: update model_metrics.json with your new test results")
    lines.append("    and re-run this script.")
    lines.append("")
    lines.append(BORDER)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one-time system metrics report for research paper."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the report to outputs/metrics_report.txt and outputs/metrics_table.csv"
    )
    args = parser.parse_args()

    models = load_model_metrics()
    kg = load_kg_metrics()
    report = format_report(models, kg)

    # Print to terminal with UTF-8
    try:
        out = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
        out.write(report)
        out.flush()
    except Exception:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    if args.save:
        out_dir = BASE_DIR / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / "metrics_report.txt"
        txt_path.write_text(report, encoding="utf-8")
        print(f"Report saved to: {txt_path}")

        # Also save as CSV for paper tables
        csv_path = out_dir / "metrics_table.csv"
        rows = [
            "Component,Accuracy,AUC-ROC,F1-Macro,Dataset,Architecture",
        ]
        for key, name in [
            ("diabetes_model", "Diabetes Risk Classifier"),
            ("diabetic_retinopathy_model", "Diabetic Retinopathy Grader"),
            ("skin_disease_model", "Skin Lesion Classifier"),
        ]:
            m = models.get(key, {})
            met = m.get("metrics", {})
            acc = met.get("accuracy", "")
            auc = met.get("auc_roc") or met.get("auc_roc_multiclass", "")
            f1 = met.get("f1_score") or met.get("f1_macro", "")
            ds = m.get("dataset", "").replace(",", ";")
            arch = m.get("architecture", "").replace(",", ";")
            rows.append(f'"{name}",{acc},{auc},{f1},"{ds}","{arch}"')

        # Combined row
        acc_v = [models[k]["metrics"]["accuracy"] for k in models if "metrics" in models[k] and "accuracy" in models[k]["metrics"]]
        auc_v = [models[k]["metrics"].get("auc_roc") or models[k]["metrics"].get("auc_roc_multiclass") for k in models if "metrics" in models[k]]
        auc_v = [x for x in auc_v if x is not None]
        f1_v = [models[k]["metrics"].get("f1_score") or models[k]["metrics"].get("f1_macro") for k in models if "metrics" in models[k]]
        f1_v = [x for x in f1_v if x is not None]

        c_acc = round(sum(acc_v)/len(acc_v), 4) if acc_v else ""
        c_auc = round(sum(auc_v)/len(auc_v), 4) if auc_v else ""
        c_f1 = round(sum(f1_v)/len(f1_v), 4) if f1_v else ""
        rows.append(f'"Overall Combined System",{c_acc},{c_auc},{c_f1},"Multimodal Composite","Ensemble + Knowledge Layer"')

        csv_path.write_text("\n".join(rows), encoding="utf-8")
        print(f"Paper table CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
