"""Generate a Chinese Markdown report for Stage C learned repair results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _read_decision(metrics_dir: Path) -> dict[str, Any]:
    path = metrics_dir / "learned_repair_decision.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_validation_gate(metrics_dir: Path) -> dict[str, Any] | None:
    path = metrics_dir / "validation_gate_decision.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_无可用数据_"
    view = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    if view.empty:
        return "_无可用数据_"
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _per_class_decision_table(decision: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for category, record in decision.get("per_class_decisions", {}).items():
        rows.append(
            {
                "category": category,
                "expected_role": record.get("expected_role", ""),
                "verdict": record.get("verdict", ""),
                "reason": record.get("reason", ""),
                "vs_baseline_cnr_delta": record.get("vs_baseline_cnr_delta", ""),
                "vs_baseline_cnr_ci_low": record.get("vs_baseline_cnr_ci_low", ""),
                "vs_shuffled_cnr_delta": record.get("vs_shuffled_cnr_delta", ""),
                "vs_shuffled_cnr_ci_low": record.get("vs_shuffled_cnr_ci_low", ""),
                "n": record.get("n", ""),
            }
        )
    return pd.DataFrame(rows)


def _validation_gate_table(gate: dict[str, Any] | None) -> pd.DataFrame:
    if not gate or not gate.get("enabled"):
        return pd.DataFrame()
    rows = []
    for category, record in gate.get("decisions", {}).items():
        rows.append(
            {
                "category": category,
                "decision": record.get("decision", ""),
                "reason": record.get("reason", ""),
                "mean_delta": record.get("mean_delta", ""),
                "ci_low": record.get("ci_low", ""),
                "n": record.get("n", ""),
            }
        )
    return pd.DataFrame(rows)


def _recommendation(decision: dict[str, Any]) -> str:
    verdict = str(decision.get("verdict", ""))
    if verdict == "keep_learned_selective_repair":
        return "建议：保留 learned selective repair 作为下一阶段主线，但按 per-class 口径解释结果；胸腔积液承担主证据，气胸按稀疏类方向性证据单独讨论。"
    if verdict == "candidate_only_but_macro_harm":
        return "建议：只把该模块定位为候选类专用修复，不要宣称全类别通用；下一步重点分析非候选类被扰动的原因。"
    if verdict == "reject_learned_selective_repair":
        return "建议：不要进入更复杂模块，先分析 predictor 是否学到有效 region abnormality，或改用更独立的 image-conditioned evidence source。"
    return "建议：证据不足，先检查输出文件完整性、test split 统计表和 per-class bootstrap。"


def build_stage_c_markdown_report(metrics_dir: Path, output_md: Path | None = None) -> str:
    decision = _read_decision(metrics_dir)
    validation_gate = _read_validation_gate(metrics_dir)
    bootstrap = _read_csv_if_exists(metrics_dir / "bootstrap_summary.csv")
    per_class = _read_csv_if_exists(metrics_dir / "per_class_bootstrap_summary.csv")

    test_cnr = bootstrap
    if not test_cnr.empty:
        test_cnr = test_cnr[(test_cnr.get("split") == "test") & (test_cnr.get("metric") == "cnr")].copy()
    per_class_test_cnr = per_class
    if not per_class_test_cnr.empty:
        per_class_test_cnr = per_class_test_cnr[
            (per_class_test_cnr.get("split") == "test") & (per_class_test_cnr.get("metric") == "cnr")
        ].copy()
    per_class_decisions = _per_class_decision_table(decision)
    validation_gate_decisions = _validation_gate_table(validation_gate)

    lines = [
        "# Stage C Learned Repair 结果报告",
        "",
        "## 1. 总体决策",
        "",
        f"- verdict: `{decision.get('verdict', '')}`",
        f"- reason: `{decision.get('reason', '')}`",
        f"- candidate_vs_baseline_cnr_delta: `{_fmt(decision.get('candidate_vs_baseline_cnr_delta'))}`",
        f"- candidate_vs_baseline_cnr_ci_low: `{_fmt(decision.get('candidate_vs_baseline_cnr_ci_low'))}`",
        f"- specificity_vs_shuffled_cnr_delta: `{_fmt(decision.get('specificity_vs_shuffled_cnr_delta'))}`",
        f"- specificity_vs_shuffled_cnr_ci_low: `{_fmt(decision.get('specificity_vs_shuffled_cnr_ci_low'))}`",
        f"- macro_all_cnr_delta: `{_fmt(decision.get('macro_all_cnr_delta'))}`",
        f"- macro_all_cnr_ci_low: `{_fmt(decision.get('macro_all_cnr_ci_low'))}`",
        "",
        "## 2. Per-Class \u5224\u5b9a",
        "",
        _markdown_table(
            per_class_decisions,
            [
                "category",
                "expected_role",
                "verdict",
                "reason",
                "vs_baseline_cnr_delta",
                "vs_baseline_cnr_ci_low",
                "vs_shuffled_cnr_delta",
                "vs_shuffled_cnr_ci_low",
                "n",
            ],
        ),
        "",
        "## Validation-Gated DCEM",
        "",
        f"- enabled: `{_fmt(validation_gate.get('enabled') if validation_gate else False)}`",
        f"- method: `{_fmt(validation_gate.get('method') if validation_gate else '')}`",
        f"- selection_split: `{_fmt(validation_gate.get('selection_split') if validation_gate else '')}`",
        f"- effect_floor: `{_fmt(validation_gate.get('effect_floor') if validation_gate else '')}`",
        f"- ci_low_floor: `{_fmt(validation_gate.get('ci_low_floor') if validation_gate else '')}`",
        "",
        _markdown_table(
            validation_gate_decisions,
            ["category", "decision", "reason", "mean_delta", "ci_low", "n"],
        ),
        "",
        "## 3. Test CNR Bootstrap 摘要",
        "",
        _markdown_table(
            test_cnr,
            [
                "comparison",
                "macro_scope",
                "metric",
                "mean_delta",
                "ci_low",
                "ci_high",
                "n",
                "n_categories",
                "n_cases",
            ],
        ),
        "",
        "## 4. Per-Class Test CNR \u6458\u8981",
        "",
        _markdown_table(
            per_class_test_cnr,
            ["comparison", "category", "metric", "mean_delta", "ci_low", "ci_high", "n"],
        ),
        "",
        "## 5. 建议",
        "",
        _recommendation(decision),
        "",
    ]
    report = "\n".join(lines)
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Chinese Markdown report for Stage C metrics.")
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_md = args.output_md or (args.metrics_dir / "stage_c_result_report.md")
    report = build_stage_c_markdown_report(metrics_dir=args.metrics_dir, output_md=output_md)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
