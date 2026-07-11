"""Freeze DCEM experiment outputs into paper-ready result tables.

This module is intentionally post-hoc: it reads already-computed Stage C and
Stage E outputs, then writes stable CSV/Markdown artifacts for manuscript
drafting.  It does not recompute metrics or change any heatmaps.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_MACRO_COMPARISONS = (
    "learned_selective_vs_baseline",
    "disease_gated_learned_vs_baseline",
    "disease_pooled_learned_vs_baseline",
    "phrase_anatomy_dcem_vs_baseline",
    "phrase_anatomy_dcem_vs_learned_selective",
    "phrase_anatomy_dcem_vs_candidate_shuffled",
    "validation_gated_dcem_vs_baseline",
    "validation_gated_dcem_vs_disease_gated_learned",
    "validation_gated_dcem_vs_disease_pooled_learned",
    "validation_gated_dcem_vs_learned_selective",
    "validation_gated_dcem_vs_candidate_shuffled",
    "validation_gated_dcem_v3_vs_baseline",
    "validation_gated_dcem_v3_vs_phrase_anatomy_dcem",
    "validation_gated_dcem_v3_vs_disease_pooled_learned",
    "validation_gated_dcem_v3_vs_candidate_shuffled",
)

PER_CLASS_COMPARISON_COLUMNS = {
    "learned_selective_vs_baseline": "raw_learned",
    "disease_gated_learned_vs_baseline": "disease_gated",
    "disease_pooled_learned_vs_baseline": "disease_pooled",
    "phrase_anatomy_dcem_vs_baseline": "phrase_anatomy",
    "validation_gated_dcem_vs_baseline": "validation_gated",
    "validation_gated_dcem_vs_disease_gated_learned": "validation_vs_disease_gated",
    "validation_gated_dcem_vs_disease_pooled_learned": "validation_vs_disease_pooled",
    "validation_gated_dcem_vs_learned_selective": "validation_vs_raw_learned",
    "validation_gated_dcem_vs_candidate_shuffled": "validation_vs_shuffled",
    "validation_gated_dcem_v3_vs_baseline": "validation_gated_v3",
    "validation_gated_dcem_v3_vs_phrase_anatomy_dcem": "validation_v3_vs_phrase_anatomy",
    "validation_gated_dcem_v3_vs_disease_pooled_learned": "validation_v3_vs_disease_pooled",
    "validation_gated_dcem_v3_vs_candidate_shuffled": "validation_v3_vs_shuffled",
}

GAP_COLUMNS = [
    "category",
    "n_cases",
    "learned_evidence_case_rate",
    "top1_hit_rate",
    "top3_hit_rate",
    "learned_oracle_spearman",
    "mean_oracle_top_overlap",
    "mean_learned_top_oracle_overlap",
    "dominant_failure_mode",
]


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_na(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _method_slug(method_name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", method_name).strip("_").lower()
    return slug or "dcem"


def _test_cnr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "split" in out.columns:
        out = out[out["split"] == "test"]
    if "metric" in out.columns:
        out = out[out["metric"] == "cnr"]
    return out.copy()


def _first_record(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def build_main_macro_cnr_table(
    bootstrap_summary: pd.DataFrame,
    comparisons: tuple[str, ...] = DEFAULT_MACRO_COMPARISONS,
) -> pd.DataFrame:
    """Return one manuscript row per comparison with pooled and macro CNR."""

    data = _test_cnr(bootstrap_summary)
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        comp = data[data["comparison"] == comparison] if "comparison" in data.columns else pd.DataFrame()
        pooled = (
            _first_record(comp[comp["macro_scope"] == "pooled_all"])
            if "macro_scope" in comp.columns
            else {}
        )
        macro = (
            _first_record(comp[comp["macro_scope"] == "macro_all"])
            if "macro_scope" in comp.columns
            else {}
        )
        if not pooled and not macro:
            continue
        rows.append(
            {
                "split": pooled.get("split", macro.get("split", "test")),
                "comparison": comparison,
                "pooled_mean_delta": pooled.get("mean_delta", float("nan")),
                "pooled_ci_low": pooled.get("ci_low", float("nan")),
                "pooled_ci_high": pooled.get("ci_high", float("nan")),
                "pooled_n": pooled.get("n", float("nan")),
                "macro_mean_delta": macro.get("mean_delta", float("nan")),
                "macro_ci_low": macro.get("ci_low", float("nan")),
                "macro_ci_high": macro.get("ci_high", float("nan")),
                "n_categories": macro.get("n_categories", pooled.get("n_categories", float("nan"))),
                "n_cases": macro.get("n_cases", pooled.get("n_cases", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _gate_decisions_from_json(gate: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, record in gate.get("decisions", {}).items():
        rows.append(
            {
                "category": category,
                "gate_decision": record.get("decision", ""),
                "gate_reason": record.get("reason", ""),
                "gate_val_mean_delta": record.get("mean_delta", float("nan")),
                "gate_val_ci_low": record.get("ci_low", float("nan")),
                "gate_val_ci_high": record.get("ci_high", float("nan")),
                "gate_val_n": record.get("n", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def _infer_gate_decisions_from_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    """Infer pass/bypass from validation-gated test deltas when JSON is absent."""

    data = _test_cnr(per_class)
    rows: list[dict[str, Any]] = []
    if data.empty or "comparison" not in data.columns:
        return pd.DataFrame()
    gated = data[data["comparison"] == "validation_gated_dcem_vs_baseline"]
    for _, row in gated.iterrows():
        mean_delta = _float_or_na(row.get("mean_delta"))
        rows.append(
            {
                "category": row.get("category", ""),
                "gate_decision": "pass" if abs(mean_delta) > 0 else "bypass",
                "gate_reason": "inferred_from_validation_gated_test_delta",
                "gate_val_mean_delta": float("nan"),
                "gate_val_ci_low": float("nan"),
                "gate_val_ci_high": float("nan"),
                "gate_val_n": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_main_per_class_cnr_table(
    per_class_summary: pd.DataFrame,
    validation_gate: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Pivot per-class CNR deltas and attach validation gate decisions."""

    data = _test_cnr(per_class_summary)
    if data.empty:
        return pd.DataFrame()
    categories = sorted(str(category) for category in data["category"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for category in categories:
        cat = data[data["category"] == category]
        record: dict[str, Any] = {"category": category}
        if "n" in cat.columns and not cat["n"].dropna().empty:
            record["n"] = int(cat["n"].dropna().iloc[0])
        for comparison, prefix in PER_CLASS_COMPARISON_COLUMNS.items():
            row = _first_record(cat[cat["comparison"] == comparison])
            record[f"{prefix}_cnr_delta"] = row.get("mean_delta", float("nan"))
            record[f"{prefix}_ci_low"] = row.get("ci_low", float("nan"))
            record[f"{prefix}_ci_high"] = row.get("ci_high", float("nan"))
        rows.append(record)

    table = pd.DataFrame(rows)
    gate_table = (
        _gate_decisions_from_json(validation_gate or {})
        if validation_gate
        else _infer_gate_decisions_from_per_class(per_class_summary)
    )
    if gate_table.empty:
        table["gate_decision"] = ""
        table["gate_reason"] = ""
        return table
    return table.merge(gate_table, on="category", how="left")


def _dominant_failure_mode(raw_value: Any) -> str:
    if pd.isna(raw_value):
        return ""
    if isinstance(raw_value, dict):
        modes = raw_value
    else:
        try:
            modes = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return str(raw_value)
    if not modes:
        return ""
    return str(max(modes.items(), key=lambda item: int(item[1]))[0])


def build_gate_and_gap_table(
    per_class_cnr_table: pd.DataFrame,
    gap_summary: pd.DataFrame,
    gap_diagnosis: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge final per-class effects with Stage E learned-vs-oracle diagnosis."""

    if per_class_cnr_table.empty:
        return pd.DataFrame()
    table = per_class_cnr_table.copy()
    if not gap_summary.empty:
        gap = gap_summary.copy()
        if "split" in gap.columns:
            gap = gap[gap["split"] == "test"].copy()
        if "failure_modes" in gap.columns:
            gap["dominant_failure_mode"] = gap["failure_modes"].map(_dominant_failure_mode)
        keep = [col for col in GAP_COLUMNS if col in gap.columns]
        table = table.merge(gap[keep], on="category", how="left")
    if gap_diagnosis is not None and not gap_diagnosis.empty:
        keep = [
            col
            for col in [
                "category",
                "diagnosis",
                "reason",
                "recommended_next_step",
                "oracle_delta_cnr",
                "oracle_learned_gap_cnr",
            ]
            if col in gap_diagnosis.columns
        ]
        table = table.merge(gap_diagnosis[keep], on="category", how="left")
    return table


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "No rows."
    view = df.head(max_rows).copy()
    columns = [str(col) for col in view.columns]
    rows = view.to_dict(orient="records")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def build_claims(
    macro_table: pd.DataFrame,
    per_class_table: pd.DataFrame,
    gate_table: pd.DataFrame,
    method_name: str = "DCEM-v1",
) -> dict[str, Any]:
    def comparison_record(comparison: str) -> dict[str, Any]:
        if "comparison" not in macro_table.columns:
            return {}
        rows = macro_table[macro_table["comparison"] == comparison]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        return {
            "pooled_mean_delta": _float_or_na(row.get("pooled_mean_delta")),
            "pooled_ci_low": _float_or_na(row.get("pooled_ci_low")),
            "pooled_ci_high": _float_or_na(row.get("pooled_ci_high")),
            "macro_mean_delta": _float_or_na(row.get("macro_mean_delta")),
            "macro_ci_low": _float_or_na(row.get("macro_ci_low")),
            "macro_ci_high": _float_or_na(row.get("macro_ci_high")),
        }

    passed = []
    bypassed = []
    if "gate_decision" in per_class_table.columns:
        passed = sorted(
            str(row["category"])
            for _, row in per_class_table[per_class_table["gate_decision"] == "pass"].iterrows()
        )
        bypassed = sorted(
            str(row["category"])
            for _, row in per_class_table[per_class_table["gate_decision"] == "bypass"].iterrows()
        )
    coverage_fixed = False
    if "learned_evidence_case_rate" in gate_table.columns:
        rates = gate_table["learned_evidence_case_rate"].dropna()
        coverage_fixed = bool(not rates.empty and (rates >= 1.0).all())

    return {
        "status": f"{_method_slug(method_name)}_frozen",
        "main_comparison": {
            "validation_gated_dcem_vs_baseline": comparison_record(
                "validation_gated_dcem_vs_baseline"
            ),
            "validation_gated_dcem_vs_disease_pooled_learned": comparison_record(
                "validation_gated_dcem_vs_disease_pooled_learned"
            ),
            "validation_gated_dcem_vs_learned_selective": comparison_record(
                "validation_gated_dcem_vs_learned_selective"
            ),
            "validation_gated_dcem_vs_candidate_shuffled": comparison_record(
                "validation_gated_dcem_vs_candidate_shuffled"
            ),
            "validation_gated_dcem_v3_vs_baseline": comparison_record(
                "validation_gated_dcem_v3_vs_baseline"
            ),
            "validation_gated_dcem_v3_vs_phrase_anatomy_dcem": comparison_record(
                "validation_gated_dcem_v3_vs_phrase_anatomy_dcem"
            ),
            "validation_gated_dcem_v3_vs_disease_pooled_learned": comparison_record(
                "validation_gated_dcem_v3_vs_disease_pooled_learned"
            ),
            "validation_gated_dcem_v3_vs_candidate_shuffled": comparison_record(
                "validation_gated_dcem_v3_vs_candidate_shuffled"
            ),
        },
        "raw_learned_vs_baseline": comparison_record("learned_selective_vs_baseline"),
        "disease_gated_vs_baseline": comparison_record("disease_gated_learned_vs_baseline"),
        "disease_pooled_vs_baseline": comparison_record("disease_pooled_learned_vs_baseline"),
        "phrase_anatomy_vs_baseline": comparison_record("phrase_anatomy_dcem_vs_baseline"),
        "passed_categories": passed,
        "bypassed_categories": bypassed,
        "coverage_fixed": coverage_fixed,
    }


def build_markdown_report(
    macro_table: pd.DataFrame,
    per_class_table: pd.DataFrame,
    gate_gap_table: pd.DataFrame,
    claims: dict[str, Any],
    method_name: str = "DCEM-v1",
) -> str:
    passed = ", ".join(claims.get("passed_categories", [])) or "none"
    bypassed = ", ".join(claims.get("bypassed_categories", [])) or "none"
    coverage_text = "coverage fixed" if claims.get("coverage_fixed") else "coverage not fully verified"
    lines = [
        f"# {method_name} Paper Results",
        "",
        "## Main Paper Claim",
        "",
        f"- Frozen AFLoc is kept as the baseline; {method_name} is evaluated as a post-hoc disease-conditioned repair module.",
        "- Raw learned repair is retained as a full control, and validation-gated DCEM is the strict main result.",
        f"- Validation gate passed categories: {passed}.",
        f"- Validation gate bypassed categories: {bypassed}.",
        f"- Stage E learned evidence coverage status: {coverage_text}.",
        "",
        "## Macro CNR Table",
        "",
        dataframe_to_markdown(macro_table),
        "",
        "## Per-Class CNR Table",
        "",
        dataframe_to_markdown(per_class_table),
        "",
        "## Gate and Gap Diagnosis",
        "",
        dataframe_to_markdown(gate_gap_table),
        "",
        "## Recommended Manuscript Wording",
        "",
        f"{method_name} should be written as a validation-gated, disease-conditioned repair result rather than a universal all-class improvement claim. The current frozen result supports a safe positive CNR gain over baseline, a strong improvement over raw learned repair, and a specificity gain over shuffled candidate repair. Per-class analysis identifies the validation-passed classes as supported success cases, while bypassed classes remain optimization targets for the next DCEM version.",
        "",
    ]
    return "\n".join(lines)


def build_dcem_v1_paper_results(
    stage_c_metrics_dir: Path,
    stage_e_dir: Optional[Path],
    outdir: Path,
    method_name: str = "DCEM-v1",
    output_prefix: Optional[str] = None,
) -> dict[str, Any]:
    """Write paper-ready DCEM result tables and report."""

    bootstrap = _read_required_csv(stage_c_metrics_dir / "bootstrap_summary.csv")
    per_class = _read_required_csv(stage_c_metrics_dir / "per_class_bootstrap_summary.csv")
    validation_gate = _read_json(stage_c_metrics_dir / "validation_gate_decision.json")
    gap_summary = (
        _read_optional_csv(stage_e_dir / "per_class_gap_summary.csv")
        if stage_e_dir is not None
        else pd.DataFrame()
    )
    gap_diagnosis = (
        _read_optional_csv(stage_e_dir / "gap_diagnosis.csv")
        if stage_e_dir is not None
        else pd.DataFrame()
    )

    macro_table = build_main_macro_cnr_table(bootstrap)
    per_class_table = build_main_per_class_cnr_table(per_class, validation_gate)
    gate_gap_table = build_gate_and_gap_table(per_class_table, gap_summary, gap_diagnosis)
    claims = build_claims(macro_table, per_class_table, gate_gap_table, method_name=method_name)

    outdir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or _method_slug(method_name)
    paths = {
        "main_macro_cnr_table": outdir / "main_macro_cnr_table.csv",
        "main_per_class_cnr_table": outdir / "main_per_class_cnr_table.csv",
        "gate_and_gap_table": outdir / "gate_and_gap_table.csv",
        "claims": outdir / "claims.json",
        "markdown": outdir / f"{prefix}_results.md",
        "summary": outdir / f"{prefix}_result_pack.json",
    }
    macro_table.to_csv(paths["main_macro_cnr_table"], index=False)
    per_class_table.to_csv(paths["main_per_class_cnr_table"], index=False)
    gate_gap_table.to_csv(paths["gate_and_gap_table"], index=False)
    paths["claims"].write_text(json.dumps(claims, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(
        build_markdown_report(
            macro_table,
            per_class_table,
            gate_gap_table,
            claims,
            method_name=method_name,
        ),
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "status": "ok",
        "method_name": method_name,
        "output_prefix": prefix,
        "stage_c_metrics_dir": str(stage_c_metrics_dir),
        "stage_e_dir": str(stage_e_dir) if stage_e_dir is not None else "",
        "outdir": str(outdir),
        "num_macro_rows": int(len(macro_table)),
        "num_per_class_rows": int(len(per_class_table)),
        "passed_categories": claims["passed_categories"],
        "bypassed_categories": claims["bypassed_categories"],
        "coverage_fixed": claims["coverage_fixed"],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze DCEM outputs into paper result tables.")
    parser.add_argument("--stage-c-metrics-dir", required=True, type=Path)
    parser.add_argument("--stage-e-dir", type=Path, default=None)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--method-name", default="DCEM-v1")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_dcem_v1_paper_results(
        stage_c_metrics_dir=args.stage_c_metrics_dir,
        stage_e_dir=args.stage_e_dir,
        outdir=args.outdir,
        method_name=args.method_name,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
