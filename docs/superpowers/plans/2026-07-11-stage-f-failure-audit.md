# Stage F Failure Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-hoc Stage F diagnostic that audits Pneumonia, Consolidation, and Lung Opacity failure modes before designing DCEM-v4.

**Architecture:** Reuse Stage C and Stage E artifacts instead of recomputing localization. The new module loads per-case metrics, case-level gap rows, region-level gap rows, and optional prepared-input phrases, then writes phrase subtype, region confusion, shortcut, case example, and markdown report outputs.

**Tech Stack:** Python 3.9, pandas, numpy, pytest, existing AnaPrior CSV/NPZ artifacts, bash runner.

## Global Constraints

- Do not train or tune on MS-CXR boxes; Stage F is post-hoc diagnosis only.
- Do not modify raw learned, disease-pooled, phrase-anatomy, validation-gated, or oracle evaluation logic.
- Default audit categories are exactly `Pneumonia,Consolidation,Lung Opacity`.
- Outputs must be CSV/JSON/Markdown files that can guide DCEM-v4 input and loss design.
- Follow TDD: write failing tests before production code.

---

### Task 1: Failure Audit Core

**Files:**
- Create: `tests/test_audit_failure_modes.py`
- Create: `anaprior/eval/audit_failure_modes.py`

**Interfaces:**
- Consumes:
  - `build_failure_audit(case_gap: pd.DataFrame, region_gap: pd.DataFrame, per_case_metrics: pd.DataFrame, categories: Sequence[str], method: str, baseline_method: str) -> FailureAuditResult`
  - `write_failure_audit(result: FailureAuditResult, outdir: Path) -> dict[str, str]`
- Produces:
  - Dataclass `FailureAuditResult(phrase_summary, region_confusion, shortcut_summary, case_examples, report_markdown, recommendations)`
  - CSV-ready data frames for phrase subtype distribution, learned-vs-oracle top-region confusion, shortcut burden, and representative failure cases.

- [ ] **Step 1: Write failing tests**

```python
def test_build_failure_audit_summarizes_three_target_classes():
    case_gap = pd.DataFrame([...])
    region_gap = pd.DataFrame([...])
    metrics = pd.DataFrame([...])
    result = build_failure_audit(
        case_gap=case_gap,
        region_gap=region_gap,
        per_case_metrics=metrics,
        categories=["Pneumonia", "Consolidation", "Lung Opacity"],
        method="phrase_anatomy_dcem",
        baseline_method="baseline",
    )
    assert set(result.phrase_summary["category"]) == {"Pneumonia", "Consolidation", "Lung Opacity"}
    assert "cardiac_or_hilar_shortcut_rate" in result.shortcut_summary.columns
    assert "recommended_v4_action" in result.recommendations.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit_failure_modes.py -q`

Expected: FAIL because `anaprior.eval.audit_failure_modes` does not exist.

- [ ] **Step 3: Implement core audit functions**

Implement:

```python
TARGET_FAILURE_CATEGORIES = ("Pneumonia", "Consolidation", "Lung Opacity")

def phrase_subtype(phrase: str) -> str:
    ...

def build_failure_audit(...) -> FailureAuditResult:
    ...

def write_failure_audit(result: FailureAuditResult, outdir: Path) -> dict[str, str]:
    ...
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_audit_failure_modes.py -q`

Expected: PASS.

### Task 2: CLI and Runner

**Files:**
- Modify: `tests/test_audit_failure_modes.py`
- Modify: `anaprior/eval/audit_failure_modes.py`
- Create: `scripts/run_stage_f_failure_audit_server.sh`

**Interfaces:**
- Consumes Stage E output files:
  - `--case-gap-csv`
  - `--region-gap-csv`
  - `--metrics-csv`
  - `--outdir`
  - `--categories`
  - `--method`
  - `--baseline-method`
- Produces:
  - `failure_phrase_summary.csv`
  - `failure_region_confusion.csv`
  - `failure_shortcut_summary.csv`
  - `failure_case_examples.csv`
  - `failure_v4_recommendations.csv`
  - `failure_audit_report.md`
  - `failure_audit_manifest.json`

- [ ] **Step 1: Write failing CLI and script tests**

```python
def test_main_writes_manifest_and_report(tmp_path):
    ...
    assert (tmp_path / "failure_audit_manifest.json").exists()
    assert (tmp_path / "failure_audit_report.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit_failure_modes.py -q`

Expected: FAIL because CLI output writing is missing.

- [ ] **Step 3: Implement CLI and bash runner**

Add `main(argv=None) -> int` to the module and a bash runner that preflights the three required CSVs before calling `python -m anaprior.eval.audit_failure_modes`.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/test_audit_failure_modes.py tests/test_stage_c_server_script.py -q`

Expected: PASS.

### Task 3: Verification

**Files:**
- Test only.

- [ ] **Step 1: Run relevant tests**

Run: `pytest tests/test_audit_failure_modes.py tests/test_analyze_learned_oracle_gap.py -q`

Expected: PASS.

- [ ] **Step 2: Run full test suite if practical**

Run: `pytest -q`

Expected: PASS.

