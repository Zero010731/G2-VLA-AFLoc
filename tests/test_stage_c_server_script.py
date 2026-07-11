from pathlib import Path


def test_stage_b_8class_predictor_server_script_contains_training_pipeline() -> None:
    script = Path("scripts/run_stage_b_8class_predictor_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "EIGHT_FINDINGS" in script
    assert "Atelectasis,Cardiomegaly,Consolidation,Edema,Lung Opacity,Pleural Effusion,Pneumonia,Pneumothorax" in script
    assert "ANAPRIOR_RUN_SMOKE" in script
    assert "LABEL_POLICY" in script
    assert "EVAL_BATCH_SIZE" in script
    assert "DCEM_V2_RANK_LOSS_WEIGHT" in script
    assert "DCEM_V2_RANK_MARGIN" in script
    assert "DCEM_V2_MAX_RANK_PAIRS_PER_FINDING" in script
    assert "anaprior.data.build_region_finding_table" in script
    assert "--split-name train" in script
    assert "--split-name valid" in script
    assert "region_finding_report_train.json" in script
    assert "region_finding_report_valid.json" in script
    assert "Validating train/valid region table coverage" in script
    assert "missing_or_empty" in script
    assert script.index("Validating train/valid region table coverage") < script.index("[3/5] Extracting train region features")
    assert "anaprior.features.extract_region_features" in script
    assert "anaprior.train.train_region_predictor" in script
    assert "--eval-batch-size" in script
    assert "--rank-loss-weight" in script
    assert "--rank-margin" in script
    assert "--max-rank-pairs-per-finding" in script
    assert "region_predictor.pt" in script
    assert "checkpoint finding_vocab" in script


def test_stage_b_split_server_script_writes_to_output_base() -> None:
    script = Path("scripts/run_stage_b_patient_splits_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "anaprior.data.build_patient_splits" in script
    assert "MS_CXR_JSON" in script
    assert "CHEST_IMAGENOME_ROOT" in script
    assert "SPLIT_DIR" in script
    assert "leakage_report.json" in script
    assert "train_mscxr_overlap" in script


def test_stage_c_server_script_contains_full_pipeline_commands() -> None:
    script = Path("scripts/run_stage_c_learned_repair_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "MAX_CASES" in script
    assert "STAGE_C_ALPHA" in script
    assert "ANAPRIOR_RUN_SMOKE" in script
    assert "AFLOC_BERT_TYPE" in script
    assert "AFLOC_HF_LOCAL_FILES_ONLY" in script
    assert "/mnt/zhangran/Bio_ClinicalBERT" in script
    assert "USE_PRECOMPUTED_REGION_SCORE_CSV" in script
    assert "PROGRESS_EVERY" in script
    assert "EXCLUDE_REGIONS" in script
    assert "DATASET" in script
    assert "ENABLE_VALIDATION_GATE" in script
    assert "STAGE_C_METHODS" in script
    assert "VALIDATION_GATE_SOURCE_METHOD" in script
    assert "VALIDATION_GATE_METHOD_NAME" in script
    assert "VALIDATION_GATE_EFFECT_FLOOR" in script
    assert "VALIDATION_GATE_CI_LOW_FLOOR" in script
    assert "--dataset" in script
    assert "--validation-gate" in script
    assert "--validation-gate-source-method" in script
    assert "--validation-gate-method-name" in script
    assert "--validation-gate-effect-floor" in script
    assert "--validation-gate-ci-low-floor" in script
    assert "--exclude-regions" in script
    assert "--progress-every" in script
    assert "Skipping AFLoc feature extraction" in script
    assert "Skipping learned score export" in script
    assert "anaprior.eval.stage_c_preflight" in script
    assert "preflight_report.json" in script
    assert "anaprior.eval.prepare_mscxr_repair_inputs" in script
    assert "anaprior.features.extract_region_features" in script
    assert "anaprior.eval.predict_region_scores" in script
    assert "anaprior.eval.eval_mscxr_learned_repair" in script
    assert "anaprior.eval.score_mscxr_learned_repair_metrics" in script
    assert "anaprior.eval.report_stage_c_results" in script
    assert "stage_c_result_report.md" in script
    assert "learned_repair_decision.json" in script


def test_stage_d_oracle_recoverability_server_script_contains_profile_command() -> None:
    script = Path("scripts/run_stage_d_oracle_recoverability_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "ANAPRIOR_RUN_SMOKE" in script
    assert "PREPARED_INPUTS_NPZ" in script
    assert "${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c/mscxr/mscxr_repair_inputs.npz" in script
    assert "DATASET" in script
    assert "STAGE_D_ALPHA" in script
    assert "BOOTSTRAP_REPLICATES" in script
    assert "anaprior.eval.eval_mscxr_oracle_recoverability" in script
    assert "--dataset" in script
    assert "oracle_recoverability_summary.json" in script
    assert "recoverability_profile.csv" in script


def test_stage_e_8class_gap_diagnosis_server_script_contains_gap_command() -> None:
    script = Path("scripts/run_stage_e_8class_gap_diagnosis_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "PREPARED_INPUTS_NPZ" in script
    assert "${ANAPRIOR_OUTPUT_BASE}/anaprior_stage_c/mscxr/mscxr_repair_inputs.npz" in script
    assert "REGION_SCORE_CSV" in script
    assert "LEARNED_METRICS_CSV" in script
    assert "ORACLE_PROFILE_CSV" in script
    assert "DATASET" in script
    assert "EIGHT_FINDINGS" in script
    assert "EVIDENCE_MODE" in script
    assert "[Stage E] Preflight checks" in script
    assert "required input missing" in script
    assert 'find "${ANAPRIOR_OUTPUT_BASE}"' in script
    assert "anaprior.eval.analyze_learned_oracle_gap" in script
    assert "--dataset" in script
    assert "--evidence-mode" in script
    assert "--oracle-profile-csv" in script
    assert "gap_diagnosis.csv" in script


def test_stage_f_failure_audit_server_script_contains_failure_audit_command() -> None:
    script = Path("scripts/run_stage_f_failure_audit_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "CASE_GAP_CSV" in script
    assert "REGION_GAP_CSV" in script
    assert "METRICS_CSV" in script
    assert "PREPARED_INPUTS_NPZ" in script
    assert "FAILURE_CATEGORIES" in script
    assert "Pneumonia,Consolidation,Lung Opacity" in script
    assert "[Stage F] Preflight checks" in script
    assert "required input missing" in script
    assert 'find "${ANAPRIOR_OUTPUT_BASE}"' in script
    assert "anaprior.eval.audit_failure_modes" in script
    assert "--case-gap-csv" in script
    assert "--region-gap-csv" in script
    assert "--metrics-csv" in script
    assert "--prepared-inputs-npz" in script
    assert "failure_audit_report.md" in script


def test_stage_g_dp_msa_train_server_script_contains_training_command() -> None:
    script = Path("scripts/run_stage_g_dp_msa_train_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "DP_MSA_TRAIN_CACHE" in script
    assert "DP_MSA_VALID_CACHE" in script
    assert "anaprior.train.train_dp_msa_adapter" in script
    assert "--train-cache" in script
    assert "--valid-cache" in script
    assert "--lambda-weight" in script
    assert "--residual-l1-weight" in script
    assert "dp_msa_adapter.pt" in script


def test_stage_g_build_dp_msa_cache_server_script_contains_cache_command() -> None:
    script = Path("scripts/run_stage_g_build_dp_msa_cache_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "PREPARED_INPUTS_NPZ" in script
    assert "REGION_SCORE_CSV" in script
    assert "BASE_HMAPS_NPY" in script
    assert "TARGET_MIX_BETA" in script
    assert "DISEASE_BETA_JSON" in script
    assert "anaprior.train.build_dp_msa_training_cache" in script
    assert "--prepared-inputs-npz" in script
    assert "--region-score-csv" in script
    assert "--base-hmaps-npy" in script
    assert "--target-mix-beta" in script
    assert "--disease-beta-json" in script
    assert "--valid-fraction" in script
    assert "--min-score-sum" in script
    assert "train_dp_msa_v0.pt" in script
    assert "valid_dp_msa_v0.pt" in script


def test_stage_h_dp_msa_eval_server_script_contains_eval_command() -> None:
    script = Path("scripts/run_stage_h_dp_msa_eval_server.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "ANAPRIOR_OUTPUT_BASE" in script
    assert "/mnt3/zhangran" in script
    assert "PREPARED_INPUTS_NPZ" in script
    assert "REGION_SCORE_CSV" in script
    assert "DP_MSA_CKPT" in script
    assert "BASE_HMAPS_NPY" in script
    assert "BASE_METHOD_NAME" in script
    assert "METHOD_NAME" in script
    assert "LAMBDA_OVERRIDE" in script
    assert "anaprior.eval.eval_mscxr_dp_msa_repair" in script
    assert "--prepared-inputs-npz" in script
    assert "--region-score-csv" in script
    assert "--checkpoint" in script
    assert "--base-hmaps-npy" in script
    assert "--base-method-name" in script
    assert "--method-name" in script
    assert "--lambda-override" in script
    assert "dp_msa_repair_summary.json" in script
