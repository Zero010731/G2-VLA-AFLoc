from pathlib import Path

from anaprior.eval.stage_c_preflight import StageCPreflightConfig, run_stage_c_preflight


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_stage_c_preflight_reports_missing_required_inputs(tmp_path: Path) -> None:
    report = run_stage_c_preflight(
        StageCPreflightConfig(
            ckpt=tmp_path / "missing.ckpt",
            mimic_image_root=tmp_path / "missing_mimic",
            chest_imagenome_root=tmp_path / "missing_imagenome",
            prior_table=tmp_path / "missing_prior.json",
            base_hmaps_npy=tmp_path / "missing_hmaps.npy",
            predictor_ckpt=tmp_path / "missing_predictor.pt",
            outroot=tmp_path / "out",
        )
    )

    assert report["status"] == "error"
    missing_names = {item["name"] for item in report["missing"]}
    assert {
        "ckpt",
        "mimic_image_root",
        "chest_imagenome_root",
        "chest_imagenome_scene_graph_zip",
        "prior_table",
        "base_hmaps_npy",
        "predictor_ckpt",
    } <= missing_names


def test_stage_c_preflight_passes_when_required_inputs_exist(tmp_path: Path) -> None:
    imagenome_root = tmp_path / "chest-imagenome"
    scene_zip = imagenome_root / "silver_dataset" / "scene_graph.zip"
    report_path = tmp_path / "preflight.json"
    config = StageCPreflightConfig(
        ckpt=touch(tmp_path / "weights" / "Pretrained_CXR.ckpt"),
        mimic_image_root=tmp_path / "mimic-cxr" / "jpg",
        chest_imagenome_root=imagenome_root,
        prior_table=touch(tmp_path / "prior_table.json"),
        base_hmaps_npy=touch(tmp_path / "hmaps.npy"),
        predictor_ckpt=touch(tmp_path / "region_predictor.pt"),
        outroot=tmp_path / "out",
        report_json=report_path,
    )
    config.mimic_image_root.mkdir(parents=True)
    touch(scene_zip)

    report = run_stage_c_preflight(config)

    assert report["status"] == "ok"
    assert report["missing"] == []
    assert config.outroot.exists()
    assert report_path.exists()
