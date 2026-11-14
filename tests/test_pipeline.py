from src import pipeline


def test_development_pipeline_never_includes_final_holdout():
    assert {"data", "classify", "seasonal", "intermittent", "inventory", "xgboost", "neural", "prophet", "comparison", "reliability", "inventory-alternatives", "selection", "reports", "dashboard-data"}.issubset(pipeline.STAGES)
    assert all("final_holdout" not in " ".join(stage) for stage in pipeline.STAGES.values())


def test_resume_skips_existing_stage_output(tmp_path, monkeypatch):
    target = tmp_path / "summary.csv"
    target.write_text("saved\n", encoding="utf-8")
    monkeypatch.setitem(pipeline.OUTPUTS, "inventory", target)
    monkeypatch.setattr(pipeline, "_invoke", lambda *args: (_ for _ in ()).throw(AssertionError("must skip")))
    assert pipeline.run_stage("inventory", resume=True) == "skipped"


def test_inventory_stage_points_at_expanded_calibration():
    assert pipeline.STAGES["inventory"][:2] == ("src.evaluation.expanded_inventory", "--window-days")


def test_resume_requires_passed_xgboost_audit():
    assert pipeline._verified_existing("xgboost") is True


def test_resume_requires_complete_neural_coverage():
    assert pipeline._verified_existing("neural") is True
