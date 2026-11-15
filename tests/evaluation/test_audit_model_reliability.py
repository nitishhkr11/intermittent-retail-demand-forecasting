from src.evaluation.audit_model_reliability import audit


def test_reliability_audit_reads_saved_diagnostics_and_reports_undercoverage():
    result = audit()
    assert result["passed"] is True
    assert result["neural"]["cells"] == 64
    assert result["neural"]["early_stopped_seeds"] == 303
    assert result["prophet"]["fits"] == 48
    assert result["prophet"]["all_variants_below_80_percent"] is True
    assert result["xgboost"]["failed"] == 0
