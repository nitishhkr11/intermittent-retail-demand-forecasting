from __future__ import annotations

import json

import pandas as pd

from src.deployment.development_data import build_development_data


def test_development_bundle_builder_filters_sales_at_read_time(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    data = tmp_path / "data"
    processed, interim = data / "processed", data / "interim"
    processed.mkdir(parents=True)
    interim.mkdir(parents=True)
    pd.DataFrame({"date": dates, "sales": [1.0] * 35 + [999.0] * 5}).to_parquet(processed / "aggregate.parquet", index=False)
    rows = [{"date": date, "item_id": item, "sales": 1.0 if date < dates[-5] else 999.0}
            for item in ("A", "B") for date in dates]
    pd.DataFrame(rows).to_parquet(interim / "panel.parquet", index=False)
    pd.DataFrame(rows[:40]).to_parquet(processed / "selected.parquet", index=False)
    config = {
        "data": {"processed_dir": str(processed), "interim_dir": str(interim)},
        "output": {"aggregate_filename": "aggregate.parquet", "panel_filename": "panel.parquet", "selected_items_daily_sales_filename": "selected.parquet"},
        "backtest": {"holdout_days": 5},
    }
    original = pd.read_parquet
    sales_reads = []

    def audited_read(path, *args, **kwargs):
        columns = kwargs.get("columns") or []
        if "sales" in columns:
            filters = kwargs.get("filters")
            assert filters and filters[0][0:2] == ("date", "<=")
            sales_reads.append((str(path), filters[0][2]))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", audited_read)
    out = tmp_path / "dashboard_data"
    manifest = build_development_data(config, output_dir=out)
    assert manifest["development_days"] == 35
    assert manifest["development_end"] == "2020-02-04"
    assert manifest["all_item_days"] == 70
    assert manifest["holdout_dates_included"] is False
    assert manifest["holdout_target_values_read"] is False
    assert len(sales_reads) == 3
    for name in ("aggregate_development.parquet", "all_items_development.parquet", "selected_items_development.parquet"):
        table = original(out / name)
        assert pd.to_datetime(table["date"]).max() == pd.Timestamp("2020-02-04")
        assert 999.0 not in table["sales"].to_numpy()
    saved_manifest = json.loads((out / "development_data_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
