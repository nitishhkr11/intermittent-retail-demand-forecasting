"""Unit tests for src/utils/provenance.py (run metadata that is safe to share)."""

from __future__ import annotations

import json
import os
import platform
import re

import pytest

from src.utils.provenance import (
    build_run_metadata,
    canonical_json,
    config_section_hash,
    environment_snapshot,
    file_sha256,
    library_versions,
    python_version_tuple,
    read_json,
    write_json,
)


def test_file_sha256_matches_known_digest(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_sha256_reads_large_files_in_chunks(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 3_000_000)
    assert file_sha256(path, chunk_size=1000) == file_sha256(path)


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_config_section_hash_is_stable_and_sensitive_to_changes():
    config = {"neural": {"hidden_size": 32}, "prophet": {"interval_width": 0.8}, "other": 1}
    first = config_section_hash(config, ["neural", "prophet"])
    assert first == config_section_hash(config, ["prophet", "neural"][::-1])
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    config["neural"]["hidden_size"] = 64
    assert config_section_hash(config, ["neural", "prophet"]) != first
    # a section that was not asked for does not influence the hash
    changed_other = dict(config, other=2)
    assert config_section_hash(changed_other, ["neural", "prophet"]) == config_section_hash(config, ["neural", "prophet"])


def test_config_section_hash_raises_for_a_missing_section():
    with pytest.raises(KeyError):
        config_section_hash({"neural": {}}, ["neural", "does_not_exist"])


def test_library_versions_reports_installed_and_missing_packages():
    versions = library_versions(["numpy", "definitely-not-an-installed-package-xyz"])
    assert isinstance(versions["numpy"], str) and versions["numpy"]
    assert versions["definitely-not-an-installed-package-xyz"] is None


def test_environment_snapshot_has_no_host_user_or_path_details():
    snapshot = environment_snapshot()
    assert {"python", "python_implementation", "os_family", "machine", "libraries"} <= set(snapshot)
    text = json.dumps(snapshot)
    for private in (platform.node(), os.path.expanduser("~"), os.getcwd()):
        if private and private != "/":
            assert private not in text


def test_build_run_metadata_records_component_time_hash_and_details():
    config = {"neural": {"hidden_size": 32}}
    record = build_run_metadata("lstm_gru", config=config, config_keys=["neural"], extra={"seeds": [0, 1]})
    assert record["component"] == "lstm_gru"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["generated_at_utc"])
    assert record["config_sections"] == ["neural"]
    assert record["config_sha256"] == config_section_hash(config, ["neural"])
    assert record["details"] == {"seeds": [0, 1]}
    assert "environment" in record


def test_build_run_metadata_without_config_keys_has_no_hash():
    record = build_run_metadata("x", config={}, config_keys=())
    assert record["config_sha256"] is None
    assert "details" not in record


def test_build_run_metadata_rejects_an_empty_component_name():
    with pytest.raises(ValueError):
        build_run_metadata("", config={})


def test_json_round_trip_creates_parent_folders_and_sorts_keys(tmp_path):
    target = tmp_path / "nested" / "dir" / "meta.json"
    write_json(target, {"b": 2, "a": {"z": 1, "y": 2}})
    assert read_json(target) == {"a": {"y": 2, "z": 1}, "b": 2}
    text = target.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"') and text.endswith("\n")


def test_read_json_rejects_a_non_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_python_version_tuple_matches_the_interpreter():
    import sys

    assert python_version_tuple() == tuple(sys.version_info[:3])
