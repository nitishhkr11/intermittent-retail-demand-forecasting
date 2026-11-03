"""Run provenance: what produced a saved artifact, with which settings and library versions.

Phase 3 notebooks write a small `*_run_metadata.json` next to their forecast and
metric files, and the dashboard shows it in its provenance panel. The record is
deliberately limited to facts that help someone reproduce or audit a run:

  * the component name and a UTC timestamp,
  * the Python / platform / library versions that actually ran,
  * a SHA-256 hash of the configuration section(s) the run used,
  * any extra, caller-supplied facts (seeds, window lengths, ...).

It never records host names, user names, absolute paths or environment
variables, so a metadata file can be shared (or zipped into a delivery)
without leaking anything about the machine that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from src.utils.config import load_config

#: Libraries whose versions are worth recording when they are installed.
TRACKED_LIBRARIES = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "scikit-learn",
    "xgboost",
    "torch",
    "prophet",
    "cmdstanpy",
    "streamlit",
    "plotly",
    "pyarrow",
)


def file_sha256(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, as a lowercase hex string."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON text (sorted keys, no whitespace variation) for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_section_hash(config: Mapping[str, Any], keys: Iterable[str]) -> str:
    """SHA-256 over the named top-level config sections.

    A section that is missing from `config` raises KeyError instead of being
    silently skipped: a hash of "nothing" would look like a valid fingerprint.
    """
    selected = {key: config[key] for key in keys}
    return hashlib.sha256(canonical_json(selected).encode("utf-8")).hexdigest()


def library_versions(names: Iterable[str] = TRACKED_LIBRARIES) -> Dict[str, Optional[str]]:
    """Installed version of each library, or None if it is not installed."""
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> Dict[str, Any]:
    """Python, operating-system family and tracked-library versions (no host or user details)."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_family": platform.system(),
        "machine": platform.machine(),
        "libraries": library_versions(),
    }


def build_run_metadata(
    component: str,
    config: Optional[Mapping[str, Any]] = None,
    config_keys: Iterable[str] = (),
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the metadata record for one run of `component`.

    `config_keys` names the top-level config sections that governed the run; their
    combined hash lets a reader tell whether a saved artifact still corresponds to
    the current `config.yaml`.
    """
    if not component:
        raise ValueError("component must be a non-empty string.")
    config = load_config() if config is None else config
    keys = tuple(config_keys)
    record: Dict[str, Any] = {
        "component": component,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment_snapshot(),
        "config_sections": list(keys),
        "config_sha256": config_section_hash(config, keys) if keys else None,
    }
    if extra:
        record["details"] = dict(extra)
    return record


def write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    """Write `payload` as indented, key-sorted UTF-8 JSON (parent folders are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON object from `path`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"'{path}' does not contain a JSON object.")
    return payload


def python_version_tuple() -> tuple:
    """(major, minor, micro) of the running interpreter, for compatibility messages."""
    return tuple(sys.version_info[:3])
