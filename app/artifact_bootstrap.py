"""Restore saved development artifacts from the tracked data bundle."""

from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile


BUNDLE_RELATIVE_PATH = Path(
    "reports/final/dashboard_data_bundle_development_only.zip"
)


def restore_packaged_artifacts(project_root: Path) -> tuple[str, ...]:
    """Restore missing development artifacts without overwriting existing files."""

    root = Path(project_root).resolve()
    bundle = root / BUNDLE_RELATIVE_PATH
    artifact_root = (root / "artifacts").resolve()

    if not bundle.is_file():
        raise FileNotFoundError(
            f"Dashboard artifact bundle not found: {bundle}"
        )

    restored = []

    with ZipFile(bundle, "r") as archive:
        for entry in archive.infolist():
            member = PurePosixPath(entry.filename)

            if (
                entry.is_dir()
                or not member.parts
                or member.is_absolute()
                or member.parts[0] != "artifacts"
                or ".." in member.parts
            ):
                continue

            destination = root.joinpath(*member.parts).resolve()

            if not destination.is_relative_to(artifact_root):
                raise ValueError(
                    f"Unsafe ZIP path: {entry.filename}"
                )

            if destination.exists():
                if not destination.is_file():
                    raise IsADirectoryError(destination)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)

            try:
                with archive.open(entry) as source:
                    with destination.open("xb") as target:
                        copyfileobj(source, target)
            except FileExistsError:
                continue

            restored.append(
                destination.relative_to(root).as_posix()
            )

    return tuple(restored)
