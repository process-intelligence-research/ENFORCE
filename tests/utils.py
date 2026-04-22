"""Shared test utilities for the ENFORCE benchmark test suite."""

from pathlib import Path

import pytest

DATA_DOWNLOAD_URL = "https://surfdrive.surf.nl/s/wxH67jTWfAbqTH5"

# Config keys that hold paths to external data files.
DATA_FILE_PATH_KEYS = (
    "INPUT_DATA_PATH_TRAIN",
    "OUTPUT_DATA_PATH_TRAIN",
    "INPUT_DATA_PATH_TEST",
    "OUTPUT_DATA_PATH_TEST",
    "INPUT_DATA_PATH",
    "OUTPUT_DATA_PATH",
    "PARAMS_PATH",
)


def skip_if_data_missing(paths: list[str | None], problem: str) -> None:
    """Skip the current test with a download hint if any required data files are absent.

    Parameters
    ----------
    paths:
        File paths extracted from the problem's config (``None`` values are ignored).
    problem:
        Problem name, used only for the skip message.
    """
    missing = [p for p in paths if p and p != "unused" and not Path(p).exists()]
    if missing:
        files_list = "\n    ".join(missing)
        pytest.skip(
            f"Required data files for '{problem}' not found:\n    {files_list}\n"
            f"Download the benchmark datasets from: {DATA_DOWNLOAD_URL}\n"
            f"and place them under data/raw/<problem>/ as described in the README."
        )
