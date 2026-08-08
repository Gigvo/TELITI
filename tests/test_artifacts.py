"""Artefact resolution — local disk first, Hugging Face Hub as fallback.

The weights are gitignored, so a fresh clone has code and no model. These cover
the paths that decide whether such a clone runs or returns 503 from everything.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _restore_module_state():
    """Reload with the real environment after every test.

    These tests mutate module-level constants via importlib.reload. Without this
    the last patched state leaks into whatever runs next — and because the leak
    is a path constant, the symptom is an unrelated test failing on a missing
    file. Cleaning up on the way out keeps that from ever being anyone's problem.
    """
    yield

    import api.artifacts

    importlib.reload(api.artifacts)


def _reload(monkeypatch, **env):
    """Re-import with a patched environment — module constants read os.environ at import."""
    for key in ("TELITI_MODEL_REPO", "TELITI_MODEL_DIR", "TELITI_ARTIFACTS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import api.artifacts as artifacts

    return importlib.reload(artifacts)


def test_local_model_is_preferred_over_the_hub(monkeypatch, tmp_path):
    """A local copy must win: no network, no surprise version drift.

    Builds its own model directory rather than relying on artifacts/ being
    populated — the weights are gitignored, so a test that depends on them
    passes or fails according to whether someone happens to have trained
    locally, which is not a property of the code.
    """
    local = tmp_path / "scam_model"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")

    artifacts = _reload(
        monkeypatch,
        TELITI_MODEL_DIR=str(local),
        TELITI_MODEL_REPO="someone/teliti",
    )
    source, origin = artifacts.resolve_model_dir()
    assert origin == "local"
    assert source == str(local)


def test_hub_is_used_when_there_is_no_local_model(monkeypatch, tmp_path):
    artifacts = _reload(
        monkeypatch,
        TELITI_MODEL_DIR=str(tmp_path / "absent"),
        TELITI_MODEL_REPO="someone/teliti",
    )
    source, origin = artifacts.resolve_model_dir()
    assert origin == "hub"
    assert source == "someone/teliti"


def test_placeholder_repo_is_treated_as_unconfigured(monkeypatch, tmp_path):
    """An unedited CHANGE-ME/... default must fail fast.

    Attempting the download would 404 after a confusing delay. Tests the
    *behaviour* rather than asserting the shipped default still is the
    placeholder — once the repo is published that constant is correctly a real
    repo id, and a test pinned to the placeholder would start failing on the
    commit that fixed it.
    """
    artifacts = _reload(
        monkeypatch,
        TELITI_MODEL_DIR=str(tmp_path / "absent"),
        TELITI_MODEL_REPO="CHANGE-ME/teliti-job-scam-mdistilbert",
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        artifacts.resolve_model_dir()
    assert "TELITI_MODEL_REPO" in str(excinfo.value)


def test_empty_repo_disables_the_hub(monkeypatch, tmp_path):
    """Explicitly empty means "never touch the network" — for an air-gapped or
    byte-reproducible build where a silent fetch is worse than a loud failure."""
    # TELITI_ARTIFACTS must also point somewhere empty: resolve_file checks disk
    # before the Hub, so the real artifacts/ would satisfy it locally and prove
    # nothing about whether the Hub was consulted.
    artifacts = _reload(
        monkeypatch,
        TELITI_ARTIFACTS=str(tmp_path),
        TELITI_MODEL_DIR=str(tmp_path / "absent"),
        TELITI_MODEL_REPO="",
    )
    with pytest.raises(FileNotFoundError):
        artifacts.resolve_model_dir()
    assert artifacts.resolve_file("calibrator_deployment.json") is None


def test_small_artefacts_resolve_locally(monkeypatch):
    artifacts = _reload(monkeypatch)
    found = artifacts.resolve_file("calibrator_deployment.json")
    assert found is not None and found.is_file()


def test_missing_artefact_returns_none_rather_than_raising(monkeypatch, tmp_path):
    """Callers have documented degraded modes; a hard failure here would take
    down analysis over a file that only affects presentation."""
    artifacts = _reload(monkeypatch, TELITI_ARTIFACTS=str(tmp_path), TELITI_MODEL_REPO="")
    assert artifacts.resolve_file("thresholds.json") is None


def test_model_adjacent_file_resolves_locally(monkeypatch):
    """training_summary.json sits in scam_model/ locally but at the repo root on
    the Hub — same file, two layouts."""
    artifacts = _reload(monkeypatch)
    found = artifacts.resolve_model_file("training_summary.json")
    assert found is not None and found.is_file()
