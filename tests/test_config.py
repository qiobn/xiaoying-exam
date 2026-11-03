from pathlib import Path

from knowledge_agent.config import load_settings


def test_settings_file_is_loaded_with_shell_override(tmp_path, monkeypatch):
    config_dir = tmp_path / "local-config"
    config_dir.mkdir()
    (config_dir / ".env").write_text(
        "KNOWLEDGE_AGENT_MODE=real\nOPENAI_MODEL=file-model\nKNOWLEDGE_AGENT_CHUNK_SIZE=777\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KNOWLEDGE_AGENT_SETTINGS_DIR", str(config_dir))
    monkeypatch.delenv("KNOWLEDGE_AGENT_MODE", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("KNOWLEDGE_AGENT_CHUNK_SIZE", raising=False)
    settings = load_settings()
    assert settings.mode == "real"
    assert settings.model == "file-model"
    assert settings.chunk_size == 777

    monkeypatch.setenv("OPENAI_MODEL", "shell-model")
    assert load_settings().model == "shell-model"


def test_default_paths_are_rooted_at_project_directory(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_AGENT_DB", raising=False)
    settings = load_settings()
    root = Path(__file__).resolve().parents[1]
    assert settings.db_path == root / "runtime" / "knowledge.db"
