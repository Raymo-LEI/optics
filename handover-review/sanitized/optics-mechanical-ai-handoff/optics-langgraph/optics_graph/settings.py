"""Local settings. Secrets never enter graph state or returned API objects."""
from pathlib import Path
import json
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "data"
CONFIG = ROOT / "config" / "local.json"
DEFAULT_REQUEST = "Assemble a simplified Michelson interferometer. Keep the laser and detector fixed, use two mirrors and one beam splitter, and perform geometric assembly only."


def read_settings() -> dict:
    settings = {"model": "gpt-6.1-sol", "api_key": "", "robot_endpoint": ""}
    if CONFIG.exists():
        settings.update(json.loads(CONFIG.read_text()))
    settings["api_key"] = os.environ.get("OPENAI_API_KEY") or settings["api_key"]
    return settings


def save_settings(model: str, api_key: str = "") -> dict:
    if not model or len(model) > 120 or any(c.isspace() for c in model):
        raise ValueError("The model name is invalid")
    current = read_settings()
    current["model"] = model
    if api_key:
        if not api_key.startswith("sk-") or any(c.isspace() for c in api_key):
            raise ValueError("The API key format is invalid")
        current["api_key"] = api_key
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=CONFIG.parent, prefix=".local-")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(current, stream)
        os.replace(name, CONFIG)
    finally:
        Path(name).unlink(missing_ok=True)
    return public_settings()


def public_settings() -> dict:
    config = read_settings()
    return {"model": config["model"], "key_configured": bool(config["api_key"]),
            "provider": "OpenAI", "robot_configured": bool(config["robot_endpoint"])}
