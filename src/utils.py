from __future__ import annotations
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"

def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def infer_target_column(df_columns) -> str:
    # Most common for this dataset: "Result"
    candidates = ["Result", "result", "label", "Label", "Class", "class", "target", "Target", "status", "Status"]
    for c in candidates:
        if c in df_columns:
            return c
    # fallback: last column
    return df_columns[-1]
