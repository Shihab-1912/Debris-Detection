from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_utils import MODEL_PATH, train_and_save_model  # noqa: E402


if __name__ == "__main__":
    artifact = train_and_save_model()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Training rows: {artifact['training_rows']}")
    for metric, value in artifact["metrics"].items():
        print(f"{metric}: {value:.4f}")
