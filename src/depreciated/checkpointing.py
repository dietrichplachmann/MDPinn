# src/checkpointing.py
import os
import json
import hashlib
from datetime import datetime
import torch

def file_sha256(path: str) -> str:
    """Compute SHA256 hash for a file (for data versioning)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_run_id() -> str:
    """Timestamp-based run id."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_checkpoint(
    run_id: str,
    model,
    optimizer,
    config: dict,
    train_history: dict | None = None,
    checkpoint_dir: str = "checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"run_{run_id}.pt")

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_history": train_history or {},
    }

    torch.save(payload, ckpt_path)
    print(f"[checkpoint] Saved run to {ckpt_path}")
    return ckpt_path


def load_checkpoint(path: str, map_location=None):
    """Load entire checkpoint dict."""
    return torch.load(path, map_location=map_location)

def save_config_json(run_id: str, config: dict, checkpoint_dir: str = "checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    json_path = os.path.join(checkpoint_dir, f"run_{run_id}_config.json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[checkpoint] Saved config to {json_path}")
    return json_path
