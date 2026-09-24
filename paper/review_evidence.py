"""Reproduce the 2026-09-17 desk audit without loading models or running MD.

Run from any directory with the project's Python environment. The split helper
is extracted from the actual source so this check does not import TorchMD-Net.
"""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "src/waterbox_data.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "random_split")
namespace = {"random": random, "Subset": lambda dataset, indices: indices}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
split = lambda seed: namespace["random_split"](range(1593), seed=seed)
train42, val42, test42 = split(42)
reference = set(np.random.default_rng(42).choice(1593, size=200, replace=False).tolist())
evidence = {
    "review_date": "2026-09-17",
    "assumptions": "1593 configurations, unchanged dataset ordering, documented/default seed 42 for fine-tuning and rollout selection; original run commands are not fully archived",
    "split_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "split_sizes": [len(train42), len(val42), len(test42)],
    "seed42_rollout_dataset_indices": test42[:6],
    "seed42_reference_target_overlap_seed42_test": len(reference & set(test42)),
    "by_training_seed": {},
}
for seed in range(6):
    train, val, test = map(set, split(seed))
    evidence["by_training_seed"][seed] = {
        "rollout_start_membership": ["train" if i in train else "validation" if i in val else "test" for i in test42[:6]],
        "seed42_test_overlap_original_train_val_test": [len(set(test42) & x) for x in (train, val, test)],
        "seed42_training_pool_overlap_original_test": len(set(train42) & test),
        "reference_target_overlap_original_test": len(reference & test),
    }
evidence["extended_checkpoint_history"] = {}
for config_path in sorted((ROOT / "checkpoints/waterbox_study_zbl_bonded_ext70").glob("*/seed*/config.json")):
    config = json.loads(config_path.read_text())
    history = json.loads((config_path.parent / "best_model_history.json").read_text())
    matching = [row["epoch"] for row in history if row.get("val_checkpoint_score_raw") == config["best_model_score"]]
    evidence["extended_checkpoint_history"][str(config_path.parent.relative_to(ROOT))] = {
        "history_epoch_labels_matching_saved_best_score": matching,
        "last_history_epoch_label": history[-1]["epoch"],
        "caution": "History callback ordering can affect epoch labels; verify actual epoch from remote checkpoint metadata.",
    }
path = ROOT / "results/waterbox_rollout_stable_compare/water_absolute_seed1_local_v1/raw_results.csv"
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
before = {row["velocity_seed"]: float(row["plateau_temperature_mean"]) for row in rows if row["condition"] == "before_finetune"}
after = {row["velocity_seed"]: float(row["plateau_temperature_mean"]) for row in rows if row["condition"] == "after_finetune"}
assert before.keys() == after.keys()
differences = [after[key] - before[key] for key in sorted(before)]
mean = statistics.mean(differences)
half_width = 2.776445105 * statistics.stdev(differences) / math.sqrt(len(differences))
evidence["local_finetune_paired_temperature"] = {
    "source": str(path.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "n": len(differences), "after_minus_before_k": differences,
    "mean_difference_k": mean,
    "approximate_95_percent_paired_t_interval_k": [mean - half_width, mean + half_width],
    "assumptions": "Independent velocity draws conditional on one fixed checkpoint and geometry; normal paired differences; n=5; no inference across training seeds.",
}
output = Path(__file__).with_name("review_evidence_2026_09_17.json")
output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(json.dumps(evidence, indent=2))
