#!/usr/bin/env python
"""Verify TensorNet's own periodic neighbor search actually respects box
vectors - never directly checked before (paper/main.tex Section 5, open item
2; CLAUDE.md open items list, item 2). The bond-inference/molecule-grouping
step used by the momentum loss (structural_metrics.py) was separately fixed
and confirmed periodicity-aware - that's a different, simpler computation
this project owns. TensorNet's own distance/neighbor module, inside the
installed torchmdnet package, has not been checked the same way.

What this tests, and why "shift every atom by a lattice vector and wrap back
into the cell" is NOT the right test: that operation reproduces the exact
same absolute coordinates you started with (up to float32 noise) - it's a
no-op on the numbers the model actually sees, so it can't reveal anything.
The real, non-trivial invariance a correctly periodic model must satisfy is:
translate a SINGLE molecule by one full lattice vector, leave it UNWRAPPED
(so its raw coordinates now sit outside the nominal box) and leave every
other atom untouched - the whole system's predicted energy and every atom's
predicted force must come out exactly unchanged, because that molecule's new
coordinate is just a different periodic image of the same physical position.
If the model's distance computation uses raw Euclidean distance instead of
minimum-image distance, that molecule will suddenly look far from its real
neighbors, and both the energy and (at least) that molecule's own forces
should move by far more than float32 noise.

A HALF-lattice-vector shift is included as a negative control - that IS a
genuinely different configuration and should NOT be invariant. If it comes
out ~0 too, the test itself is broken (e.g. the model is barely using
position information), not evidence that periodicity works.

Tests an architectural property of the model's forward pass, not training
quality - any trained checkpoint (any condition/seed) works equally well as
the test subject.

Usage:
    python src/verify_periodicity.py --checkpoint checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt

IMPORTANT - written without torchmdnet installed locally (same caveat as
every other water-box script in this repo). Run this on the training box and
read the printed lattice diagonal/off-diagonal check first (see
_lattice_matrix's docstring) before trusting the invariance numbers below it.
"""

from __future__ import annotations

import torch

from evaluate_waterbox import load_waterbox_checkpoint
from structural_metrics import infer_molecule_groups
from waterbox_data import load_waterbox_dataset, random_split

_EXPECTED_BOX_ANGSTROM = 12.42  # waterbox_data.py's own empirically-confirmed box size


def _lattice_matrix(box):
    """box comes off a WaterBox sample with shape (1,3,3) (waterbox_data.py's
    docstring) - squeeze the leading singleton so lattice[i] is lattice vector i.
    Row-vector convention (lattice vector i = row i) assumed here, NOT verified
    against torchmdnet source. structural_metrics.py's own bond-inference
    already assumes this box is orthorhombic (it only ever uses the diagonal,
    see _infer_bonds_from_positions) - the diagonal/off-diagonal print in
    verify_periodicity below is a direct empirical check of that same
    assumption, not a new one, but it has never actually been printed before."""
    lattice = box.squeeze(0) if box.dim() == 3 else box
    if lattice.shape != (3, 3):
        raise ValueError(f"Expected a (3,3) lattice matrix after squeezing, got {tuple(lattice.shape)}")
    return lattice


def _predict(model, z, pos, box, device):
    pos_req = pos.detach().clone().requires_grad_(True)
    batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
    with torch.enable_grad():
        energy, force = model(z, pos_req, batch=batch, box=box)
    return energy.detach(), force.detach()


def verify_periodicity(checkpoint_path, data_root="./data", n_configs=5, seed=42, device=None):
    """For n_configs real held-out configurations, shift one molecule by a
    full (unwrapped) lattice vector and compare the model's energy/force
    predictions against the unshifted original. See module docstring for why
    this - not a whole-system shift-and-wrap - is the right invariance to
    check, and for the half-lattice-vector negative control included below.

    Returns a list of per-config, per-shift result dicts; also prints a
    human-readable report as it runs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_waterbox_checkpoint(checkpoint_path, device=device)

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)
    n_configs = min(n_configs, len(test_data))

    results = []
    for idx in range(n_configs):
        sample = test_data[idx].to(device)
        box = getattr(sample, "box", None)
        if box is None:
            raise ValueError(f"Sample {idx} has no box - can't test periodicity without one.")
        box = box.float()
        pos = sample.pos.float()
        z = sample.z

        lattice = _lattice_matrix(box)
        diag = lattice.diagonal()
        off_diag_max = (lattice - torch.diag(diag)).abs().max().item()
        print(
            f"[config {idx}] lattice diagonal = {[round(v, 3) for v in diag.tolist()]} A, "
            f"max off-diagonal = {off_diag_max:.2e} A "
            f"(expect ~{_EXPECTED_BOX_ANGSTROM} A diagonal, ~0 off-diagonal for this dataset - "
            "if this doesn't hold, the row-vector lattice convention assumed above is wrong, "
            "fix that before trusting anything below)"
        )

        group_ids = infer_molecule_groups(z, pos, box=box).to(device)
        # Which molecule gets shifted is arbitrary - the property under test
        # (minimum-image distance correctness) is per-atom, not molecule-specific.
        molecule_atoms = (group_ids == group_ids[0]).nonzero(as_tuple=True)[0]

        e0, f0 = _predict(model, z, pos, box, device)

        # model_args sets remove_ref_energy=False (train_waterbox.py), so this is
        # a raw absolute total-system energy, not a small residual - plausibly
        # large in magnitude (real DFT total energies for ~200 atoms are O(1e3-1e5)
        # eV). Print it and the plain float32 rounding floor AT THAT MAGNITUDE
        # (|E0| * 2**-23, float32's relative machine epsilon) alongside the
        # invariance diffs below - if the diffs are the same order of magnitude as
        # this floor, that's ordinary float32 noise from a large absolute energy
        # scale, not evidence of any reduced-precision (TF32/fp16) computation.
        e0_val = float(e0.squeeze().item())
        fp32_ulp = abs(e0_val) * 2 ** -23
        print(f"  baseline energy E0 = {e0_val:.4f} eV, plain float32 ULP at this magnitude ~= {fp32_ulp:.6f} eV")

        shifts = [
            ("+a", lattice[0], True),
            ("+b", lattice[1], True),
            ("+c", lattice[2], True),
            ("+a+b+c", lattice[0] + lattice[1] + lattice[2], True),
            ("-a", -lattice[0], True),
            ("+half_a (control)", 0.5 * lattice[0], False),
        ]
        for shift_name, shift_vec, expected_invariant in shifts:
            pos_shifted = pos.clone()
            pos_shifted[molecule_atoms] = pos_shifted[molecule_atoms] + shift_vec
            e1, f1 = _predict(model, z, pos_shifted, box, device)

            energy_diff = float((e1 - e0).abs().item())
            force_diff_mean = float((f1 - f0).abs().mean().item())
            force_diff_max = float((f1 - f0).abs().max().item())

            results.append({
                "config": idx,
                "shift": shift_name,
                "expected_invariant": expected_invariant,
                "energy_diff": energy_diff,
                "force_diff_mean": force_diff_mean,
                "force_diff_max": force_diff_max,
            })
            note = "expect ~0 (float32 noise)" if expected_invariant else "expect LARGE (negative control)"
            print(
                f"  shift={shift_name:18s} energy_diff={energy_diff:.6f} eV  "
                f"force_diff(mean/max)={force_diff_mean:.6f}/{force_diff_max:.6f} eV/A  [{note}]"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Any trained water-box .ckpt - tests architecture, not training quality")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n-configs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    verify_periodicity(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        n_configs=args.n_configs,
        seed=args.seed,
    )
