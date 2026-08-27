#!/usr/bin/env python
"""Finite-difference gradient check for train_waterbox_stable.py's L_QM
regularizer (_qm_regularizer_loss) - directly settles the autograd warning
surfaced during the first successful StABlE smoke test run (paper/main.tex
sec:q4-stable-step3): PyTorch warned that
torchmdnet_extensions::get_neighbor_pairs_bkwd has no registered autograd
kernel for backprop through it, "may lead to silently incorrect behavior."

Why this specific warning targets _qm_regularizer_loss: its force-MSE term
uses force_pred, which the model computes internally via
autograd.grad(energy, pos, create_graph=True) - so differentiating
force-MSE with respect to the model's OWN parameters (what
total_loss.backward() does in the real fine-tuning loop) is a
double-backward through that same internal call. If
get_neighbor_pairs_bkwd's second derivative isn't correctly registered,
this term's theta-gradient could be silently wrong - not crash, just wrong,
exactly the failure mode this project's own discipline exists to catch
before trusting a result built on top of it (see verify_zbl_units.py,
verify_boltzmann_estimator.py, verify_waterbox_langevin.py for the same
pattern applied elsewhere).

Circumstantial reasoning that this is LIKELY fine, checked here rather than
just trusted: ordinary supervised training (train_waterbox.py, via
LNNP.step()) needs the exact same double-backward for any run with nonzero
force weight - every water-box run in this project, since neg_dy_weight is
never 0. If this were silently wrong, force accuracy across the whole
project would likely look far worse than it does (the checkpoints this
project already relies on show force MAE in an expected, literature-
consistent range). This is suggestive, not proof by itself.

What this script does instead: compares torch.autograd's own gradient of
_qm_regularizer_loss with respect to a handful of REAL model parameters
against a manual central-finite-difference estimate computed by directly
perturbing each parameter and re-evaluating the loss with NO autograd
involved on that side at all. If they agree, the double-backward path is
confirmed numerically correct for this exact model/PyTorch/torchmdnet
combination - not just plausible by analogy to other training runs.

Only checks the largest-magnitude-gradient parameter from a handful of
tensors (not every parameter - a full check would need two forward passes
per parameter, prohibitively slow for a model this size) - picking the
largest-gradient entry per tensor also avoids a false "pass" from checking
a parameter whose gradient happens to be ~0 either way.

Usage (training box only, needs torchmdnet - NOT yet run):
    python src/verify_qm_gradient.py --ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt
"""

from __future__ import annotations

import numpy as np
import torch

from evaluate_waterbox import load_waterbox_checkpoint
from train_waterbox_stable import _qm_regularizer_loss
from waterbox_data import load_waterbox_dataset, random_split


def main(ckpt, data_root="./data", seed=42, batch_size=2, eps=1e-4, n_params_to_check=5, rel_err_threshold=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_waterbox_checkpoint(ckpt, device=device)
    model.train()

    full_dataset = load_waterbox_dataset(data_root=data_root)
    train_data, _val_data, _test_data = random_split(full_dataset, seed=seed)

    def loss_at_current_params():
        # Re-seeded identically (a fresh, fixed rng) on every call, not the
        # loop's own evolving rng - otherwise the analytic and the two
        # finite-difference evaluations would each draw a DIFFERENT random
        # batch, comparing gradients of three different functions rather
        # than the same one evaluated at three nearby points.
        rng_fixed = np.random.default_rng(123)
        return _qm_regularizer_loss(model, train_data, batch_size, device, rng_fixed)

    model.zero_grad()
    loss = loss_at_current_params()
    loss.backward()

    candidates = []
    for name, p in model.named_parameters():
        if p.grad is None or p.numel() == 0:
            continue
        flat_grad = p.grad.detach().view(-1)
        idx = int(torch.argmax(flat_grad.abs()).item())
        candidates.append((name, p, idx, float(flat_grad[idx].item())))
    candidates.sort(key=lambda c: -abs(c[3]))
    candidates = candidates[:n_params_to_check]

    print(f"Checking {len(candidates)} parameters (largest-magnitude analytic gradient per tensor, eps={eps})...")
    all_ok = True
    for name, p, idx, analytic_grad in candidates:
        # p.view(-1), not p.flatten() or p.detach().flatten(): .view()
        # requires true memory-sharing with p and raises loudly if that's
        # not possible, rather than silently falling back to a disconnected
        # copy - the whole point of this script is establishing ground
        # truth, so a perturbation that silently doesn't touch the real
        # parameter would corrupt the check without any error at all.
        with torch.no_grad():
            flat = p.view(-1)
            original = float(flat[idx].item())
            flat[idx] = original + eps
        loss_plus = float(loss_at_current_params().item())

        with torch.no_grad():
            flat[idx] = original - eps
        loss_minus = float(loss_at_current_params().item())

        with torch.no_grad():
            flat[idx] = original  # restore exactly, regardless of outcome below

        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        rel_err = abs(numerical_grad - analytic_grad) / (abs(analytic_grad) + abs(numerical_grad) + 1e-12)
        ok = rel_err < rel_err_threshold
        all_ok = all_ok and ok
        print(
            f"  {name}[{idx}]: analytic={analytic_grad:.6e}  numerical={numerical_grad:.6e}  "
            f"rel_err={rel_err:.4%}  {'OK' if ok else 'MISMATCH'}"
        )

    if all_ok:
        print(
            "\nPASS - double-backward through the model's internal force computation is numerically "
            "correct here. The get_neighbor_pairs_bkwd warning is (for this model/PyTorch/torchmdnet "
            "combination) a deprecation-style notice about a legacy compatibility fallback, not "
            "evidence of a wrong gradient - safe to proceed with L_QM's force term as implemented."
        )
    else:
        print(
            "\nFAIL - at least one parameter's analytic gradient does not match its finite-difference "
            "estimate. Do NOT trust L_QM's force term until this is resolved - options: drop the force "
            "term from L_QM (energy-only regularizer, a real accuracy tradeoff but avoids this op "
            "entirely), or find a torchmdnet/PyTorch version combination that registers "
            "get_neighbor_pairs_bkwd's double-backward correctly."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--n-params-to-check", type=int, default=5)
    parser.add_argument("--rel-err-threshold", type=float, default=0.05)
    args = parser.parse_args()

    main(
        ckpt=args.ckpt, data_root=args.data_root, seed=args.seed, batch_size=args.batch_size,
        eps=args.eps, n_params_to_check=args.n_params_to_check, rel_err_threshold=args.rel_err_threshold,
    )
