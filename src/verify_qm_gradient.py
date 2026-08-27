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

A first real run (2026) found 4/5 parameters agreeing within 5%, one at
~8.4% - a borderline single mismatch, not the wildly-wrong-or-wrong-signed
pattern a genuinely broken double-backward would be expected to produce.
Before concluding the double-backward is actually broken, this script now
also (1) checks whether the UNPERTURBED loss itself is bit-identical
across repeated evaluations with unchanged parameters - GNN message-passing
architectures commonly rely on CUDA scatter/reduce operations that are
non-deterministic by default (this project's own code already uses
`scatter` elsewhere, e.g. molecular_zbl.py), which would inject noise into
a finite-difference check independent of any real gradient error - and (2)
automatically re-runs the finite-difference computation at several
additional eps values for any parameter that fails, since a genuine
gradient bug should stay robustly wrong across a wide range of step sizes,
while noise-limited estimation typically does not.

Usage (training box only, needs torchmdnet - NOT yet run):
    python src/verify_qm_gradient.py --ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt
"""

from __future__ import annotations

import numpy as np
import torch

from evaluate_waterbox import load_waterbox_checkpoint
from train_waterbox_stable import _qm_regularizer_loss
from waterbox_data import load_waterbox_dataset, random_split


def _finite_diff_grad(loss_fn, p, idx, eps):
    """One central-difference estimate of d(loss_fn())/d(p.view(-1)[idx]),
    restoring the original value before returning regardless of outcome."""
    with torch.no_grad():
        flat = p.view(-1)
        original = float(flat[idx].item())
        flat[idx] = original + eps
    loss_plus = float(loss_fn().item())

    with torch.no_grad():
        flat[idx] = original - eps
    loss_minus = float(loss_fn().item())

    with torch.no_grad():
        flat[idx] = original
    return (loss_plus - loss_minus) / (2 * eps)


def main(ckpt, data_root="./data", seed=42, batch_size=2, eps=1e-4, n_params_to_check=5, rel_err_threshold=0.05,
         eps_sweep=(1e-2, 1e-3, 1e-5)):
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

    print("Checking whether the unperturbed loss is even deterministic across repeated calls "
          "(same params, same batch) - establishes a noise floor before trusting any single "
          "finite-difference comparison...")
    # No torch.no_grad() here even though nothing below needs gradients:
    # _qm_regularizer_loss internally forces torch.enable_grad() regardless
    # of the calling context (the model's derivative=True forward pass
    # needs it), so an outer no_grad() would be silently ineffective, not
    # actually save anything - only .item() is read below, .backward() is
    # never called on these.
    repeat_losses = [float(loss_at_current_params().item()) for _ in range(5)]
    noise_spread = max(repeat_losses) - min(repeat_losses)
    print(f"  5 repeated evaluations: {repeat_losses}")
    print(f"  spread = {noise_spread:.6e} (0.0 would mean fully deterministic)")

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

    print(f"\nChecking {len(candidates)} parameters (largest-magnitude analytic gradient per tensor, eps={eps})...")
    all_ok = True
    mismatches = []
    for name, p, idx, analytic_grad in candidates:
        # p.view(-1), not p.flatten() or p.detach().flatten(): .view()
        # requires true memory-sharing with p and raises loudly if that's
        # not possible, rather than silently falling back to a disconnected
        # copy - the whole point of this script is establishing ground
        # truth, so a perturbation that silently doesn't touch the real
        # parameter would corrupt the check without any error at all.
        numerical_grad = _finite_diff_grad(loss_at_current_params, p, idx, eps)
        rel_err = abs(numerical_grad - analytic_grad) / (abs(analytic_grad) + abs(numerical_grad) + 1e-12)
        ok = rel_err < rel_err_threshold
        all_ok = all_ok and ok
        # Noise-floor context: if the unperturbed loss itself varies by
        # noise_spread between calls, the finite-difference formula
        # (dividing a loss difference by 2*eps) amplifies that same spread
        # into a gradient-estimate uncertainty of roughly this magnitude -
        # a mismatch smaller than or comparable to this is NOT evidence of
        # a real gradient bug, it's the check's own noise floor.
        implied_noise_floor = noise_spread / (2 * eps)
        print(
            f"  {name}[{idx}]: analytic={analytic_grad:.6e}  numerical={numerical_grad:.6e}  "
            f"rel_err={rel_err:.4%}  {'OK' if ok else 'MISMATCH'}  "
            f"(noise-floor-implied grad uncertainty ~{implied_noise_floor:.3e})"
        )
        if not ok:
            mismatches.append((name, p, idx, analytic_grad))

    for name, p, idx, analytic_grad in mismatches:
        print(f"\nMismatch on {name}[{idx}] - sweeping eps to check whether the disagreement is "
              f"robust (real bug) or shrinks/is noise-floor-limited (finite-difference artifact)...")
        for sweep_eps in sorted(set(eps_sweep) | {eps}):
            numerical_grad = _finite_diff_grad(loss_at_current_params, p, idx, sweep_eps)
            rel_err = abs(numerical_grad - analytic_grad) / (abs(analytic_grad) + abs(numerical_grad) + 1e-12)
            implied_noise_floor = noise_spread / (2 * sweep_eps)
            print(
                f"    eps={sweep_eps:.0e}: numerical={numerical_grad:.6e}  rel_err={rel_err:.4%}  "
                f"noise-floor-implied grad uncertainty ~{implied_noise_floor:.3e}"
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
            "\nFAIL at the default eps - see the eps-sweep output above for each mismatching parameter "
            "before deciding what this means. If rel_err shrinks substantially at larger eps and/or is "
            "comparable to (or smaller than) the noise-floor-implied uncertainty at that eps, this is "
            "most likely finite-difference/CUDA-nondeterminism noise, not a genuine double-backward bug - "
            "safe to proceed. If rel_err stays large and robust across every eps tried, well above the "
            "noise floor at each one, treat this as a real gradient bug: do NOT trust L_QM's force term "
            "until resolved - options: drop the force term from L_QM (energy-only regularizer, a real "
            "accuracy tradeoff but avoids this op entirely), or find a torchmdnet/PyTorch version "
            "combination that registers get_neighbor_pairs_bkwd's double-backward correctly."
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
