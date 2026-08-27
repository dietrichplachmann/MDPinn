#!/usr/bin/env python
"""Independent correctness check for boltzmann_estimator.py before it is
ever trusted inside train_waterbox_stable.py - this project's established
discipline (see verify_zbl_units.py, verify_periodicity.py) is to check any
formula pulled from a paper against a ground truth computed a completely
different way, not just re-read the derivation and hope. This is especially
warranted here: a subtle sign or normalization bug in a gradient estimator
would not crash anything - it would silently produce plausible-looking but
wrong gradients, exactly the failure mode flagged as the single highest
risk in the StABlE implementation plan (paper/main.tex sec:q4-stable-plan).

Ground truth construction: a small synthetic discrete "toy Boltzmann
system" of K states, each with a fixed random feature vector f_i (so
U_theta(state i) = theta . f_i is exactly linear in theta - this makes the
TRUE gradient grad_theta U_theta(state i) = f_i known exactly, with no
autograd needed to know it independently) and a fixed random observable
value g_i. Because there are only K states, E_theta[g] = sum_i g_i *
softmax(-U_theta(i)/kT)_i can be computed EXACTLY (no sampling noise at
all) for any theta, and its true derivative w.r.t. theta obtained via
central finite differences on that exact expectation - a sampling-noise-free
ground truth, independent of anything in boltzmann_estimator.py.

Separately, Monte Carlo "MD sampling" is emulated by drawing N states i.i.d.
from the EXACT Boltzmann probabilities at theta0 (np.random.choice with
those probabilities as weights) - a legitimate stand-in for "N configurations
sampled from P_theta0 via real MD", since the estimator's derivation only
depends on samples being drawn from P_theta, not on how. Feeding those draws
into boltzmann_estimator_pseudo_loss and comparing the resulting
autograd-computed gradient against the finite-difference ground truth tests
both the algebra (does it match the known-correct closed form) and the
actual autograd wiring (does calling .backward() on this specific
tensor expression really reproduce it), not just the math on paper.

Usage (runs anywhere with torch, no torchmdnet/ase needed):
    python src/verify_boltzmann_estimator.py
"""

from __future__ import annotations

import numpy as np
import torch

from boltzmann_estimator import boltzmann_estimator_pseudo_loss, observable_loss_value

RNG_SEED = 0
K_STATES = 300
KT = 1.0
FINITE_DIFF_EPS = 1e-4


def _exact_expectation(theta: float, feat: np.ndarray, g: np.ndarray, kT: float) -> float:
    """E_theta[g] computed exactly over all K states (no sampling)."""
    u = theta * feat
    u = u - u.min()  # numerically stable softmax, doesn't change the distribution
    w = np.exp(-u / kT)
    p = w / w.sum()
    return float((g * p).sum())


def _finite_diff_grad(theta: float, feat: np.ndarray, g: np.ndarray, kT: float, eps: float) -> float:
    plus = _exact_expectation(theta + eps, feat, g, kT)
    minus = _exact_expectation(theta - eps, feat, g, kT)
    return (plus - minus) / (2 * eps)


def _sample_states(theta: float, feat: np.ndarray, kT: float, n: int, rng: np.random.Generator) -> np.ndarray:
    u = theta * feat
    u = u - u.min()
    w = np.exp(-u / kT)
    p = w / w.sum()
    return rng.choice(len(feat), size=n, p=p)


def test_scalar_theta_scalar_observable():
    print("=== test 1: scalar theta, scalar observable ===")
    rng = np.random.default_rng(RNG_SEED)
    feat = rng.normal(size=K_STATES)
    g_vals = feat + 0.5 * rng.normal(size=K_STATES)  # correlated with feat -> nonzero gradient
    theta0 = 0.3
    g_target = 0.7

    true_dEg_dtheta = _finite_diff_grad(theta0, feat, g_vals, KT, FINITE_DIFF_EPS)
    exact_g_mean = _exact_expectation(theta0, feat, g_vals, KT)
    print(f"  exact E_theta0[g] = {exact_g_mean:.6f}, finite-diff dE[g]/dtheta = {true_dEg_dtheta:.6f}")

    for n_samples in (2000, 20000):
        rng_mc = np.random.default_rng(123)
        idx = _sample_states(theta0, feat, KT, n_samples, rng_mc)
        f_sampled = torch.tensor(feat[idx], dtype=torch.float64)
        g_sampled = torch.tensor(g_vals[idx], dtype=torch.float64)
        theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
        U = theta * f_sampled

        pseudo = boltzmann_estimator_pseudo_loss(g_sampled, U, torch.tensor(g_target, dtype=torch.float64), KT)
        pseudo.backward()
        mc_g_mean = g_sampled.mean().item()
        expected_grad = (mc_g_mean - g_target) * true_dEg_dtheta
        got_grad = theta.grad.item()
        rel_err = abs(got_grad - expected_grad) / (abs(expected_grad) + 1e-12)
        print(f"  N={n_samples:6d}  mc_g_mean={mc_g_mean:.5f}  expected_grad={expected_grad:.6f}  "
              f"got_grad={got_grad:.6f}  rel_err={rel_err:.4%}")
        assert rel_err < 0.05, f"gradient mismatch too large at N={n_samples}: rel_err={rel_err:.4%}"

    # Unbiasedness check: average theta.grad over many independent MC draws at
    # a FIXED small N - should converge close to (true exact g_mean - g_target) * true_dEg_dtheta
    # as more trials are averaged, catching a systematic bias a single lucky
    # draw could hide.
    n_trials, n_small = 400, 500
    grads = []
    for trial in range(n_trials):
        rng_mc = np.random.default_rng(1000 + trial)
        idx = _sample_states(theta0, feat, KT, n_small, rng_mc)
        f_sampled = torch.tensor(feat[idx], dtype=torch.float64)
        g_sampled = torch.tensor(g_vals[idx], dtype=torch.float64)
        theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
        U = theta * f_sampled
        pseudo = boltzmann_estimator_pseudo_loss(g_sampled, U, torch.tensor(g_target, dtype=torch.float64), KT)
        pseudo.backward()
        grads.append(theta.grad.item())
    mean_grad = float(np.mean(grads))
    expected_grad_exact = (exact_g_mean - g_target) * true_dEg_dtheta
    rel_err = abs(mean_grad - expected_grad_exact) / (abs(expected_grad_exact) + 1e-12)
    print(f"  unbiasedness: mean over {n_trials} trials of N={n_small} = {mean_grad:.6f}, "
          f"exact-expectation target = {expected_grad_exact:.6f}, rel_err={rel_err:.4%}")
    assert rel_err < 0.03, f"estimator looks biased: rel_err={rel_err:.4%} averaged over {n_trials} trials"
    print("  PASS\n")


def test_vector_theta():
    print("=== test 2: vector theta (3 parameters) ===")
    rng = np.random.default_rng(RNG_SEED + 1)
    feat = rng.normal(size=(K_STATES, 3))  # each state has a 3-dim feature
    g_vals = feat[:, 0] - 0.5 * feat[:, 1] + 0.3 * rng.normal(size=K_STATES)
    theta0 = np.array([0.2, -0.15, 0.1])
    g_target = -0.1

    def exact_expectation_vec(theta_vec):
        u = feat @ theta_vec
        u = u - u.min()
        w = np.exp(-u / KT)
        p = w / w.sum()
        return float((g_vals * p).sum())

    true_grad = np.zeros(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = FINITE_DIFF_EPS
        true_grad[i] = (exact_expectation_vec(theta0 + step) - exact_expectation_vec(theta0 - step)) / (2 * FINITE_DIFF_EPS)
    exact_g_mean = exact_expectation_vec(theta0)
    print(f"  exact E_theta0[g] = {exact_g_mean:.6f}, finite-diff grad = {true_grad}")

    rng_mc = np.random.default_rng(456)
    u0 = feat @ theta0
    u0 = u0 - u0.min()
    p0 = np.exp(-u0 / KT)
    p0 /= p0.sum()
    idx = rng_mc.choice(K_STATES, size=30000, p=p0)
    feat_sampled = torch.tensor(feat[idx], dtype=torch.float64)
    g_sampled = torch.tensor(g_vals[idx], dtype=torch.float64)
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    U = (feat_sampled * theta).sum(dim=1)

    pseudo = boltzmann_estimator_pseudo_loss(g_sampled, U, torch.tensor(g_target, dtype=torch.float64), KT)
    pseudo.backward()
    mc_g_mean = g_sampled.mean().item()
    expected_grad = (mc_g_mean - g_target) * true_grad
    got_grad = theta.grad.numpy()
    rel_err = np.abs(got_grad - expected_grad) / (np.abs(expected_grad) + 1e-12)
    print(f"  mc_g_mean={mc_g_mean:.5f}")
    print(f"  expected_grad={expected_grad}")
    print(f"  got_grad     ={got_grad}")
    print(f"  rel_err={rel_err}")
    # Pure relative error is meaningless for components whose true gradient is
    # near zero (component 2 here barely influences g by construction) - a
    # tiny absolute Monte Carlo fluctuation around ~0 blows up the ratio
    # without indicating an actual estimator problem. Use a combined
    # absolute+relative check (np.allclose's own convention) instead.
    close = np.allclose(got_grad, expected_grad, rtol=0.06, atol=0.01)
    print(f"  allclose(rtol=0.06, atol=0.01) = {close}")
    assert close, f"vector-theta gradient mismatch too large: got={got_grad} expected={expected_grad}"
    print("  PASS\n")


def test_multibin_reduces_to_scalar():
    print("=== test 3: multi-bin observable reduces to scalar case at B=1 ===")
    rng = np.random.default_rng(RNG_SEED + 2)
    n = 5000
    theta0 = 0.4
    feat = torch.tensor(rng.normal(size=n), dtype=torch.float64)
    g_scalar = torch.tensor(rng.normal(size=n) + feat.numpy(), dtype=torch.float64)
    g_target_scalar = 0.2

    theta_a = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    U_a = theta_a * feat
    loss_a = boltzmann_estimator_pseudo_loss(g_scalar, U_a, torch.tensor(g_target_scalar, dtype=torch.float64), KT)
    loss_a.backward()

    theta_b = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    U_b = theta_b * feat
    g_vec = g_scalar.unsqueeze(-1)  # [N, 1] - a single "bin"
    g_target_vec = torch.tensor([g_target_scalar], dtype=torch.float64)
    loss_b = boltzmann_estimator_pseudo_loss(g_vec, U_b, g_target_vec, KT)
    loss_b.backward()

    print(f"  scalar-path grad = {theta_a.grad.item():.8f}")
    print(f"  B=1-vector-path grad = {theta_b.grad.item():.8f}")
    assert abs(theta_a.grad.item() - theta_b.grad.item()) < 1e-10, "B=1 vector path should exactly match scalar path"
    print("  PASS\n")


def test_multibin_against_finite_difference():
    print("=== test 4: genuinely multi-bin (B=3) observable vs finite-difference ground truth ===")
    rng = np.random.default_rng(RNG_SEED + 3)
    feat = rng.normal(size=K_STATES)
    # 3 bins, each a different (fixed) function of state index - stands in for RDF bins
    bin_fns = [
        feat,
        np.sin(feat),
        feat ** 2 - 1.0,
    ]
    g_vals = np.stack(bin_fns, axis=1)  # [K, 3]
    theta0 = 0.25
    g_target = np.array([0.1, -0.05, 0.3])

    def exact_expectation_bins(theta):
        u = theta * feat
        u = u - u.min()
        w = np.exp(-u / KT)
        p = w / w.sum()
        return (g_vals * p[:, None]).sum(axis=0)  # [3]

    exact_g_mean = exact_expectation_bins(theta0)
    true_dEg_dtheta = (exact_expectation_bins(theta0 + FINITE_DIFF_EPS) - exact_expectation_bins(theta0 - FINITE_DIFF_EPS)) / (2 * FINITE_DIFF_EPS)
    # expected d L_obs/d theta = mean_bins[(g_mean_b - target_b) * dEg_b/dtheta]
    expected_grad = float(np.mean((exact_g_mean - g_target) * true_dEg_dtheta))
    print(f"  exact E_theta0[g] per bin = {exact_g_mean}")
    print(f"  finite-diff dE[g]/dtheta per bin = {true_dEg_dtheta}")
    print(f"  expected total grad (exact-expectation mismatch) = {expected_grad:.6f}")

    rng_mc = np.random.default_rng(789)
    idx = _sample_states(theta0, feat, KT, 40000, rng_mc)
    feat_sampled = torch.tensor(feat[idx], dtype=torch.float64)
    g_sampled = torch.tensor(g_vals[idx], dtype=torch.float64)
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    U = theta * feat_sampled

    pseudo = boltzmann_estimator_pseudo_loss(g_sampled, U, torch.tensor(g_target, dtype=torch.float64), KT)
    pseudo.backward()
    mc_g_mean = g_sampled.mean(dim=0).numpy()
    expected_grad_mc = float(np.mean((mc_g_mean - g_target) * true_dEg_dtheta))
    got_grad = theta.grad.item()
    rel_err = abs(got_grad - expected_grad_mc) / (abs(expected_grad_mc) + 1e-12)
    print(f"  mc_g_mean per bin = {mc_g_mean}")
    print(f"  expected_grad (using mc g_mean, matching what the function itself uses) = {expected_grad_mc:.6f}")
    print(f"  got_grad = {got_grad:.6f}  rel_err={rel_err:.4%}")
    assert rel_err < 0.06, f"multi-bin gradient mismatch too large: rel_err={rel_err:.4%}"
    print("  PASS\n")


def test_observable_loss_value_matches_definition():
    print("=== test 5: observable_loss_value sanity ===")
    g = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([3.0, 2.0], dtype=torch.float64)
    got = observable_loss_value(g, target).item()
    g_mean = g.mean(dim=0)
    expected = 0.5 * float(((g_mean - target) ** 2).mean())
    print(f"  got={got:.6f} expected={expected:.6f}")
    assert abs(got - expected) < 1e-12
    print("  PASS\n")


def test_edge_cases():
    print("=== test 6: edge cases raise clearly, don't silently misbehave ===")
    g_ok = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    # N=1: undefined N/(N-1) correction
    try:
        U_one = theta * torch.tensor([1.0], dtype=torch.float64)
        boltzmann_estimator_pseudo_loss(torch.tensor([1.0], dtype=torch.float64), U_one, torch.tensor(0.0), KT)
        raise AssertionError("expected ValueError for N=1, did not raise")
    except ValueError as e:
        print(f"  N=1 correctly raised: {e}")

    # U without requires_grad: would silently be a no-op gradient if unchecked
    try:
        U_detached = (theta * torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)).detach()
        boltzmann_estimator_pseudo_loss(g_ok, U_detached, torch.tensor(0.0), KT)
        raise AssertionError("expected ValueError for detached U, did not raise")
    except ValueError as e:
        print(f"  detached U correctly raised: {e}")

    # Mismatched sample counts
    try:
        U_short = theta * torch.tensor([1.0, 2.0], dtype=torch.float64)
        boltzmann_estimator_pseudo_loss(g_ok, U_short, torch.tensor(0.0), KT)
        raise AssertionError("expected ValueError for shape mismatch, did not raise")
    except ValueError as e:
        print(f"  shape mismatch correctly raised: {e}")
    print("  PASS\n")


if __name__ == "__main__":
    test_scalar_theta_scalar_observable()
    test_vector_theta()
    test_multibin_reduces_to_scalar()
    test_multibin_against_finite_difference()
    test_observable_loss_value_matches_definition()
    test_edge_cases()
    print("All boltzmann_estimator.py checks passed.")
