#!/usr/bin/env python
"""Differentiable Boltzmann estimator for StABlE-style stability-aware
fine-tuning (paper/main.tex sec:q4-stable-path / sec:q4-stable-plan).

What this is for: ordinary supervised training (train_waterbox.py) only ever
sees isolated configurations with known DFT energy/forces - it never sees
what happens when the trained model is actually run forward in time. Q4's
epot-gap/force-error investigation (paper/main.tex sec:q4-epot-gap) found
that per-configuration FORCE error, not energy error, at a rollout's
starting geometry predicts how much that rollout overheats - exactly the
kind of thing a static per-config loss doesn't directly optimize against.
StABlE (Raja, Amin, Pedregosa & Krishnapriyan 2025, arXiv:2402.13984) closes
that loop: it fine-tunes the model so that MD trajectories simulated under
the model's own dynamics have the correct ensemble-average value of some
observable (this project's planned choice: the O-O/O-H/H-H radial
distribution function, since rollout_waterbox_ase.py already computes this
against a real-DFT reference for every rollout).

THE MATH, worked from scratch (not copied blind from the paper - this
project's own established discipline, see verify_zbl_units.py/
verify_periodicity.py, is to re-derive/independently check anything a
downstream training run will be trusted on):

For a Boltzmann distribution P_theta(Gamma) = exp(-U_theta(Gamma)/kT) / Z_theta
over configurations Gamma, the exact (infinite-sample) gradient of an
observable's expectation is:

    d/dtheta E_theta[g(Gamma)] = -(1/kT) * Cov_theta(g(Gamma), grad_theta U_theta(Gamma))

Derivation: d P_theta/d theta = P_theta * [-grad_theta U_theta/kT - d(log Z_theta)/d theta],
and d(log Z_theta)/d theta = -E_theta[grad_theta U_theta]/kT (standard
partition-function identity), so d P_theta/d theta =
-(1/kT) P_theta [grad_theta U_theta - E_theta[grad_theta U_theta]]; multiplying
by g(Gamma) and integrating gives the covariance form above directly. This
independently reproduces the estimator reported in Raja et al. 2025 (their
eq., quoted in paper/main.tex sec:q4-stable-path) once the exact population
covariance is replaced by ITS OWN finite-sample unbiased estimator (Bessel's
correction, dividing by N-1 rather than N):

    Cov_sample_unbiased(x, y) = (N/(N-1)) * (mean(x*y) - mean(x)*mean(y))

giving the estimator this module implements:

    E[g(Gamma)] gradient estimate = (N/(kT*(N-1))) * (mean(g)*mean(grad_theta U_theta) - mean(g*grad_theta U_theta))

IMPORTANT - this is the ON-POLICY form (samples drawn directly from the
CURRENT theta's own Boltzmann distribution), matching how StABlE's own
alternating simulate/learn loop is described (a short fresh simulation
window feeds exactly one gradient step, then replicas reset and a fresh
window is simulated under the just-updated theta - paper/main.tex
sec:q4-stable-path). This is deliberately NOT the same as DiffTRe's
(Thaler & Zavadlav 2021, the paper StABlE's estimator is built on)
production implementation, which instead reweights ONE reference
trajectory across MANY gradient steps via importance-sampling weights
w_i ~ exp(-(U_theta(Gamma_i) - U_theta_hat(Gamma_i))/kT) for computational
efficiency - checked directly against DiffTRe's paper before implementing
this, since blindly copying that scheme in would have introduced a real,
documented failure mode (effective-sample-size collapse, requiring a fresh
reference trajectory once too few importance weights remain non-negligible)
that on-policy resampling every gradient step avoids by construction. If
this module is ever extended to reuse samples across multiple gradient
steps for efficiency, that importance-weighting machinery (and the
effective-sample-size monitoring DiffTRe uses to know when to stop reusing
a stale trajectory) would need to be added back in - do not silently start
reusing samples without it.

Per-bin loss weighting for vector-valued (histogram/RDF) observables: this
module uses a plain, unweighted mean-squared-error across bins, matching
DiffTRe's own RDF/ADF loss terms directly (checked against their paper -
they weight ACROSS different observable types by magnitude, e.g. RDF vs.
pressure, but do not apply any additional per-bin variance weighting WITHIN
a single RDF-type observable). If this is ever extended to combine RDF with
a differently-scaled observable (e.g. mean bond length, in eV/Angstrom-ish
units vs. RDF's dimensionless-density units), an explicit magnitude-
balancing weight per observable (not per bin) would need to be reintroduced
the way DiffTRe does it - this module's current scope is a single
vector-valued observable.

Verification: src/verify_boltzmann_estimator.py checks this module's actual
autograd-computed gradient against an independently-computed, closed-form
ground truth (an exact finite-sum Boltzmann expectation over a small
synthetic discrete state space, not just a symbolic re-derivation) before
this is ever trusted against a real model - run it before using this module
in train_waterbox_stable.py.
"""

from __future__ import annotations

import torch


def observable_mean(g: torch.Tensor) -> torch.Tensor:
    """Sample mean of a (possibly vector-valued) observable over the sample
    dimension (dim 0). Detached - this is purely a monitoring/reporting
    quantity, never part of the backward pass (the pseudo-loss below
    recomputes its own detached copy internally rather than depending on
    this function's output being reused across the two purposes)."""
    return g.detach().mean(dim=0)


def observable_loss_value(g: torch.Tensor, g_target: torch.Tensor) -> torch.Tensor:
    """The actual (informational) StABlE observable-matching loss value:
    mean squared error between the sampled observable's estimated mean and
    its target, averaged over bins for a vector-valued observable - matches
    DiffTRe's plain per-bin sum-of-squares convention (see module
    docstring). Use this for logging/early-stopping; it is NOT the tensor
    to call .backward() on (see boltzmann_estimator_pseudo_loss)."""
    g_mean = observable_mean(g)
    g_target = g_target.detach() if isinstance(g_target, torch.Tensor) else torch.as_tensor(g_target)
    return 0.5 * torch.mean((g_mean - g_target) ** 2)


def boltzmann_estimator_pseudo_loss(
    g: torch.Tensor,
    U: torch.Tensor,
    g_target: torch.Tensor,
    kT: float,
) -> torch.Tensor:
    """Returns a scalar whose .backward() populates .grad on whatever
    parameters U depends on with d(L_obs)/d(theta), where
    L_obs = 0.5 * mean_bins[(E_theta[g] - g_target)^2] and the expectation's
    own theta-gradient is estimated via the on-policy Boltzmann/covariance
    estimator derived in this module's docstring. The returned tensor's
    VALUE is not itself meaningful (it is a construct chosen purely so that
    ordinary autograd differentiation reproduces the estimator - see
    observable_loss_value for the real loss value to log).

    g: [N] or [N, B] tensor of per-sample observable values (B = number of
       histogram bins for a vector-valued observable, e.g. RDF bins). Must
       NOT require grad through the sampling process (any grad-tracking is
       stripped via .detach() regardless) - the estimator's derivation
       depends on g being evaluated at frozen, already-sampled
       configurations, not backpropagated through however those
       configurations were generated (see module docstring: on-policy
       resampling every gradient step means the sampler itself never needs
       to be differentiable).
    U: [N] tensor of per-sample potential energies, MUST require grad and
       trace back to the model parameters being fine-tuned - obtained by a
       genuine forward pass of the model on the (otherwise frozen/detached)
       sampled configurations, exactly the same kind of single-configuration
       autograd call evaluate_waterbox.py already performs per test config.
    g_target: scalar or [B] tensor, the reference/target observable value(s)
       (e.g. the real-DFT RDF already computed by rollout_waterbox_ase.py).
    kT: Boltzmann constant times temperature, in the SAME energy units as U
       (this project's convention throughout is eV - see waterbox_data.py's
       unit-conversion transform). At 300 K, kT ~ 0.02585 eV.
    """
    if not U.requires_grad:
        raise ValueError(
            "U must require grad and trace back to the model's parameters - it should "
            "come from a fresh forward pass through the model with autograd enabled on "
            "already-sampled (otherwise frozen) configurations, not a detached or "
            "torch.no_grad() evaluation. A detached U would silently make this pseudo-loss's "
            ".backward() a no-op (zero gradient reaching the model), not raise an error on "
            "its own - this check exists specifically to fail loudly instead."
        )
    g = g.detach()

    n_samples = g.shape[0]
    if n_samples < 2:
        raise ValueError(
            f"boltzmann_estimator_pseudo_loss needs at least 2 samples for the unbiased "
            f"covariance correction N/(N-1) to be defined, got N={n_samples}."
        )
    if U.shape[0] != n_samples:
        raise ValueError(
            f"g and U must have the same number of samples along dim 0 (one observable "
            f"value and one energy per sampled configuration), got g.shape[0]={n_samples} "
            f"vs U.shape[0]={U.shape[0]}."
        )

    g_target = g_target.detach() if isinstance(g_target, torch.Tensor) else torch.as_tensor(g_target)

    g_mean = g.mean(dim=0)  # [B] or scalar, detached - no theta-dependence
    U_mean = U.mean()  # scalar, autograd-tracked back to theta
    if g.dim() == 1:
        gU_mean = (g * U).mean()  # scalar, autograd-tracked (only U carries grad)
    else:
        gU_mean = (g * U.unsqueeze(-1)).mean(dim=0)  # [B], autograd-tracked

    bessel_correction = n_samples / (n_samples - 1)
    # d E_theta[g] / d theta estimate (Raja et al. 2025's estimator, on-policy
    # form - see module docstring for the from-scratch derivation confirming
    # this exact sign/normalization). g_mean is a detached constant here, so
    # autograd differentiating this expression w.r.t. theta correctly passes
    # through only via U_mean and gU_mean, reproducing the covariance formula
    # without this module ever hand-computing a gradient itself.
    raw_estimator_pseudo = (bessel_correction / kT) * (g_mean * U_mean - gU_mean)

    # Chain rule for L_obs = 0.5 * mean_bins[(g_mean - g_target)^2]:
    # d L_obs / d theta = mean_bins[(g_mean - g_target) * d(g_mean)/d theta].
    # mismatch is a detached scaling coefficient (no additional grad path
    # through it is wanted or needed - it is evaluated at the CURRENT g_mean,
    # not something being differentiated further).
    mismatch = (g_mean - g_target).detach()
    per_bin_pseudo = mismatch * raw_estimator_pseudo
    return per_bin_pseudo.mean()
