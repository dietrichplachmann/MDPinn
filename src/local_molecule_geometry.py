#!/usr/bin/env python
"""Pure numpy/torch pieces of the localized per-molecule Boltzmann estimator
(paper/main.tex sec:q4-stable-local-estimator-scope) - split out from
local_molecule_observable.py specifically so this core logic stays
exercisable with synthetic arrays on this Windows checkout (no torchmdnet
installed), matching diagnose_short_range_collapse.py's and
waterbox_langevin.py's own established split between "what's locally
testable" and "what needs the training box." See
local_molecule_observable.py's module docstring for the full rationale
(why a local, per-molecule observable/energy is the fix for the
global-RDF-vs-localized-instability mismatch, and why the captured atomic
energy is deliberately GNN-only/ZBL-exclusive).

See src/verify_local_molecule_geometry.py for the synthetic test of both
functions below.
"""

from __future__ import annotations

import numpy as np
import torch

from diagnose_short_range_collapse import pairwise_min_image_distances


def local_molecule_energies(captured_atomic_energies, group_ids_tensor, num_molecules):
    """captured_atomic_energies: AtomicEnergyCapture.captured straight from
    one forward call - shape (n_atoms, 1) or (n_atoms,), still attached to
    theta (differentiable). group_ids_tensor: (n_atoms,) long tensor on the
    same device, atom -> molecule id in [0, num_molecules), same atom
    ordering as the z/pos the model was just called with (see
    local_molecule_observable.py's docstring on why that alignment is
    expected but not yet confirmed against a running model). Returns a
    (num_molecules,) tensor: U_theta(gamma) = sum of each molecule's own
    atomic energies - index_add_ is differentiable and correctly routes
    each output element's gradient back to exactly the atoms that
    contributed to it, the same scatter-style idiom
    physics_losses.per_fragment_momentum_loss already uses and this project
    already trusts, here for energy instead of force.
    """
    x = captured_atomic_energies.reshape(-1)
    out = torch.zeros(num_molecules, dtype=x.dtype, device=x.device)
    out.index_add_(0, group_ids_tensor, x)
    return out


def per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids):
    """(N,) z, (N,3) positions, (3,) orthorhombic box_lengths, (N,) group_ids
    (diagnose_short_range_collapse.molecule_group_ids' own convention,
    contiguous ids in [0, num_molecules), one O + two H per group for pure
    water - the same assumption diagnose_short_range_collapse.py already
    makes, USUALLY true) -> (values, valid_group_ids): g(gamma) per
    molecule that actually resolved to exactly 1 O + 2 H (the mean of that
    molecule's own two O-H distances, Raja et al. 2025 Section 4.4's exact
    water observable), periodicity-aware via this project's own established
    minimum-image convention (pairwise_min_image_distances - not
    ase.geometry.get_distances, matching every other distance computation
    in this codebase), and the corresponding molecule ids `values` maps to.

    A molecule NOT resolving to exactly 1 O + 2 H is skipped (not raised as
    an error) - confirmed on a real training-box smoke test to happen on
    real DFT reference configurations (not just a hypothetical edge case):
    distance-threshold bond inference (structural_metrics.infer_bonds,
    shared by molecule_group_ids) occasionally misses a real but thermally
    stretched O-H bond, splitting one water molecule into two degenerate
    groups - evaluate_waterbox.py's own summarize_molecule_groups already
    anticipated exactly this ("asserting every water molecule works out to
    exactly {O: 1, H: 2}, not some other split caused by a missed/spurious
    bond") without ever having been strictly enforced before this function.
    Given ~12,800 molecules sampled for a typical reference average
    (n_reference_configs=200 x 64 molecules), skipping the rare degenerate
    one is a negligible approximation, not a bug to work around by raising.

    values[i] corresponds to molecule valid_group_ids[i], NOT necessarily
    to molecule i, once any molecule has been skipped - callers pairing
    this with a per-molecule energy tensor (e.g. local_molecule_energies,
    computed for ALL num_molecules regardless of composition, since a pure
    energy scatter-sum doesn't care about elemental identity) MUST index
    that tensor by valid_group_ids too, to keep g and U paired by molecule
    identity - see train_waterbox_stable.py's own call site.

    Pure geometry, no gradient - g never carries a grad path through the
    sampler (boltzmann_estimator.py's own requirement, already established
    for the whole-box RDF observable this replaces).

    A Python loop over molecules (typically 64 for this system) is used
    here, not a fully vectorized scatter - this runs once per collected
    snapshot, not once per training batch/hot-loop iteration (this
    project's own "vectorize anything called once per batch" lesson does
    not apply at this scale: 64 simple index/mean operations is negligible
    next to the GPU forward pass this snapshot's energy computation already
    costs).
    """
    dist = pairwise_min_image_distances(positions, box_lengths)
    z = np.asarray(z)
    group_ids = np.asarray(group_ids)
    o_mask = z == 8
    h_mask = z == 1
    num_molecules = int(group_ids.max()) + 1 if group_ids.size else 0
    values, valid_ids = [], []
    n_skipped = 0
    for gid in range(num_molecules):
        atoms = np.where(group_ids == gid)[0]
        o_idx = atoms[o_mask[atoms]]
        h_idx = atoms[h_mask[atoms]]
        if o_idx.size != 1 or h_idx.size != 2:
            n_skipped += 1
            continue
        values.append(dist[o_idx[0], h_idx].mean())
        valid_ids.append(gid)
    if n_skipped:
        print(
            f"WARNING: per_molecule_mean_oh_bond_length skipped {n_skipped}/{num_molecules} "
            "molecule(s) that did not resolve to exactly 1 O + 2 H (most likely a bond-inference "
            "miss on a thermally stretched O-H distance, not a code bug - see this function's "
            "docstring)."
        )
    return np.asarray(values, dtype=np.float64), np.asarray(valid_ids, dtype=np.int64)
