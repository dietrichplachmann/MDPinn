#!/usr/bin/env python
"""Synthetic verification of local_molecule_geometry.py's two pure
functions (paper/main.tex sec:q4-stable-local-estimator-scope) - no
torchmdnet/ase needed, runs on this Windows checkout.

Run: python src/verify_local_molecule_geometry.py
"""

from __future__ import annotations

import numpy as np
import torch

from local_molecule_geometry import local_molecule_energies, per_molecule_mean_oh_bond_length


def _synthetic_two_molecule_box(box_length=10.0):
    """2 water molecules, hand-placed so their O-H bond lengths are exact,
    known values (0.95 A and 1.10 A) - a real, checkable ground truth,
    not just "runs without crashing." Order deliberately interleaved
    (molecule 1's atoms, then molecule 0's atoms) so a bug that silently
    assumes atoms are stored in contiguous per-molecule blocks would be
    caught, matching structural_metrics.infer_molecule_groups' own
    documented "does NOT assume contiguous blocks" guarantee.
    """
    # Molecule 0: O at origin, H1 at 0.95 A, H2 at 0.95 A (different direction).
    o0 = np.array([1.0, 1.0, 1.0])
    h0a = o0 + np.array([0.95, 0.0, 0.0])
    h0b = o0 + np.array([0.0, 0.95, 0.0])
    # Molecule 1: O elsewhere, H1/H2 at 1.10 A - placed far enough from
    # molecule 0 that no cross-molecule pair is anywhere near bond-length
    # range, so a same_molecule-grouping bug would show up as a wildly
    # wrong bond length, not a subtly-off one.
    o1 = np.array([5.0, 5.0, 5.0])
    h1a = o1 + np.array([1.10, 0.0, 0.0])
    h1b = o1 + np.array([0.0, 0.0, 1.10])

    # Interleaved storage order: H1a, O1, H0a, H0b, O0, H1b.
    positions = np.stack([h1a, o1, h0a, h0b, o0, h1b])
    z = np.array([1, 8, 1, 1, 8, 1])
    # group_ids must match this same interleaved order: molecule 1's atoms
    # get group id 1, molecule 0's atoms get group id 0.
    group_ids = np.array([1, 1, 0, 0, 0, 1])
    box_lengths = np.array([box_length, box_length, box_length])
    return z, positions, group_ids, box_lengths


def check_bond_lengths():
    z, positions, group_ids, box_lengths = _synthetic_two_molecule_box()
    values, valid_ids = per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)
    by_id = dict(zip(valid_ids.tolist(), values.tolist()))

    expected = {0: 0.95, 1: 1.10}
    ok = len(valid_ids) == 2
    for gid, expected_len in expected.items():
        actual = by_id.get(gid)
        err = abs(actual - expected_len) if actual is not None else float("inf")
        status = "PASS" if err < 1e-6 else "FAIL"
        ok = ok and err < 1e-6
        print(f"  molecule {gid}: expected {expected_len:.4f} A, got {actual} A ({status})")
    return ok


def check_bond_length_periodicity():
    """Same two molecules, but molecule 1 straddles a periodic boundary
    (its two H atoms placed on opposite sides of the box edge from O) -
    confirms per_molecule_mean_oh_bond_length uses minimum-image distance,
    not raw Euclidean, matching this project's own hard-learned periodicity
    lesson (CLAUDE.md: "any utility... should be assumed non-periodic-only
    until explicitly checked")."""
    box_length = 10.0
    o0 = np.array([1.0, 1.0, 1.0])
    h0a = o0 + np.array([0.95, 0.0, 0.0])
    h0b = o0 + np.array([0.0, 0.95, 0.0])
    # Molecule 1: O right at the edge, one H wrapped to the OTHER edge -
    # raw Euclidean distance would be close to box_length, minimum-image
    # should correctly recover ~1.10 A.
    o1 = np.array([0.2, 5.0, 5.0])
    h1a = np.array([9.9, 5.0, 5.0])  # wraps to 0.2 - (-0.1) = 0.3 away via minimum image... see below
    h1b = o1 + np.array([0.0, 0.0, 1.10])

    positions = np.stack([o0, h0a, h0b, o1, h1a, h1b])
    z = np.array([8, 1, 1, 8, 1, 1])
    group_ids = np.array([0, 0, 0, 1, 1, 1])
    box_lengths = np.array([box_length, box_length, box_length])

    values, valid_ids = per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)
    by_id = dict(zip(valid_ids.tolist(), values.tolist()))
    # h1a is placed 0.3 A from o1 across the wrapped boundary (0.2 - (9.9 - 10.0) = 0.3).
    expected_mol1 = (0.3 + 1.10) / 2
    actual = by_id.get(1)
    err = abs(actual - expected_mol1) if actual is not None else float("inf")
    status = "PASS" if err < 1e-6 else "FAIL"
    print(f"  molecule 1 (periodic-wrapped H): expected {expected_mol1:.4f} A, got {actual} A ({status})")
    return err < 1e-6


def check_local_molecule_energies():
    """4 atoms, 2 molecules (ids [0, 1, 0, 1]), known per-atom energies -
    confirms index_add_ correctly sums by molecule regardless of storage
    order, and that gradients flow back correctly (each molecule's summed
    energy's gradient w.r.t. its own atoms' energies should be exactly 1,
    and exactly 0 w.r.t. the other molecule's atoms - the defining property
    a scatter-sum must have for this to be a valid U_theta(gamma))."""
    atomic_energies = torch.tensor([1.5, 2.5, 3.5, 4.5], requires_grad=True)
    group_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    num_molecules = 2

    out = local_molecule_energies(atomic_energies, group_ids, num_molecules)
    expected = torch.tensor([1.5 + 3.5, 2.5 + 4.5])
    value_ok = torch.allclose(out, expected)
    print(f"  summed energies: expected {expected.tolist()}, got {out.tolist()} ({'PASS' if value_ok else 'FAIL'})")

    # Gradient check: d(out[0])/d(atomic_energies) should be [1, 0, 1, 0].
    grad = torch.autograd.grad(out[0], atomic_energies)[0]
    expected_grad = torch.tensor([1.0, 0.0, 1.0, 0.0])
    grad_ok = torch.allclose(grad, expected_grad)
    print(f"  gradient routing: expected {expected_grad.tolist()}, got {grad.tolist()} ({'PASS' if grad_ok else 'FAIL'})")

    return value_ok and grad_ok


def check_malformed_molecule_skipped():
    """A molecule with 2 O and 1 H (not pure water - e.g. a bond-inference
    miss splitting one real water molecule into two degenerate groups, the
    exact failure a real training-box smoke test hit on real DFT reference
    data) should be SKIPPED, not raised or silently miscomputed -
    per_molecule_mean_oh_bond_length's own documented, corrected behavior
    (an earlier version of this function raised ValueError here instead;
    that was too strict for something this project has now confirmed
    happens on real data, not just a hypothetical edge case)."""
    z = np.array([8, 8, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    group_ids = np.array([0, 0, 0])
    box_lengths = np.array([10.0, 10.0, 10.0])
    values, valid_ids = per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)
    ok = values.size == 0 and valid_ids.size == 0
    print(f"  values={values.tolist()}, valid_ids={valid_ids.tolist()} ({'PASS' if ok else 'FAIL'})")
    return ok


def check_mixed_valid_and_degenerate_molecules():
    """3 molecules: two normal (1 O + 2 H each), one degenerate (2 O + 1 H,
    e.g. from a missed bond) - confirms the degenerate one is skipped while
    the two valid ones are still correctly computed AND correctly
    identified by valid_ids (not silently index-shifted), the exact
    scenario train_waterbox_stable.py's real loop depends on to keep g and
    U paired by molecule identity when a skip happens."""
    o0 = np.array([1.0, 1.0, 1.0])
    h0a = o0 + np.array([0.95, 0.0, 0.0])
    h0b = o0 + np.array([0.0, 0.95, 0.0])
    o1 = np.array([5.0, 5.0, 5.0])
    h1a = o1 + np.array([1.10, 0.0, 0.0])
    h1b = o1 + np.array([0.0, 0.0, 1.10])
    # Degenerate molecule 1 (id 1) sits BETWEEN the two valid ones (ids 0
    # and 2) in storage/id order, so a naive index-shift bug (skip one,
    # forget to renumber) would misassign molecule 2's value to id 1.
    o_bad1 = np.array([8.0, 8.0, 8.0])
    o_bad2 = o_bad1 + np.array([1.0, 0.0, 0.0])
    h_bad = o_bad1 + np.array([0.5, 0.5, 0.0])

    positions = np.stack([o0, h0a, h0b, o_bad1, o_bad2, h_bad, o1, h1a, h1b])
    z = np.array([8, 1, 1, 8, 8, 1, 8, 1, 1])
    group_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    box_lengths = np.array([10.0, 10.0, 10.0])

    values, valid_ids = per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)
    by_id = dict(zip(valid_ids.tolist(), values.tolist()))
    ok = set(valid_ids.tolist()) == {0, 2}
    ok = ok and abs(by_id[0] - 0.95) < 1e-6 and abs(by_id[2] - 1.10) < 1e-6
    print(f"  valid_ids={sorted(valid_ids.tolist())} (expected [0, 2]), by_id={by_id} ({'PASS' if ok else 'FAIL'})")
    return ok


if __name__ == "__main__":
    print("Check 1: per-molecule mean O-H bond length (known ground truth)")
    ok1 = check_bond_lengths()
    print("\nCheck 2: per-molecule mean O-H bond length (periodic boundary)")
    ok2 = check_bond_length_periodicity()
    print("\nCheck 3: local_molecule_energies (value + gradient routing)")
    ok3 = check_local_molecule_energies()
    print("\nCheck 4: malformed (all-degenerate) molecule skipped, not raised")
    ok4 = check_malformed_molecule_skipped()
    print("\nCheck 5: mixed valid + degenerate molecules - skip realigns by molecule id, not index-shifted")
    ok5 = check_mixed_valid_and_degenerate_molecules()

    results = [ok1, ok2, ok3, ok4, ok5]
    all_ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} ({sum(results)}/{len(results)})")
