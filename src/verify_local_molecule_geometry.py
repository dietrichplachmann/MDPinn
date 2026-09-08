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
    result = per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)

    expected = {0: 0.95, 1: 1.10}
    ok = True
    for gid, expected_len in expected.items():
        actual = result[gid]
        err = abs(actual - expected_len)
        status = "PASS" if err < 1e-6 else "FAIL"
        ok = ok and err < 1e-6
        print(f"  molecule {gid}: expected {expected_len:.4f} A, got {actual:.4f} A ({status})")
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

    result = per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)
    # h1a is placed 0.3 A from o1 across the wrapped boundary (0.2 - (9.9 - 10.0) = 0.3).
    expected_mol1 = (0.3 + 1.10) / 2
    err = abs(result[1] - expected_mol1)
    status = "PASS" if err < 1e-6 else "FAIL"
    print(f"  molecule 1 (periodic-wrapped H): expected {expected_mol1:.4f} A, got {result[1]:.4f} A ({status})")
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


def check_malformed_molecule_raises():
    """A molecule with 2 O and 1 H (not pure water) should raise, not
    silently produce a wrong number - per_molecule_mean_oh_bond_length's
    own documented assumption."""
    z = np.array([8, 8, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    group_ids = np.array([0, 0, 0])
    box_lengths = np.array([10.0, 10.0, 10.0])
    try:
        per_molecule_mean_oh_bond_length(z, positions, box_lengths, group_ids)
        print("  FAIL: expected ValueError for a non-water molecule, none raised")
        return False
    except ValueError as exc:
        print(f"  PASS: raised ValueError as expected ({exc})")
        return True


if __name__ == "__main__":
    print("Check 1: per-molecule mean O-H bond length (known ground truth)")
    ok1 = check_bond_lengths()
    print("\nCheck 2: per-molecule mean O-H bond length (periodic boundary)")
    ok2 = check_bond_length_periodicity()
    print("\nCheck 3: local_molecule_energies (value + gradient routing)")
    ok3 = check_local_molecule_energies()
    print("\nCheck 4: malformed molecule raises ValueError")
    ok4 = check_malformed_molecule_raises()

    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} ({sum([ok1, ok2, ok3, ok4])}/4)")
