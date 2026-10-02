#!/usr/bin/env python
"""Verify TensorNetCalculator stress against finite cell-strain energies.

This is a mandatory remote-GPU gate before any NPT trajectory is trusted.
The learned WaterBox checkpoints were trained on energies and forces, not
stress labels, so this script verifies implementation consistency only; it
does not establish that the predicted pressure is physically accurate.

Example:
    python src/verify_waterbox_stress.py \
      --ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed0/best_model.ckpt
"""

from __future__ import annotations

import argparse

import numpy as np

from waterbox_ase import TensorNetCalculator, atoms_from_waterbox_sample
from waterbox_data import load_waterbox_dataset, random_split


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")


def deformation_from_voigt(component: int, amount: float) -> np.ndarray:
    deformation = np.eye(3)
    if component < 3:
        deformation[component, component] += amount
    else:
        i, j = ((1, 2), (0, 2), (0, 1))[component - 3]
        deformation[i, j] += 0.5 * amount
        deformation[j, i] += 0.5 * amount
    return deformation


def strained_copy(atoms, component: int, amount: float):
    strained = atoms.copy()
    deformation = deformation_from_voigt(component, amount)
    strained.set_cell(np.asarray(atoms.cell) @ deformation.T, scale_atoms=False)
    strained.set_positions(np.asarray(atoms.positions) @ deformation.T)
    return strained


def finite_difference_stress(atoms, checkpoint: str, epsilon: float) -> np.ndarray:
    volume = atoms.get_volume()
    calculator = TensorNetCalculator(checkpoint)
    values = []
    for component in range(6):
        plus = strained_copy(atoms, component, epsilon)
        minus = strained_copy(atoms, component, -epsilon)
        plus.calc = calculator
        minus.calc = calculator
        e_plus = plus.get_potential_energy()
        e_minus = minus.get_potential_energy()
        values.append((e_plus - e_minus) / (2.0 * epsilon * volume))
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--test-config-index", type=int, default=0)
    parser.add_argument("--epsilons", default="0.0003,0.001,0.003")
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--atol-ev-a3", type=float, default=0.003)
    args = parser.parse_args()

    dataset = load_waterbox_dataset(args.data_root)
    _, _, test_data = random_split(dataset, seed=args.data_seed)
    atoms = atoms_from_waterbox_sample(test_data[args.test_config_index])
    atoms.calc = TensorNetCalculator(args.ckpt)
    analytic = np.asarray(atoms.get_stress(voigt=True), dtype=float)

    print("Analytic stress (eV/A^3):")
    for label, value in zip(VOIGT_LABELS, analytic):
        print(f"  {label}: {value:+.8e}")

    passed = False
    for epsilon in (float(item) for item in args.epsilons.split(",")):
        numeric = finite_difference_stress(atoms, args.ckpt, epsilon)
        difference = np.abs(analytic - numeric)
        tolerance = args.atol_ev_a3 + args.rtol * np.abs(numeric)
        this_passed = bool(np.all(difference <= tolerance))
        passed = passed or this_passed
        print(f"epsilon={epsilon:g}: {'PASS' if this_passed else 'FAIL'}")
        for label, ana, num, diff in zip(VOIGT_LABELS, analytic, numeric, difference):
            print(f"  {label}: analytic={ana:+.8e} numeric={num:+.8e} abs_diff={diff:.3e}")

    if not passed:
        raise SystemExit(
            "Stress verification failed at every epsilon. Do not run NPT; inspect box-gradient "
            "support, float32 cancellation, and strain conventions first."
        )
    print("PASS: at least one finite-difference scale agrees with analytic stress.")


if __name__ == "__main__":
    main()
