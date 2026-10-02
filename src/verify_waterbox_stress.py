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


def hydrostatic_strained_copy(atoms, amount: float):
    """Scale all three cell directions and fractional coordinates together."""
    strained = atoms.copy()
    deformation = (1.0 + amount) * np.eye(3)
    strained.set_cell(np.asarray(atoms.cell) @ deformation.T, scale_atoms=False)
    strained.set_positions(np.asarray(atoms.positions) @ deformation.T)
    return strained


def finite_difference_stress_trace(atoms, checkpoint: str, epsilon: float) -> float:
    """Return dE/d(isotropic strain)/V, equal to trace(stress)."""
    volume = atoms.get_volume()
    calculator = TensorNetCalculator(checkpoint)
    plus = hydrostatic_strained_copy(atoms, epsilon)
    minus = hydrostatic_strained_copy(atoms, -epsilon)
    plus.calc = calculator
    minus.calc = calculator
    return float(
        (plus.get_potential_energy() - minus.get_potential_energy())
        / (2.0 * epsilon * volume)
    )


def describe_coordinate_images(atoms, label: str) -> None:
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    outside = np.any((scaled < 0.0) | (scaled >= 1.0), axis=1)
    print(
        f"{label} fractional-coordinate range: "
        f"[{scaled.min():+.4f}, {scaled.max():+.4f}], "
        f"atoms outside primary cell={int(outside.sum())}/{len(atoms)}"
    )


def evaluate_coordinate_mode(
    atoms, checkpoint: str, mode: str, epsilons: list[float], rtol: float,
    atol_ev_a3: float,
) -> tuple[bool, bool]:
    atoms = atoms.copy()
    if mode == "wrapped":
        atoms.wrap()
    atoms.calc = TensorNetCalculator(checkpoint)
    analytic = np.asarray(atoms.get_stress(voigt=True), dtype=float)

    print(f"\n=== coordinate mode: {mode} ===")
    describe_coordinate_images(atoms, mode)
    print("Analytic stress (eV/A^3):")
    for label, value in zip(VOIGT_LABELS, analytic):
        print(f"  {label}: {value:+.8e}")

    component_passed = False
    hydrostatic_passed = False
    analytic_trace = float(analytic[:3].sum())
    for epsilon in epsilons:
        numeric = finite_difference_stress(atoms, checkpoint, epsilon)
        difference = np.abs(analytic - numeric)
        tolerance = atol_ev_a3 + rtol * np.abs(numeric)
        this_passed = bool(np.all(difference <= tolerance))
        component_passed = component_passed or this_passed
        print(f"epsilon={epsilon:g}: {'PASS' if this_passed else 'FAIL'}")
        for label, ana, num, diff in zip(VOIGT_LABELS, analytic, numeric, difference):
            print(f"  {label}: analytic={ana:+.8e} numeric={num:+.8e} abs_diff={diff:.3e}")

        numeric_trace = finite_difference_stress_trace(atoms, checkpoint, epsilon)
        trace_difference = abs(analytic_trace - numeric_trace)
        trace_tolerance = 3.0 * atol_ev_a3 + rtol * abs(numeric_trace)
        trace_pass = trace_difference <= trace_tolerance
        hydrostatic_passed = hydrostatic_passed or trace_pass
        print(
            f"  hydrostatic trace: {'PASS' if trace_pass else 'FAIL'} "
            f"analytic={analytic_trace:+.8e} numeric={numeric_trace:+.8e} "
            f"abs_diff={trace_difference:.3e}"
        )
    return component_passed, hydrostatic_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--test-config-index", type=int, default=0)
    parser.add_argument("--epsilons", default="0.0003,0.001,0.003,0.01")
    parser.add_argument(
        "--coordinate-mode", choices=("raw", "wrapped", "both"), default="both",
        help="Compare raw dataset image coordinates with the same atoms wrapped into the primary cell.",
    )
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--atol-ev-a3", type=float, default=0.003)
    args = parser.parse_args()

    dataset = load_waterbox_dataset(args.data_root)
    _, _, test_data = random_split(dataset, seed=args.data_seed)
    atoms = atoms_from_waterbox_sample(test_data[args.test_config_index])
    epsilons = [float(item) for item in args.epsilons.split(",")]
    modes = ("raw", "wrapped") if args.coordinate_mode == "both" else (args.coordinate_mode,)
    results = {
        mode: evaluate_coordinate_mode(
            atoms, args.ckpt, mode, epsilons, args.rtol, args.atol_ev_a3
        )
        for mode in modes
    }

    if "raw" in results and "wrapped" in results:
        raw_component, _ = results["raw"]
        wrapped_component, wrapped_hydro = results["wrapped"]
        print("\n=== interpretation gate ===")
        if not raw_component and wrapped_component:
            print(
                "Wrapped coordinates pass while raw coordinates fail: periodic image placement "
                "contaminates the analytic strain derivative. The calculator should wrap only "
                "its stress-evaluation copy before NPT is enabled."
            )
        elif not wrapped_component:
            print(
                "Wrapped coordinates still fail the six-component check: coordinate images are "
                "not the complete explanation. Do not enable analytic-stress NPT; the next step "
                "is to test/implement a finite-difference hydrostatic pressure path."
            )
        if wrapped_hydro and not wrapped_component:
            print(
                "The wrapped hydrostatic trace passes even though the full tensor fails. This is "
                "sufficient evidence to investigate a hydrostatic finite-difference fallback for "
                "the isotropic barostat, but it is not a pass for the current calculator."
            )

    passed = any(component_passed for component_passed, _ in results.values())
    if not passed:
        raise SystemExit(
            "Stress verification failed at every epsilon and coordinate mode. Do not run NPT."
        )
    print("PASS: at least one coordinate mode and finite-difference scale agrees with analytic stress.")


if __name__ == "__main__":
    main()
