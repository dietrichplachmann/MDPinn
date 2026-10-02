#!/usr/bin/env python
"""Diagnose analytic stress and gate hydrostatic finite-difference pressure.

This is a mandatory remote-GPU gate before any NPT trajectory is trusted.
The analytic box-gradient path is retained here as a diagnostic because it
fails for these checkpoints.  Isotropic NPT instead uses a separately gated
central finite-difference estimate of the hydrostatic stress trace.  The
learned WaterBox checkpoints were trained on energies and forces, not stress
labels, so passing this script verifies implementation consistency only; it
does not establish that the predicted pressure is physically accurate.

Example:
    python src/verify_waterbox_stress.py \
      --data-seed 0 \
      --ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed0/best_model.ckpt
"""

from __future__ import annotations

import argparse

import numpy as np

from waterbox_ase import TensorNetCalculator, atoms_from_waterbox_sample
from waterbox_data import load_waterbox_dataset, random_split


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
EV_A3_TO_BAR = 1.602176634e6


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


def verify_hydrostatic_fallback(
    atoms, checkpoint: str, calculator_epsilon: float, reference_epsilon: float,
    rtol: float, atol_ev_a3: float,
) -> bool:
    """Gate the exact stress path used by IsotropicMTKNPT.

    The independent reference uses the existing strained-Atoms helper rather
    than TensorNetCalculator's internal tensor scaling.  Agreement therefore
    checks calculator wiring, sign, normalization, wrapping, Voigt projection,
    and convergence between the production and larger reference strain.
    """
    wrapped = atoms.copy()
    wrapped.wrap()
    wrapped.calc = TensorNetCalculator(
        checkpoint,
        stress_mode="hydrostatic_fd",
        stress_fd_epsilon=calculator_epsilon,
    )
    fallback = np.asarray(wrapped.get_stress(voigt=True), dtype=float)
    fallback_trace = float(fallback[:3].sum())
    reference_at_calculator_epsilon = finite_difference_stress_trace(
        wrapped, checkpoint, calculator_epsilon
    )
    reference_at_larger_epsilon = finite_difference_stress_trace(
        wrapped, checkpoint, reference_epsilon
    )

    expected_projection = np.asarray(
        [fallback_trace / 3.0] * 3 + [0.0, 0.0, 0.0]
    )
    projection_pass = bool(np.allclose(fallback, expected_projection, rtol=0.0, atol=1e-12))
    trace_tolerance = 3.0 * atol_ev_a3 + rtol * abs(reference_at_larger_epsilon)
    wiring_pass = abs(fallback_trace - reference_at_calculator_epsilon) <= trace_tolerance
    convergence_pass = abs(fallback_trace - reference_at_larger_epsilon) <= trace_tolerance

    raw_reference = finite_difference_stress_trace(atoms, checkpoint, reference_epsilon)
    image_pass = abs(raw_reference - reference_at_larger_epsilon) <= trace_tolerance
    passed = projection_pass and wiring_pass and convergence_pass and image_pass

    print("\n=== hydrostatic finite-difference NPT gate ===")
    print(f"calculator strain epsilon: {calculator_epsilon:g}")
    print(f"reference strain epsilon: {reference_epsilon:g}")
    print("Returned isotropic stress (eV/A^3):")
    for label, value in zip(VOIGT_LABELS, fallback):
        print(f"  {label}: {value:+.8e}")
    print(
        f"  calculator trace: {fallback_trace:+.8e}\n"
        f"  independent trace at calculator epsilon: "
        f"{reference_at_calculator_epsilon:+.8e}\n"
        f"  independent trace at reference epsilon: "
        f"{reference_at_larger_epsilon:+.8e}\n"
        f"  raw-image trace at reference epsilon: {raw_reference:+.8e}"
    )
    potential_pressure = -fallback_trace / 3.0
    print(
        f"  implied potential pressure: {potential_pressure:+.8e} eV/A^3 "
        f"({potential_pressure * EV_A3_TO_BAR:+.1f} bar)"
    )
    print(f"isotropic Voigt projection: {'PASS' if projection_pass else 'FAIL'}")
    print(f"calculator/reference wiring: {'PASS' if wiring_pass else 'FAIL'}")
    print(f"finite-strain convergence: {'PASS' if convergence_pass else 'FAIL'}")
    print(f"periodic-image invariance: {'PASS' if image_pass else 'FAIL'}")
    print(f"HYDROSTATIC FD NPT GATE: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--data-seed", type=int, required=True,
        help=(
            "Dataset split seed. Pass the checkpoint's training seed so the gate and "
            "rollout use the same held-out starting geometry; there is deliberately no "
            "legacy seed-42 default."
        ),
    )
    parser.add_argument("--test-config-index", type=int, default=0)
    parser.add_argument("--epsilons", default="0.0003,0.001,0.003,0.01")
    parser.add_argument(
        "--coordinate-mode", choices=("raw", "wrapped", "both"), default="both",
        help="Compare raw dataset image coordinates with the same atoms wrapped into the primary cell.",
    )
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--atol-ev-a3", type=float, default=0.003)
    parser.add_argument(
        "--stress-fd-epsilon", type=float, default=0.003,
        help="Finite strain used by the production hydrostatic-pressure calculator.",
    )
    parser.add_argument(
        "--reference-epsilon", type=float, default=0.01,
        help="Larger finite strain used to check convergence of the production estimate.",
    )
    parser.add_argument(
        "--gate-only", action="store_true",
        help="Skip the already-diagnosed analytic tensor and run only the production NPT gate.",
    )
    args = parser.parse_args()

    dataset = load_waterbox_dataset(args.data_root)
    _, _, test_data = random_split(dataset, seed=args.data_seed)
    atoms = atoms_from_waterbox_sample(test_data[args.test_config_index])
    epsilons = [float(item) for item in args.epsilons.split(",")]
    if not args.gate_only:
        modes = ("raw", "wrapped") if args.coordinate_mode == "both" else (args.coordinate_mode,)
        results = {
            mode: evaluate_coordinate_mode(
                atoms, args.ckpt, mode, epsilons, args.rtol, args.atol_ev_a3
            )
            for mode in modes
        }
        print("\n=== analytic-stress diagnosis ===")
        if any(component_passed for component_passed, _ in results.values()):
            print("ANALYTIC STRESS: PASS for at least one tested coordinate mode and strain scale.")
        else:
            print(
                "ANALYTIC STRESS: FAIL. This path is disabled for NPT; the production gate below "
                "tests the numerical hydrostatic replacement."
            )

    fallback_passed = verify_hydrostatic_fallback(
        atoms,
        args.ckpt,
        args.stress_fd_epsilon,
        args.reference_epsilon,
        args.rtol,
        args.atol_ev_a3,
    )
    if not fallback_passed:
        raise SystemExit("Hydrostatic finite-difference pressure verification failed. Do not run NPT.")
    print(
        "PASS: the finite-difference hydrostatic pressure implementation is internally consistent. "
        "This does not establish physical pressure accuracy."
    )


if __name__ == "__main__":
    main()
