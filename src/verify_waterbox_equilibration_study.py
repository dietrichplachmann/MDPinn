#!/usr/bin/env python
"""Local verification of pairing, failure accounting and actual ASE handoff.

The optional ASE test uses a small harmonic potential, not TorchMD-Net. It
checks the production runner with ASE's real NoseHooverChainNVT and
VelocityVerlet, verifies the boundary energy/momenta and saved trajectories,
and checks a second model receives an identical initial physical state.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import run_waterbox_equilibration_study as study


class StudyChecks(unittest.TestCase):
    def test_selection_is_model_independent_and_ordered(self):
        rows = [{"test_config_index": i, "topology_clean": i != 0,
                 "initial_flags": ["bad"] if i == 1 else [], "raw_dataset_index": 100 + i}
                for i in range(5)]
        self.assertEqual([x["raw_dataset_index"] for x in study.choose_starts(rows[::-1], 2)], [102, 103])
        with self.assertRaises(ValueError):
            study.choose_starts(rows, 4)

    def test_initial_thresholds_are_explicit_and_inclusive(self):
        row = {"hoh_angle_min_deg": 80., "intramolecular_hh_min_a": 1.2,
               "intermolecular_oo_min_a": 2.3, "oh_max_a": 1.3,
               "intermolecular_oh_min_a": 1.3}
        self.assertEqual(study.structural_flags(row), [])
        row["intermolecular_oo_min_a"] = 2.29
        self.assertEqual(study.structural_flags(row), ["intermolecular_oo_below_floor"])

    def plan(self):
        return {"settings": {"seeds": [0, 1], "configs_per_seed": 1, "velocity_seeds": [0]},
                "starts": {str(s): [{"raw_dataset_index": 100 + s}] for s in range(2)}}

    def test_pairing_aborts_errors_and_seed_aggregation(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, start, velocity, cond, directory in study.tasks(plan, root):
                p = directory / "attempt_0001"; p.mkdir(parents=True)
                regularized = cond == study.CONDITIONS[1]
                status = ("aborted" if not regularized else "complete") if seed == 0 else ("error" if regularized else "complete")
                study.write_json(p / "run_manifest.json", {
                    "train_seed": seed, "raw_dataset_index": start["raw_dataset_index"],
                    "velocity_seed": velocity, "condition": cond, "status": status,
                    "structural_survival": int(status == "complete"),
                    "nve_tail_temperature_k": 300 if regularized else 800,
                })
            result = study.aggregate(plan, root)
            self.assertEqual(result["terminal_runs"], 3)
            with (root / "paired_seed_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["structural_survival_momentum_minus_absolute"]), 1.)
            self.assertEqual(rows[0]["nve_tail_temperature_k_momentum_minus_absolute"], "")
            self.assertEqual(rows[1]["paired_terminal"], "0")
            self.assertEqual(rows[1]["structural_survival_momentum_minus_absolute"], "")

    def test_resume_uses_latest_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            for i, status in ((1, "error"), (2, "complete")):
                q = p / f"attempt_{i:04d}"; q.mkdir()
                study.write_json(q / "run_manifest.json", {"status": status})
            self.assertEqual(study.last_attempt(p)["status"], "complete")

    def test_mismatched_initial_states_are_rejected(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, start, velocity, cond, directory in study.tasks(plan, root):
                p = directory / "attempt_0001"; p.mkdir(parents=True)
                study.write_json(p / "run_manifest.json", {
                    "train_seed": seed, "raw_dataset_index": start["raw_dataset_index"],
                    "velocity_seed": velocity, "condition": cond, "status": "complete",
                    "initial_state_sha256": cond, "structural_survival": 1,
                })
            with self.assertRaisesRegex(ValueError, "initial states differ"):
                study.aggregate(plan, root)

    def test_complete_with_structural_event_is_not_survival(self):
        rows = [{"phase": "nve", "time_fs": t, "phase_time_fs": t,
                 "temperature_k": 300., "epot_ev": -t, "ekin_ev": 1. + t,
                 "etot_ev": 1.} for t in (0., 1., 2.)]
        geometry = [{"phase": "nve", "flags": "short_oo", "hoh_angle_min_deg": 90.,
                     "intramolecular_hh_min_a": 1.4, "intermolecular_oo_min_a": 1.8}]
        result = study.summarize(rows, geometry, "complete")
        self.assertEqual(result["structural_survival"], 0)
        self.assertEqual(result["nve_energy_change_ev"], 0.)
        self.assertEqual(result["nve_potential_change_ev"], -2.)

    @unittest.skipUnless(importlib.util.find_spec("ase"), "ASE unavailable; run this check in the remote environment")
    def test_real_ase_handoff_and_pairing(self):
        import numpy as np
        from ase import Atoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.io import read

        atoms = Atoms("OHHOHH", positions=[[0, 0, 0], [.96, 0, 0], [-.24, .93, 0],
                                          [3, 0, 0], [3.96, 0, 0], [2.76, .93, 0]],
                      cell=[10, 10, 10], pbc=True)
        reference = atoms.positions.copy()

        class Harmonic(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                displacement = atoms.positions - reference
                self.results = {"energy": float(.1 * np.sum(displacement ** 2)),
                                "forces": -.2 * displacement}

        fake_ase = types.ModuleType("waterbox_ase")
        fake_ase.atoms_from_waterbox_sample = lambda sample: atoms.copy()
        fake_ase.TensorNetCalculator = Harmonic
        fake_data = types.ModuleType("waterbox_data")
        fake_data.load_waterbox_dataset = lambda *args, **kwargs: None
        fake_data.random_split = lambda *args, **kwargs: None
        plan = {"settings": {"device": "cpu", "temperature_k": 300., "dt_fs": .1,
                             "nvt_ps": .001, "nve_ps": .002, "tdamp_fs": 100.,
                             "log_stride": 1, "structure_stride": 1,
                             "trajectory_stride": 2, "maximum_temperature_k": 5000.},
                "rules": study.RULES,
                "checkpoints": {f"0/{c}": {"path": "unused", "sha256": "test"} for c in study.CONDITIONS}}
        ids = np.array([0, 0, 0, 1, 1, 1]); groups = [np.arange(3), np.arange(3, 6)]
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {
                "waterbox_ase": fake_ase, "waterbox_data": fake_data}), patch.object(
                    study, "checked_calculator", side_effect=lambda *args: Harmonic()):
            results = []
            for condition in study.CONDITIONS:
                results.append(study.run_case(plan, [None], ids, groups, 0,
                    {"raw_dataset_index": 0, "test_config_index": 0}, 0, condition,
                    Path(temp) / condition))
            self.assertEqual(results[0]["initial_state_sha256"], results[1]["initial_state_sha256"])
            for result in results:
                self.assertEqual(result["status"], "complete", result.get("error"))
                self.assertTrue(result["handoff_preserved_state"])
                p = Path(result["attempt_dir"])
                with (p / "thermodynamic_history.csv").open() as handle:
                    history = list(csv.DictReader(handle))
                nvt = [r for r in history if r["phase"] == "nvt"]
                nve = [r for r in history if r["phase"] == "nve"]
                self.assertEqual(len(nvt), 11); self.assertEqual(len(nve), 21)
                for metric in ("epot_ev", "ekin_ev", "etot_ev", "temperature_k"):
                    self.assertEqual(nvt[-1][metric], nve[0][metric])
                endpoint = read(p / "nvt_final.xyz")
                first_nve = read(p / "production.xyz", index=0)
                np.testing.assert_array_equal(endpoint.positions, first_nve.positions)
                np.testing.assert_array_equal(endpoint.get_momenta(), first_nve.get_momenta())
                self.assertLess(result["nve_maximum_energy_deviation_ev"], .001)
            plan["settings"]["maximum_temperature_k"] = 1.
            aborted = study.run_case(plan, [None], ids, groups, 0,
                {"raw_dataset_index": 0, "test_config_index": 0}, 0, study.CONDITIONS[0],
                Path(temp) / "numerical_abort")
            self.assertEqual(aborted["status"], "aborted")
            self.assertEqual(aborted["structural_survival"], 0)
            self.assertIsNone(aborted["first_structural_event"])
            self.assertTrue((Path(aborted["attempt_dir"]) / "run_manifest.json").is_file())


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib unavailable")
class PlotChecks(unittest.TestCase):
    def test_energy_reference_is_reset_at_nve_handoff(self):
        import plot_waterbox_equilibration_study as plots
        run = {"therm": [
            {"phase": "nvt", "time_fs": "0", "epot_ev": "-100", "etot_ev": "-90"},
            {"phase": "nvt", "time_fs": "1000", "epot_ev": "-130", "etot_ev": "-120"},
            {"phase": "nve", "time_fs": "1000", "epot_ev": "-130", "etot_ev": "-120"},
            {"phase": "nve", "time_fs": "2000", "epot_ev": "-131", "etot_ev": "-119.808"}],
               "structure": []}
        x, y = plots.series(run, "nve_drift")
        self.assertEqual(list(x), [1., 2.])
        self.assertAlmostEqual(y[0], 0.)
        self.assertAlmostEqual(y[1], 1.)
        _, potential = plots.series(run, "potential")
        self.assertEqual(list(potential), [0., -30., -30., -31.])

    def test_aborted_pairs_remain_in_survival_but_not_thermal_comparison(self):
        import plot_waterbox_equilibration_study as plots
        a = {"manifest": {"status": "aborted", "structural_survival": 0,
                           "last_time_fs": 20., "first_structural_event": None}}
        b = {"manifest": {"status": "complete", "structural_survival": 1,
                           "first_structural_event": None}}
        runs = {(0, 100, 0, c): r for c, r in zip(study.CONDITIONS, (a, b))}
        terminal = plots.paired_runs(runs, 0)
        self.assertEqual(len(terminal), 1)
        self.assertEqual(plots.paired_runs(runs, 0, complete_only=True), [])
        self.assertEqual(plots.metric_values(terminal, "survival", 6.), ([0.], [100.]))
        self.assertEqual(plots.metric_values(terminal, "time_without_flags", 6.), ([.02], [6.]))
        self.assertEqual(plots.metric_values(terminal, "nve_tail_temperature_k", 6.), ([], []))

    def test_render_partial_study_headlessly(self):
        import plot_waterbox_equilibration_study as plots
        plan = {"settings": {"seeds": [0, 1], "configs_per_seed": 1,
                             "velocity_seeds": [0], "nvt_ps": 1., "nve_ps": 5.,
                             "temperature_k": 300.}, "rules": study.RULES}
        run = {"manifest": {"status": "running"},
               "therm": [{"phase": "nvt", "time_fs": 0., "epot_ev": -100.,
                           "ekin_ev": 7., "etot_ev": -93., "temperature_k": 300.}],
               "structure": []}
        runs = {(0, 100, 0, study.CONDITIONS[0]): run}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plots.render(plan, runs, root)
            for stem in ("seed0_trajectories", "seed1_trajectories", "paired_seed_outcomes"):
                self.assertTrue((root / "figures" / f"{stem}.png").read_bytes().startswith(b"\x89PNG"))
                self.assertTrue((root / "figures" / f"{stem}.pdf").read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
