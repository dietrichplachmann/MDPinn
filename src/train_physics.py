#!/usr/bin/env python
"""
Physics-informed training with optional delta-learning.

This extends standard TorchMD-NET training by adding:
- momentum conservation regularization,
- optional NVE drift regularization on short trajectories.

Chemist view:
- Base supervised loss fits reference E and F (or residuals in delta mode).
- Physics losses are soft constraints that bias training toward trajectories
  with better dynamical behavior (less spurious drift / symmetry violation).
"""

import json
from pathlib import Path
from collections import Counter
from statistics import median

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

from baseline_potential import (
    calibrate_reference_energy_offset_eV,
    load_reference_energy_offset_eV,
    reference_energy_forces_batched,
)
from data_splits import contiguous_split
from physics_losses import (
    momentum_symmetry_loss,
    nve_loss_from_trajectory,
    nve_loss_with_kinetic_energy,
    build_trajectory_batch,
    periodic_bc_loss_improved,
    get_atomic_masses,
)
from rollout_nve import FORCE_TO_ACCEL, kinetic_energy
from training_history import MetricHistoryCallback


# PyTorch 2.7 checkpoint compatibility.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


class DeltaLNNP(LNNP):
    """Convert absolute labels to residual labels when delta-learning is enabled.

    This mirrors `train_standard.py` so both training modes use the same
    physical semantics for delta-learning.
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.delta_learning = bool(hparams.get("delta_learning", False))
        self.baseline_molecule = str(hparams.get("baseline_molecule", hparams.get("molecule", "aspirin")))
        self.baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
        self.baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
        self.baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))
        self.baseline_energy_offset = float(
            hparams.get("baseline_energy_offset_eV", load_reference_energy_offset_eV(self.baseline_molecule))
        )

    def data_transform(self, batch):
        batch = super().data_transform(batch)
        if not self.delta_learning:
            return batch

        U_ref, F_ref = reference_energy_forces_batched(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            molecule=self.baseline_molecule,
            box_l=batch.box if "box" in batch else None,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
            energy_offset_eV=self.baseline_energy_offset,
        )

        if hasattr(batch, "y") and batch.y is not None:
            y = batch.y
            if y.ndim == 1:
                y = y.unsqueeze(1)
            batch.y = (y.squeeze(-1) - U_ref).unsqueeze(1)

        if hasattr(batch, "neg_dy") and batch.neg_dy is not None:
            batch.neg_dy = batch.neg_dy - F_ref

        return batch


class PhysicsInformedLNNP(DeltaLNNP):
    """LNNP with extra physics losses added inside `step` during training.

    Interpretation of added terms:
    - momentum loss: discourages net translational/rotational force imbalance.
    - NVE loss: discourages potential-energy drift along short trajectory segments.
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        self.momentum_weight = float(hparams.get("momentum_weight", 0.01))
        self.nve_weight = float(hparams.get("nve_weight", 0.01))
        self.pbc_weight = float(hparams.get("pbc_weight", 0.0))

        self.traj_length = int(hparams.get("traj_length", 100))
        self.nve_freq = int(hparams.get("nve_freq", 50))
        self.nve_warmup_epochs = int(hparams.get("nve_warmup_epochs", 5))
        self.nve_ramp_epochs = int(hparams.get("nve_ramp_epochs", 20))
        self.nve_relative = bool(hparams.get("nve_relative", False))
        self.nve_relative_eps = float(hparams.get("nve_relative_eps", 1e-6))
        self.nve_drift_scale_eV = float(hparams.get("nve_drift_scale_eV", 0.1))
        self.nve_per_atom = bool(hparams.get("nve_per_atom", True))
        # `total_energy` is the recommended default for MD because it constrains
        # the physically relevant Hamiltonian drift rather than potential-only drift.
        self.nve_loss_mode = str(hparams.get("nve_loss_mode", "total_energy"))
        # Keep the training-side trajectory timestep aligned with rollout/eval.
        self.nve_dt_fs = float(hparams.get("nve_dt_fs", 0.5))
        self._last_nve_diagnostics = {}

        self.trajectory_dataset = None
        self.trajectory_start_indices = []
        self.train_batch_counter = 0
        self.validation_dataset = None
        self.validation_eval_indices = []
        self.validation_rollout_start_indices = []
        self.validation_rollout_steps = int(hparams.get("val_rollout_steps", 250))
        self.validation_rollout_count = int(hparams.get("val_rollout_count", 6))
        self.validation_rollout_energy_log_stride = max(1, int(hparams.get("val_rollout_energy_log_stride", 10)))
        self.validation_rollout_failure_penalty = float(hparams.get("val_rollout_failure_penalty", 50.0))
        self.validation_rollout_max_score = float(hparams.get("val_rollout_max_score", 1e6))
        self.train_rollout_probe_start_indices = []
        self.train_rollout_probe_steps = int(hparams.get("train_rollout_probe_steps", 100))
        self.train_rollout_probe_count = int(hparams.get("train_rollout_probe_count", 3))
        self.train_rollout_probe_energy_log_stride = max(1, int(hparams.get("train_rollout_probe_energy_log_stride", 10)))
        self.rollout_probe_dataset = None

    def _effective_nve_weight(self):
        """Return epoch-scheduled NVE weight (warmup + linear ramp)."""
        base = self.nve_weight
        epoch = int(getattr(self, "current_epoch", 0))

        if epoch < self.nve_warmup_epochs:
            return 0.0

        if self.nve_ramp_epochs <= 0:
            return base

        ramp_pos = epoch - self.nve_warmup_epochs + 1
        scale = min(1.0, max(0.0, ramp_pos / self.nve_ramp_epochs))
        return base * scale

    def _predict_absolute_energy(self, z, pos, batch):
        """Return absolute potential energy for NVE loss.

        In absolute mode: U_abs = U_model
        In delta mode:    U_abs = U_ref + DeltaU_model
        """
        # `self.model` is the underlying TorchMD-Net energy model.
        out = self.model(z, pos, batch=batch)
        if isinstance(out, tuple):
            out = out[0]

        # Absolute-mode checkpoint already predicts total U(R).
        if not self.delta_learning:
            return out

        # Delta-mode checkpoint predicts DeltaU(R); reconstruct total U_hyb(R).
        u_ref, _ = reference_energy_forces_batched(
            z=z,
            pos=pos,
            batch=batch,
            molecule=self.baseline_molecule,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
            energy_offset_eV=self.baseline_energy_offset,
        )
        return out.squeeze(-1) + u_ref

    def _predict_absolute_energy_forces(self, z, pos, batch):
        """Return absolute energy and forces for the current live model."""
        pos_req = pos.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = self(z, pos_req, batch=batch)

        if isinstance(out, tuple) and len(out) >= 2:
            energy_pred, force_pred = out[0], out[1]
        else:
            raise RuntimeError("PhysicsInformedLNNP forward did not return force predictions.")

        if self.delta_learning:
            u_ref, f_ref = reference_energy_forces_batched(
                z=z,
                pos=pos_req,
                batch=batch,
                molecule=self.baseline_molecule,
                epsilon_eV=self.baseline_eps,
                sigma_A=self.baseline_sigma,
                r_cut_A=self.baseline_cutoff,
                energy_offset_eV=self.baseline_energy_offset,
            )
            energy_pred = energy_pred.squeeze(-1) + u_ref
            force_pred = force_pred + f_ref

        return energy_pred.squeeze().detach(), force_pred.detach()

    def _absolute_force_from_prediction(self, z, pos, batch, force_pred, box=None):
        """Convert a residual force prediction into the deployed hybrid force."""
        if not self.delta_learning:
            return force_pred

        _, f_ref = reference_energy_forces_batched(
            z=z,
            pos=pos,
            batch=batch,
            molecule=self.baseline_molecule,
            box_l=box,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
            energy_offset_eV=self.baseline_energy_offset,
        )
        return force_pred + f_ref.to(device=force_pred.device, dtype=force_pred.dtype)

    def step(self, batch, loss_fn_list, stage):
        """Compute default LNNP loss, then add physics terms during train stage."""
        total_loss = super().step(batch, loss_fn_list, stage)

        if stage != "train":
            return total_loss

        try:
            supervised_loss = total_loss
            # Recompute model force prediction for physics penalties.
            # We need force vectors explicitly because momentum symmetry is a
            # force-level constraint.
            batch.pos = batch.pos.clone().detach().requires_grad_(True)
            _, neg_dy = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
            )
            box = batch.box if "box" in batch else None
            force_for_physics = neg_dy

            loss_momentum = torch.tensor(0.0, device=self.device)
            loss_nve = torch.tensor(0.0, device=self.device)
            loss_pbc = torch.tensor(0.0, device=self.device)

            if self.momentum_weight > 0:
                force_for_physics = self._absolute_force_from_prediction(
                    batch.z,
                    batch.pos,
                    batch.batch,
                    neg_dy,
                    box=box,
                )
                # Apply momentum symmetry per molecule, then average.
                unique_batches = torch.unique(batch.batch)
                for mol_idx in unique_batches:
                    mask = batch.batch == mol_idx
                    loss_momentum += momentum_symmetry_loss(batch.pos[mask], force_for_physics[mask])
                loss_momentum = loss_momentum / len(unique_batches)

            effective_nve_weight = self._effective_nve_weight()
            self._last_nve_diagnostics = {}
            if effective_nve_weight > 0 and self.train_batch_counter % self.nve_freq == 0:
                # NVE is expensive, so it is sampled every nve_freq steps.
                # For short development sweeps, avoid setting this so high that
                # the physics term almost never fires.
                loss_nve = self._compute_nve_loss(self.train_batch_counter)

            if self.pbc_weight > 0 and hasattr(batch, "box") and batch.box is not None:
                # Apply periodicity regularization per graph when box vectors are available.
                unique_batches = torch.unique(batch.batch)
                for mol_idx in unique_batches:
                    mask = batch.batch == mol_idx
                    box_l = self._extract_box_lengths(batch.box, int(mol_idx.item()))
                    if box_l is None:
                        continue

                    local_batch = torch.zeros(int(mask.sum().item()), dtype=torch.long, device=self.device)
                    loss_pbc += periodic_bc_loss_improved(
                        self.model,
                        batch.pos[mask],
                        batch.z[mask],
                        box_l,
                        neg_dy[mask],
                        local_batch,
                    )
                loss_pbc = loss_pbc / len(unique_batches)

            self.train_batch_counter += 1

            weighted_momentum = self.momentum_weight * loss_momentum
            weighted_nve = effective_nve_weight * loss_nve
            weighted_pbc = self.pbc_weight * loss_pbc

            self.log("train_loss_momentum", loss_momentum, on_step=False, on_epoch=True)
            self.log("train_loss_nve", loss_nve, on_step=False, on_epoch=True)
            self.log("train_loss_pbc", loss_pbc, on_step=False, on_epoch=True)
            self.log("train_loss_momentum_raw", loss_momentum, on_step=False, on_epoch=True)
            self.log("train_loss_nve_raw", loss_nve, on_step=False, on_epoch=True)
            self.log("train_loss_pbc_raw", loss_pbc, on_step=False, on_epoch=True)
            self.log("train_loss_momentum_weighted", weighted_momentum, on_step=False, on_epoch=True)
            self.log("train_loss_nve_weighted", weighted_nve, on_step=False, on_epoch=True)
            self.log("train_loss_pbc_weighted", weighted_pbc, on_step=False, on_epoch=True)
            self.log("train_nve_weight_effective", effective_nve_weight, on_step=False, on_epoch=True)
            nve_diag = getattr(self, "_last_nve_diagnostics", {})
            for key, value in nve_diag.items():
                self.log(f"train_nve_{key}", value, on_step=False, on_epoch=True)

            physics_loss = weighted_momentum + weighted_nve + weighted_pbc
            denom = torch.abs(supervised_loss.detach()) + 1e-12
            self.log("train_physics_to_supervised_ratio", physics_loss.detach() / denom, on_step=False, on_epoch=True)
            self.log("train_nve_to_supervised_ratio", weighted_nve.detach() / denom, on_step=False, on_epoch=True)
            self.log("train_momentum_to_supervised_ratio", weighted_momentum.detach() / denom, on_step=False, on_epoch=True)
            total_loss = total_loss + physics_loss
            self.log("train_total_with_physics", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        except Exception as exc:
            print(f"Warning: physics loss computation failed: {exc}")

        return total_loss

    def _run_model_rollout(self, dataset, start_idx, steps, dt_fs, energy_log_stride):
        energy_log_stride = max(1, int(energy_log_stride))
        s0 = dataset[start_idx]
        s1 = dataset[start_idx + 1]

        z = s0.z.to(self.device)
        x = s0.pos.to(self.device).float().clone()
        x1 = s1.pos.to(self.device).float()
        masses = get_atomic_masses(z).to(self.device)
        v = ((x1 - x) / dt_fs).clone()

        batch = torch.zeros(z.shape[0], dtype=torch.long, device=self.device)
        U0, F0 = self._predict_absolute_energy_forces(z, x, batch)
        if (
            not torch.isfinite(U0).all()
            or not torch.isfinite(F0).all()
            or F0.abs().max().item() > 1e6
        ):
            return {
                "failed": True,
                "fail_reason": "initial_force_or_energy_numerical_blowup",
                "series": {"step": [], "drift": []},
                "final_drift": None,
                "max_abs_drift": None,
            }
        a = (F0 * FORCE_TO_ACCEL) / masses.view(-1, 1)
        E0 = kinetic_energy(masses, v) + U0
        if not torch.isfinite(E0).all():
            return {
                "failed": True,
                "fail_reason": "initial_force_or_energy_numerical_blowup",
                "series": {"step": [], "drift": []},
                "final_drift": None,
                "max_abs_drift": None,
            }

        series = {"step": [], "drift": []}
        failed = False
        fail_reason = None

        for step in range(1, steps + 1):
            v_half = v + 0.5 * dt_fs * a
            x = x + dt_fs * v_half

            if torch.isnan(x).any() or torch.isinf(x).any() or x.abs().max().item() > 1e4:
                failed = True
                fail_reason = "position_numerical_blowup"
                break

            U, F = self._predict_absolute_energy_forces(z, x, batch)
            if (
                not torch.isfinite(U).all()
                or not torch.isfinite(F).all()
                or F.abs().max().item() > 1e6
            ):
                failed = True
                fail_reason = "force_or_energy_numerical_blowup"
                break
            a_new = (F * FORCE_TO_ACCEL) / masses.view(-1, 1)
            v = v_half + 0.5 * dt_fs * a_new
            a = a_new

            if step % energy_log_stride == 0 or step == steps:
                K = kinetic_energy(masses, v)
                E = K + U
                drift = E - E0
                series["step"].append(step)
                series["drift"].append(float(drift.item()))
                if torch.isnan(E) or torch.isinf(E):
                    failed = True
                    fail_reason = "energy_numerical_blowup"
                    break

        return {
            "failed": failed,
            "fail_reason": fail_reason,
            "series": series,
            "final_drift": series["drift"][-1] if series["drift"] else None,
            "max_abs_drift": max((abs(d) for d in series["drift"]), default=None),
        }

    @staticmethod
    def _median_or_none(values):
        return None if not values else float(median(values))

    def _evaluate_rollout_group(self, dataset, start_indices, count, steps, energy_log_stride):
        if dataset is None or count <= 0:
            return None
        if not start_indices:
            return {
                "count": 0,
                "success_count": 0,
                "failure_rate": 1.0,
                "median_mean_abs_drift_eV": self.validation_rollout_max_score,
                "median_max_abs_drift_eV": self.validation_rollout_max_score,
                "median_abs_final_drift_eV": self.validation_rollout_max_score,
                "fail_reasons": {"no_valid_rollout_start": 1},
                "score": self.validation_rollout_max_score,
            }

        chosen = start_indices[:count]
        energy_log_stride = max(1, int(energy_log_stride))
        fail_reasons = Counter()
        mean_abs_drifts = []
        max_abs_drifts = []
        final_abs_drifts = []

        for start_idx in chosen:
            try:
                rollout = self._run_model_rollout(
                    dataset=dataset,
                    start_idx=int(start_idx),
                    steps=steps,
                    dt_fs=self.nve_dt_fs,
                    energy_log_stride=energy_log_stride,
                )
            except Exception:
                fail_reasons["rollout_exception"] += 1
                continue
            if rollout["failed"]:
                fail_reasons[rollout["fail_reason"] or "unknown"] += 1
                continue

            drifts = rollout["series"].get("drift", [])
            if not drifts:
                continue
            mean_abs_drifts.append(float(sum(abs(d) for d in drifts) / len(drifts)))
            max_abs_drifts.append(float(rollout["max_abs_drift"]))
            final_abs_drifts.append(float(abs(rollout["final_drift"])))

        failure_rate = float((len(chosen) - len(mean_abs_drifts)) / len(chosen))
        median_mean_abs_drift = self._median_or_none(mean_abs_drifts)
        median_max_abs_drift = self._median_or_none(max_abs_drifts)
        median_abs_final_drift = self._median_or_none(final_abs_drifts)
        if mean_abs_drifts:
            score = (
                median_mean_abs_drift
                + 0.5 * median_max_abs_drift
                + self.validation_rollout_failure_penalty * failure_rate
            )
        else:
            score = self.validation_rollout_max_score
            median_mean_abs_drift = self.validation_rollout_max_score
            median_max_abs_drift = self.validation_rollout_max_score
            median_abs_final_drift = self.validation_rollout_max_score

        return {
            "count": len(chosen),
            "success_count": len(mean_abs_drifts),
            "failure_rate": failure_rate,
            "median_mean_abs_drift_eV": median_mean_abs_drift,
            "median_max_abs_drift_eV": median_max_abs_drift,
            "median_abs_final_drift_eV": median_abs_final_drift,
            "fail_reasons": dict(fail_reasons),
            "score": float(score),
        }

    def _evaluate_static_guardrails(self):
        if self.validation_dataset is None or not self.validation_eval_indices:
            return None

        energy_abs_err = []
        energy_sq_err = []
        force_abs_err = []
        force_sq_err = []
        force_count = 0

        for idx in self.validation_eval_indices:
            sample = self.validation_dataset[int(idx)].to(self.device)
            batch = torch.zeros(sample.z.shape[0], dtype=torch.long, device=self.device)
            energy_pred, force_pred = self._predict_absolute_energy_forces(sample.z, sample.pos.float(), batch)
            energy_true = sample.y.squeeze()
            force_true = sample.neg_dy

            e_diff = float((energy_pred - energy_true).item())
            energy_abs_err.append(abs(e_diff))
            energy_sq_err.append(e_diff * e_diff)

            f_diff = (force_pred - force_true).reshape(-1)
            force_abs_err.append(float(torch.abs(f_diff).sum().item()))
            force_sq_err.append(float((f_diff * f_diff).sum().item()))
            force_count += int(f_diff.numel())

        return {
            "energy_mae": float(sum(energy_abs_err) / len(energy_abs_err)),
            "energy_rmse": float((sum(energy_sq_err) / len(energy_sq_err)) ** 0.5),
            "force_mae": float(sum(force_abs_err) / force_count),
            "force_rmse": float((sum(force_sq_err) / force_count) ** 0.5),
        }

    def on_validation_epoch_end(self):
        if self.trainer is None or self.trainer.sanity_checking:
            return

        try:
            static_metrics = self._evaluate_static_guardrails()
            if static_metrics is not None:
                self.log("val_energy_mae", static_metrics["energy_mae"], on_step=False, on_epoch=True)
                self.log("val_energy_rmse", static_metrics["energy_rmse"], on_step=False, on_epoch=True)
                self.log("val_force_mae", static_metrics["force_mae"], on_step=False, on_epoch=True)
                self.log("val_force_rmse", static_metrics["force_rmse"], on_step=False, on_epoch=True)

            rollout_metrics = self._evaluate_rollout_group(
                dataset=self.validation_dataset,
                start_indices=self.validation_rollout_start_indices,
                count=self.validation_rollout_count,
                steps=self.validation_rollout_steps,
                energy_log_stride=self.validation_rollout_energy_log_stride,
            )
            if rollout_metrics is not None:
                self.log("val_rollout_score", rollout_metrics["score"], on_step=False, on_epoch=True, prog_bar=True)
                self.log("val_rollout_success_count", float(rollout_metrics["success_count"]), on_step=False, on_epoch=True)
                self.log("val_rollout_failure_rate", rollout_metrics["failure_rate"], on_step=False, on_epoch=True, prog_bar=True)
                self.log("val_rollout_median_mean_abs_drift_eV", rollout_metrics["median_mean_abs_drift_eV"], on_step=False, on_epoch=True)
                self.log("val_rollout_median_max_abs_drift_eV", rollout_metrics["median_max_abs_drift_eV"], on_step=False, on_epoch=True)
                self.log("val_rollout_median_abs_final_drift_eV", rollout_metrics["median_abs_final_drift_eV"], on_step=False, on_epoch=True)
                self.log("val_rollout_fail_position_numerical_blowup_count", float(rollout_metrics["fail_reasons"].get("position_numerical_blowup", 0)), on_step=False, on_epoch=True)
                self.log("val_rollout_fail_energy_numerical_blowup_count", float(rollout_metrics["fail_reasons"].get("energy_numerical_blowup", 0)), on_step=False, on_epoch=True)
                self.log("val_rollout_fail_force_or_energy_numerical_blowup_count", float(rollout_metrics["fail_reasons"].get("force_or_energy_numerical_blowup", 0)), on_step=False, on_epoch=True)
                self.log("val_rollout_fail_initial_force_or_energy_numerical_blowup_count", float(rollout_metrics["fail_reasons"].get("initial_force_or_energy_numerical_blowup", 0)), on_step=False, on_epoch=True)
                self.log("val_rollout_fail_exception_count", float(rollout_metrics["fail_reasons"].get("rollout_exception", 0)), on_step=False, on_epoch=True)
                self.log("val_rollout_fail_no_valid_start_count", float(rollout_metrics["fail_reasons"].get("no_valid_rollout_start", 0)), on_step=False, on_epoch=True)
        except Exception as exc:
            print(f"Warning: rollout-aware validation metrics failed: {exc}")

    def on_train_epoch_end(self):
        if self.trainer is None or self.trainer.sanity_checking:
            return

        try:
            probe_metrics = self._evaluate_rollout_group(
                dataset=self.rollout_probe_dataset,
                start_indices=self.train_rollout_probe_start_indices,
                count=self.train_rollout_probe_count,
                steps=self.train_rollout_probe_steps,
                energy_log_stride=self.train_rollout_probe_energy_log_stride,
            )
            if probe_metrics is not None:
                self.log("train_short_rollout_mean_abs_drift_eV", probe_metrics["median_mean_abs_drift_eV"], on_step=False, on_epoch=True)
                self.log("train_short_rollout_max_abs_drift_eV", probe_metrics["median_max_abs_drift_eV"], on_step=False, on_epoch=True)
                self.log("train_short_rollout_failure_rate", probe_metrics["failure_rate"], on_step=False, on_epoch=True)
                self.log("train_short_rollout_success_count", float(probe_metrics["success_count"]), on_step=False, on_epoch=True)
        except Exception as exc:
            print(f"Warning: short-rollout probe metrics failed: {exc}")

    def _compute_nve_loss(self, batch_idx):
        """Compute trajectory drift penalty on absolute energy.

        Important: this always evaluates absolute energy drift.
        In delta mode that means baseline + learned residual, matching the
        deployed hybrid potential rather than just DeltaU alone.
        """
        self._last_nve_diagnostics = {}
        if self.trajectory_dataset is None or not self.trajectory_start_indices:
            return torch.tensor(0.0, device=self.device)

        start_idx = self.trajectory_start_indices[(batch_idx * 137) % len(self.trajectory_start_indices)]
        traj_batch = build_trajectory_batch(self.trajectory_dataset, start_idx, self.traj_length, self.device)

        # Wrap absolute-energy predictor to match expected callable signature
        # expected by nve_loss_from_trajectory(...).
        def abs_energy_model(z, pos, batch):
            return self._predict_absolute_energy(z, pos, batch=batch)

        try:
            if self.nve_loss_mode == "total_energy":
                masses = get_atomic_masses(traj_batch["Z"]).to(self.device)
                loss, stats = nve_loss_with_kinetic_energy(
                    abs_energy_model,
                    traj_batch,
                    self.device,
                    masses=masses,
                    dt=self.nve_dt_fs,
                    relative=self.nve_relative,
                    eps=self.nve_relative_eps,
                    drift_scale_eV=self.nve_drift_scale_eV,
                    per_atom=self.nve_per_atom,
                    return_stats=True,
                )
                self._last_nve_diagnostics = stats
                return loss

            if self.nve_loss_mode != "potential_only":
                raise ValueError(
                    f"Unsupported nve_loss_mode='{self.nve_loss_mode}'. "
                    "Expected 'total_energy' or 'potential_only'."
                )

            loss, stats = nve_loss_from_trajectory(
                abs_energy_model,
                traj_batch,
                self.device,
                relative=self.nve_relative,
                eps=self.nve_relative_eps,
                dt=self.nve_dt_fs,
                drift_scale_eV=self.nve_drift_scale_eV,
                per_atom=self.nve_per_atom,
                return_stats=True,
            )
            self._last_nve_diagnostics = stats
            return loss
        except Exception as exc:
            print(f"Warning: NVE loss failed: {exc}")
            self._last_nve_diagnostics = {}
            return torch.tensor(0.0, device=self.device)

    def _extract_box_lengths(self, box, graph_idx):
        """Extract orthorhombic box lengths (Lx,Ly,Lz) for one graph if available."""
        if box is None:
            return None

        try:
            # Supported shapes commonly seen in batched data:
            # - (B, 3, 3): full box vectors per graph
            # - (3, 3): single box matrix
            # - (B, 3): lengths per graph
            # - (3,): single set of lengths
            if box.dim() == 3:
                box_g = box[graph_idx]
                return torch.linalg.norm(box_g, dim=1)
            if box.dim() == 2:
                if box.shape[0] == 3 and box.shape[1] == 3:
                    return torch.linalg.norm(box, dim=1)
                if box.shape[1] == 3:
                    return box[graph_idx]
            if box.dim() == 1 and box.numel() == 3:
                return box
        except Exception:
            return None

        return None


def _contiguous_train_starts(train_subset, traj_length):
    """Return start indices whose full trajectory window stays inside the train split.

    `random_split` returns a Subset with shuffled sample indices, which is fine for
    supervised regression but not for NVE windows that require consecutive frames.
    This helper finds windows on the underlying MD17 trajectory where every frame in
    the window belongs to the training subset.
    """
    if not hasattr(train_subset, "indices"):
        return []

    train_indices = sorted(int(idx) for idx in train_subset.indices)
    train_index_set = set(train_indices)
    starts = []
    for start_idx in train_indices:
        end_idx = start_idx + traj_length
        if all(frame_idx in train_index_set for frame_idx in range(start_idx, end_idx)):
            starts.append(start_idx)
    return starts


def _contiguous_subset_starts(subset, required_span=2):
    """Return starts whose next `required_span-1` frames stay inside the subset."""
    if not hasattr(subset, "indices"):
        return []

    subset_indices = sorted(int(idx) for idx in subset.indices)
    subset_index_set = set(subset_indices)
    starts = []
    for start_idx in subset_indices:
        end_idx = start_idx + required_span
        if all(frame_idx in subset_index_set for frame_idx in range(start_idx, end_idx)):
            starts.append(start_idx)
    return starts


def _evenly_sample_indices(indices, max_count):
    if not indices or max_count <= 0:
        return []
    if len(indices) <= max_count:
        return [int(idx) for idx in indices]
    stride = max(1, len(indices) // max_count)
    return [int(idx) for idx in indices[::stride][:max_count]]


def train_physics_informed_model(
    dataset="MD17",
    molecule="aspirin",
    batch_size=32,
    num_epochs=100,
    lr=1e-4,
    model_type="tensornet",
    save_dir="checkpoints/physics_informed",
    log_dir="logs/physics_informed",
    force_weight=0.95,
    energy_weight=0.05,
    momentum_weight=0.01,
    nve_weight=0.01,
    pbc_weight=0.0,
    traj_length=100,
    nve_freq=50,
    nve_warmup_epochs=5,
    nve_ramp_epochs=20,
    nve_relative=False,
    nve_relative_eps=1e-6,
    nve_drift_scale_eV=0.1,
    nve_per_atom=True,
    nve_loss_mode="total_energy",
    nve_dt_fs=0.5,
    delta_learning=False,
    baseline_epsilon_eV=0.01,
    baseline_sigma_A=1.0,
    baseline_cutoff_A=5.0,
    embedding_dimension=256,
    num_layers=6,
    num_rbf=64,
    checkpoint_name="best_model",
    train_loss="mse_loss",
    train_loss_arg=None,
    weight_decay=0.0,
    lr_patience=15,
    lr_min=1e-7,
    lr_factor=0.8,
    num_workers=4,
    seed=42,
    trainer_callbacks=None,
    trainer_kwargs=None,
    val_rollout_steps=250,
    val_rollout_count=6,
    val_rollout_energy_log_stride=10,
    val_rollout_failure_penalty=50.0,
    val_rollout_max_score=1e6,
    val_static_eval_count=128,
    train_rollout_probe_steps=100,
    train_rollout_probe_count=3,
    train_rollout_probe_energy_log_stride=10,
):
    """Train physics-informed model with optional delta-learning targets.

    Practical meaning:
    - Standard terms fit reference data.
    - Added physics terms reduce unphysical behavior between data points.
    - Delta mode keeps the same constraints, but applied to the hybrid model.
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Physics-Informed Training")
    print("=" * 70)
    print(f"dataset={dataset}, molecule={molecule}, model={model_type}")
    print(f"delta_learning={delta_learning}")

    if dataset != "MD17":
        raise NotImplementedError(f"Dataset '{dataset}' is not implemented in train_physics.py (supported: MD17).")

    pl.seed_everything(seed, workers=True)

    full_dataset = MD17(root="./data", molecules=molecule)

    train_data, val_data, test_data = contiguous_split(full_dataset)

    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    baseline_energy_offset_eV = float(load_reference_energy_offset_eV(molecule))
    if delta_learning:
        baseline_energy_offset_eV = float(
            calibrate_reference_energy_offset_eV(
                molecule=molecule,
                dataset=dataset,
                data_root="./data",
                epsilon_eV=baseline_epsilon_eV,
                sigma_A=baseline_sigma_A,
                r_cut_A=baseline_cutoff_A,
            )
        )
        print(f"calibrated_baseline_energy_offset_eV={baseline_energy_offset_eV}")

    model_args = {
        "delta_learning": bool(delta_learning),
        "baseline_molecule": molecule,
        "baseline_epsilon_eV": float(baseline_epsilon_eV),
        "baseline_sigma_A": float(baseline_sigma_A),
        "baseline_cutoff_A": float(baseline_cutoff_A),
        "baseline_energy_offset_eV": baseline_energy_offset_eV,
        "model": model_type,
        "prior_model": None,
        "output_model": "Scalar",
        "load_model": None,
        "remove_ref_energy": False,
        "train_loss": train_loss,
        "train_loss_arg": train_loss_arg,
        "charge": False,
        "spin": False,
        "precision": 32,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "embedding_dimension": int(embedding_dimension),
        "num_layers": int(num_layers),
        "num_rbf": int(num_rbf),
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "max_z": 100,
        "max_num_neighbors": 128,
        "derivative": True,
        "lr": lr,
        "lr_patience": lr_patience,
        "lr_min": lr_min,
        "lr_factor": lr_factor,
        "lr_warmup_steps": 0,
        "weight_decay": weight_decay,
        "y_weight": energy_weight,
        "neg_dy_weight": force_weight,
        "ema_alpha_y": 1.0,
        "ema_alpha_neg_dy": 1.0,
        "momentum_weight": momentum_weight,
        "nve_weight": nve_weight,
        "pbc_weight": pbc_weight,
        "traj_length": traj_length,
        "nve_freq": nve_freq,
        "nve_warmup_epochs": nve_warmup_epochs,
        "nve_ramp_epochs": nve_ramp_epochs,
        "nve_relative": bool(nve_relative),
        "nve_relative_eps": float(nve_relative_eps),
        "nve_drift_scale_eV": float(nve_drift_scale_eV),
        "nve_per_atom": bool(nve_per_atom),
        "nve_loss_mode": str(nve_loss_mode),
        "nve_dt_fs": float(nve_dt_fs),
        "val_rollout_steps": int(val_rollout_steps),
        "val_rollout_count": int(val_rollout_count),
        "val_rollout_energy_log_stride": int(val_rollout_energy_log_stride),
        "val_rollout_failure_penalty": float(val_rollout_failure_penalty),
        "val_rollout_max_score": float(val_rollout_max_score),
        "train_rollout_probe_steps": int(train_rollout_probe_steps),
        "train_rollout_probe_count": int(train_rollout_probe_count),
        "train_rollout_probe_energy_log_stride": int(train_rollout_probe_energy_log_stride),
        "atom_filter": -1,
        "reduce_op": "add",
        "equivariance_invariance_group": "O(3)",
        "box_vecs": None,
        "check_errors": True,
        "static_shapes": False,
        "vector_cutoff": False,
        "aggr": "add",
        "neighbor_embedding": True,
        "attn_activation": "silu",
        "num_heads": 8,
        "distance_influence": "both",
    }

    # PhysicsInformedLNNP includes both delta label handling and physics losses.
    model = PhysicsInformedLNNP(model_args)
    model.trajectory_dataset = full_dataset
    model.trajectory_start_indices = _contiguous_train_starts(train_data, traj_length)
    model.validation_dataset = full_dataset
    model.validation_eval_indices = _evenly_sample_indices(sorted(int(idx) for idx in val_data.indices), val_static_eval_count)
    model.validation_rollout_start_indices = _evenly_sample_indices(_contiguous_subset_starts(val_data, required_span=2), val_rollout_count)
    model.rollout_probe_dataset = full_dataset
    model.train_rollout_probe_start_indices = _evenly_sample_indices(_contiguous_subset_starts(train_data, required_span=2), train_rollout_probe_count)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_rollout_score",
        dirpath=save_dir,
        filename=checkpoint_name,
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_rollout_score", patience=30, mode="min")
    history_callback = MetricHistoryCallback(save_dir, checkpoint_name)
    logger = TensorBoardLogger(save_dir=log_dir, name="physics_informed")
    trainer_callbacks = list(trainer_callbacks or [])
    trainer_kwargs = dict(trainer_kwargs or {})

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop, history_callback, *trainer_callbacks],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1000.0,
        inference_mode=False,
        **trainer_kwargs,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    fit_metrics = {key: float(value.item()) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")}

    print("Testing best checkpoint...")
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    test_callback_metrics = {
        key: float(value.item()) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")
    }
    best_model_score = checkpoint_callback.best_model_score
    best_model_score = float(best_model_score.item()) if best_model_score is not None else None
    best_model_path = checkpoint_callback.best_model_path or str(Path(save_dir) / f"{checkpoint_name}.ckpt")
    val_metrics = dict(fit_metrics)
    val_metrics.update({f"post_test.{key}": value for key, value in test_callback_metrics.items()})
    if best_model_score is not None:
        val_metrics.setdefault("val_rollout_score", best_model_score)

    config = {
        "model_args": model_args,
        "training": {
            "dataset": dataset,
            "molecule": molecule,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "seed": seed,
            "weight_decay": weight_decay,
            "train_loss": train_loss,
            "train_loss_arg": train_loss_arg,
            "delta_learning": bool(delta_learning),
        },
        "validation_metrics": val_metrics,
        "history_paths": {
            "json": str(Path(save_dir) / f"{checkpoint_name}_history.json"),
            "csv": str(Path(save_dir) / f"{checkpoint_name}_history.csv"),
            "plot": str(Path(save_dir) / f"{checkpoint_name}_history.png"),
        },
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "test_results": test_results[0] if test_results else None,
        "physics_weights": {
            "momentum": momentum_weight,
            "nve": nve_weight,
            "pbc": pbc_weight,
        },
        "physics_schedule": {
            "nve_freq": nve_freq,
            "nve_warmup_epochs": nve_warmup_epochs,
            "nve_ramp_epochs": nve_ramp_epochs,
            "nve_relative": bool(nve_relative),
            "nve_relative_eps": float(nve_relative_eps),
            "nve_drift_scale_eV": float(nve_drift_scale_eV),
            "nve_per_atom": bool(nve_per_atom),
            "nve_loss_mode": str(nve_loss_mode),
            "nve_dt_fs": float(nve_dt_fs),
        },
        "rollout_validation": {
            "monitor_metric": "val_rollout_score",
            "val_rollout_steps": int(val_rollout_steps),
            "val_rollout_count": int(val_rollout_count),
            "val_rollout_energy_log_stride": int(val_rollout_energy_log_stride),
            "val_rollout_failure_penalty": float(val_rollout_failure_penalty),
            "val_rollout_max_score": float(val_rollout_max_score),
            "val_static_eval_count": int(val_static_eval_count),
            "train_rollout_probe_steps": int(train_rollout_probe_steps),
            "train_rollout_probe_count": int(train_rollout_probe_count),
            "train_rollout_probe_energy_log_stride": int(train_rollout_probe_energy_log_stride),
        },
    }

    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Model: {save_dir}/{checkpoint_name}.ckpt")
    return {
        "trainer": trainer,
        "model": model,
        "test_results": test_results,
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "validation_metrics": val_metrics,
        "config": config,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum-weight", type=float, default=0.01)
    parser.add_argument("--nve-weight", type=float, default=0.01)
    parser.add_argument("--traj-length", type=int, default=100)
    parser.add_argument("--nve-freq", type=int, default=50)
    parser.add_argument("--nve-warmup-epochs", type=int, default=5)
    parser.add_argument("--nve-ramp-epochs", type=int, default=20)
    parser.add_argument("--nve-relative", dest="nve_relative", action="store_true")
    parser.add_argument("--nve-absolute", dest="nve_relative", action="store_false")
    parser.set_defaults(nve_relative=False)
    parser.add_argument("--nve-relative-eps", type=float, default=1e-6)
    parser.add_argument("--nve-drift-scale-ev", type=float, default=0.1)
    parser.add_argument("--nve-per-atom", dest="nve_per_atom", action="store_true")
    parser.add_argument("--nve-total-drift", dest="nve_per_atom", action="store_false")
    parser.set_defaults(nve_per_atom=True)
    parser.add_argument("--nve-loss-mode", type=str, default="total_energy", choices=["total_energy", "potential_only"])
    parser.add_argument("--nve-dt-fs", type=float, default=0.5)
    parser.add_argument("--val-rollout-steps", type=int, default=250)
    parser.add_argument("--val-rollout-count", type=int, default=6)
    parser.add_argument("--val-rollout-energy-log-stride", type=int, default=10)
    parser.add_argument("--val-rollout-failure-penalty", type=float, default=50.0)
    parser.add_argument("--val-rollout-max-score", type=float, default=1e6)
    parser.add_argument("--val-static-eval-count", type=int, default=128)
    parser.add_argument("--train-rollout-probe-steps", type=int, default=100)
    parser.add_argument("--train-rollout-probe-count", type=int, default=3)
    parser.add_argument("--train-rollout-probe-energy-log-stride", type=int, default=10)
    parser.add_argument("--delta-learning", action="store_true")
    parser.add_argument("--baseline-eps", type=float, default=0.01)
    parser.add_argument("--baseline-sigma", type=float, default=1.0)
    parser.add_argument("--baseline-cutoff", type=float, default=5.0)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--checkpoint-name", type=str, default="best_model")
    args = parser.parse_args()

    train_physics_informed_model(
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum_weight=args.momentum_weight,
        nve_weight=args.nve_weight,
        traj_length=args.traj_length,
        nve_freq=args.nve_freq,
        nve_warmup_epochs=args.nve_warmup_epochs,
        nve_ramp_epochs=args.nve_ramp_epochs,
        nve_relative=args.nve_relative,
        nve_relative_eps=args.nve_relative_eps,
        nve_drift_scale_eV=args.nve_drift_scale_ev,
        nve_per_atom=args.nve_per_atom,
        nve_loss_mode=args.nve_loss_mode,
        nve_dt_fs=args.nve_dt_fs,
        val_rollout_steps=args.val_rollout_steps,
        val_rollout_count=args.val_rollout_count,
        val_rollout_energy_log_stride=args.val_rollout_energy_log_stride,
        val_rollout_failure_penalty=args.val_rollout_failure_penalty,
        val_rollout_max_score=args.val_rollout_max_score,
        val_static_eval_count=args.val_static_eval_count,
        train_rollout_probe_steps=args.train_rollout_probe_steps,
        train_rollout_probe_count=args.train_rollout_probe_count,
        train_rollout_probe_energy_log_stride=args.train_rollout_probe_energy_log_stride,
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        checkpoint_name=args.checkpoint_name,
    )
