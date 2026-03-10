#!/usr/bin/env python
"""
NVE rollout evaluation for trained checkpoints.

Delta-learning consistency rule implemented here:
- If checkpoint stores delta metadata, dynamics use hybrid quantities
  U_hyb = U_ref + DeltaU and F_hyb = F_ref + DeltaF.

Chemist view:
- This file tests whether learned forces produce stable trajectories when
  integrated forward in time.
- If energy drift explodes quickly, the learned potential is not yet robust for MD.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import random_split

from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

from baseline_potential import lj_energy_forces_batched


# PyTorch 2.7 checkpoint compatibility.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


FORCE_TO_ACCEL = 0.009648533
AMU_A2_FS2_TO_EV = 0.01036427
# FORCE_TO_ACCEL converts force units (eV/A)/amu to acceleration A/fs^2.
# AMU_A2_FS2_TO_EV converts kinetic term m*v^2 to eV.

ATOMIC_MASSES = {
    1: 1.00784,
    6: 12.0107,
    7: 14.0067,
    8: 15.999,
    9: 18.998,
    15: 30.973762,
    16: 32.065,
    17: 35.453,
}


def get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    """Map atomic numbers to masses in amu."""
    masses = torch.empty_like(z, dtype=torch.float32)
    for i, zi in enumerate(z.tolist()):
        if zi not in ATOMIC_MASSES:
            raise KeyError(f"Missing atomic mass for Z={zi}. Add it to ATOMIC_MASSES.")
        masses[i] = float(ATOMIC_MASSES[zi])
    return masses


def load_lnnp_from_ckpt(ckpt_path: str, device: str) -> LNNP:
    """Load checkpoint and attach delta metadata fields used during rollouts."""
    ckpt = torch.load(ckpt_path, map_location=device)

    if "hyper_parameters" in ckpt:
        hparams = ckpt["hyper_parameters"]
    elif "hparams" in ckpt:
        hparams = ckpt["hparams"]
    else:
        raise ValueError("Checkpoint has no hyper_parameters/hparams.")

    model = LNNP(hparams)
    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint has no state_dict.")
    model.load_state_dict(ckpt["state_dict"], strict=True)

    model._delta_learning = bool(hparams.get("delta_learning", False))
    model._baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
    model._baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
    model._baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))

    return model.eval().to(device)


def model_energy_forces(model: LNNP, z: torch.Tensor, pos: torch.Tensor, device: str):
    """Predict absolute potential energy and forces for current coordinates.

    Returned outputs are always intended for physical integration:
    - U_abs(R): scalar potential energy used for reporting drift
    - F_abs(R): force vectors used in velocity Verlet updates
    """
    n = z.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)

    # TorchMD-Net force prediction uses autograd internally, so positions must
    # require grad even during rollout inference.
    pos_req = pos.detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(z, pos_req, batch=batch)

    if isinstance(out, tuple) and len(out) >= 2:
        y, neg_dy = out[0], out[1]
    else:
        y, neg_dy = out, None

    if neg_dy is None:
        raise RuntimeError("Model did not return forces (neg_dy). Ensure derivative=True in checkpoint.")

    # Convert from residual predictions to absolute if checkpoint is delta-trained.
    # This enforces physically consistent hybrid dynamics in rollout:
    #   F_hyb = F_ref + DeltaF_NN
    if getattr(model, "_delta_learning", False):
        u_ref, f_ref = lj_energy_forces_batched(
            z=z,
            pos=pos_req,
            batch=batch,
            epsilon_eV=getattr(model, "_baseline_eps", 0.01),
            sigma_A=getattr(model, "_baseline_sigma", 1.0),
            r_cut_A=getattr(model, "_baseline_cutoff", 5.0),
        )
        y = y.squeeze(-1) + u_ref
        neg_dy = neg_dy + f_ref

    return y.squeeze().detach(), neg_dy.detach()


def kinetic_energy(masses_amu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute kinetic energy in eV from masses (amu) and velocities (A/fs)."""
    v2 = (v * v).sum(dim=-1)
    return 0.5 * AMU_A2_FS2_TO_EV * (masses_amu * v2).sum()


def velocity_verlet_rollout(
    model: LNNP,
    z: torch.Tensor,
    masses_amu: torch.Tensor,
    x0: torch.Tensor,
    v0: torch.Tensor,
    steps: int,
    dt_fs: float,
    device: str,
    energy_log_stride: int = 10,
    progress_stride: int = 0,
    rollout_id: int = None,
):
    """Run one NVE rollout and return logged energies/drift statistics.

    Integrator reminder:
    - half-step velocity update
    - full-step position update
    - force recompute
    - second half-step velocity update
    """
    x = x0.clone()
    v = v0.clone()

    U0, F0 = model_energy_forces(model, z, x, device=device)
    a = (F0 * FORCE_TO_ACCEL) / masses_amu.view(-1, 1)

    E0 = kinetic_energy(masses_amu, v) + U0

    series = {
        "E0": float(E0.item()),
        "U0": float(U0.item()),
        "K0": float((E0 - U0).item()),
        "dt_fs": dt_fs,
        "energy_log_stride": energy_log_stride,
        "steps": steps,
        "E": [],
        "U": [],
        "K": [],
        "drift": [],
        "step": [],
    }

    failed = False
    fail_reason = None

    for step in range(1, steps + 1):
        v_half = v + 0.5 * dt_fs * a
        x = x + dt_fs * v_half

        # Quick hard-failure detection to stop obviously unstable rollouts.
        if torch.isnan(x).any() or torch.isinf(x).any() or x.abs().max().item() > 1e4:
            failed = True
            fail_reason = "position_numerical_blowup"
            break

        U, F = model_energy_forces(model, z, x, device=device)
        a_new = (F * FORCE_TO_ACCEL) / masses_amu.view(-1, 1)
        v = v_half + 0.5 * dt_fs * a_new
        a = a_new

        if step % energy_log_stride == 0 or step == steps:
            # Track instantaneous drift relative to step-0 energy.
            # For ideal NVE and perfect numerics this would stay near 0.
            K = kinetic_energy(masses_amu, v)
            E = K + U
            drift = E - E0

            series["step"].append(step)
            series["E"].append(float(E.item()))
            series["U"].append(float(U.item()))
            series["K"].append(float(K.item()))
            series["drift"].append(float(drift.item()))

            if torch.isnan(E) or torch.isinf(E):
                failed = True
                fail_reason = "energy_numerical_blowup"
                break

        if progress_stride > 0 and (step % progress_stride == 0 or step == steps):
            tag = f"[rollout {rollout_id}] " if rollout_id is not None else ""
            print(f"{tag}step {step}/{steps}")

    return {
        "failed": failed,
        "fail_reason": fail_reason,
        "series": series,
        "final_step": int(series["step"][-1]) if series["step"] else 0,
        "final_drift": float(series["drift"][-1]) if series["drift"] else None,
        "max_abs_drift": float(max(abs(d) for d in series["drift"])) if series["drift"] else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="MD17")
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--n-rollouts", type=int, default=20)
    parser.add_argument("--energy-log-stride", type=int, default=10)
    parser.add_argument("--progress-stride", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    model = load_lnnp_from_ckpt(args.ckpt, device=device)

    if args.dataset != "MD17":
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented in rollout_nve.py (supported: MD17).")

    full = MD17(root=args.data_root, molecules=args.molecule)

    train_size = int(0.8 * len(full))
    val_size = int(0.1 * len(full))
    test_size = len(full) - train_size - val_size
    _, _, test_data = random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_indices = sorted(list(test_data.indices))
    test_start_indices = [i for i in test_indices if (i + 1) < len(full)]
    if not test_start_indices:
        raise RuntimeError("No valid start indices found in test set.")

    stride = max(1, len(test_start_indices) // args.n_rollouts)
    starts = test_start_indices[::stride][: args.n_rollouts]
    if len(starts) < args.n_rollouts:
        extra = torch.randint(0, len(test_start_indices), (args.n_rollouts - len(starts),)).tolist()
        starts += [test_start_indices[i] for i in extra]

    results = {
        "ckpt": args.ckpt,
        "molecule": args.molecule,
        "steps": args.steps,
        "dt_fs": args.dt,
        "n_rollouts": args.n_rollouts,
        "energy_log_stride": args.energy_log_stride,
        "seed": args.seed,
        "rollouts": [],
        "summary": {},
    }

    n_failed = 0
    max_abs_drifts = []
    final_drifts = []

    for ridx, start_idx in enumerate(starts):
        print(f"Starting rollout {ridx + 1}/{len(starts)} (start_idx={start_idx})")
        s0 = full[start_idx]
        s1 = full[start_idx + 1]

        z = s0.z.to(device)
        x0 = s0.pos.to(device).float()
        x1 = s1.pos.to(device).float()

        masses = get_atomic_masses(z).to(device)
        v0 = (x1 - x0) / args.dt

        out = velocity_verlet_rollout(
            model=model,
            z=z,
            masses_amu=masses,
            x0=x0,
            v0=v0,
            steps=args.steps,
            dt_fs=args.dt,
            device=device,
            energy_log_stride=args.energy_log_stride,
            progress_stride=args.progress_stride,
            rollout_id=ridx + 1,
        )

        results["rollouts"].append({"rollout_id": ridx, "start_idx": int(start_idx), **out})
        status = "FAILED" if out["failed"] else "OK"
        print(
            f"Finished rollout {ridx + 1}/{len(starts)}: {status}, "
            f"final_step={out['final_step']}, max_abs_drift={out['max_abs_drift']}"
        )

        if out["failed"]:
            n_failed += 1
        else:
            max_abs_drifts.append(out["max_abs_drift"])
            final_drifts.append(out["final_drift"])

    results["summary"] = {
        "failed": int(n_failed),
        "success": int(args.n_rollouts - n_failed),
        "failure_rate": float(n_failed / args.n_rollouts),
        "mean_max_abs_drift_eV": float(sum(max_abs_drifts) / len(max_abs_drifts)) if max_abs_drifts else None,
        "mean_final_drift_eV": float(sum(final_drifts) / len(final_drifts)) if final_drifts else None,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        ck = Path(args.ckpt)
        out_path = ck.with_suffix("").with_name(ck.stem + f"_rollout_{args.steps}steps.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
