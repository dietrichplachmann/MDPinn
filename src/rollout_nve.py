#!/usr/bin/env python
"""
NVE rollout evaluation for TorchMD-Net models (MD17)

What this does:
- Loads a trained TorchMD-Net Lightning checkpoint (.ckpt)
- Builds an initial state (x0, v0) from consecutive MD17 frames
- Runs an NVE rollout using velocity Verlet with model-predicted forces
- Logs total energy drift and basic failure checks

Example:
  python rollout_nve.py --ckpt checkpoints/standard/best_model.ckpt --molecule aspirin --steps 20000 --dt 0.5 --n-rollouts 20
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

# PyTorch 2.7 torch.load compatibility (your repo uses this pattern)
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


# --- Unit conversions (MD17 conventions) ---
# Forces from TorchMD-Net on MD17 are typically in eV/Å.
# Masses in amu.
# Need acceleration in Å/fs^2:
#   a(Å/fs^2) = F(eV/Å) * FORCE_TO_ACCEL / m(amu)
FORCE_TO_ACCEL = 0.009648533  # (eV/Å)/amu -> Å/fs^2

# Kinetic energy conversion:
#   1 amu * (Å/fs)^2 = 0.01036427 eV  (same constant used in your physics_losses.py)
AMU_A2_FS2_TO_EV = 0.01036427

# Boltzmann constant in eV/K
K_BOLTZMANN_EV_PER_K = 8.617333262e-5


ATOMIC_MASSES = {
    1: 1.00784,   # H
    6: 12.0107,   # C
    7: 14.0067,   # N
    8: 15.999,    # O
    9: 18.998,    # F
    15: 30.973762,  # P
    16: 32.065,   # S
    17: 35.453,   # Cl
}


def get_atomic_masses(z: torch.Tensor) -> torch.Tensor:
    """z: (N,) atomic numbers -> masses (N,) in amu"""
    masses = torch.empty_like(z, dtype=torch.float32)
    for i, zi in enumerate(z.tolist()):
        if zi not in ATOMIC_MASSES:
            raise KeyError(f"Missing atomic mass for Z={zi}. Add it to ATOMIC_MASSES.")
        masses[i] = float(ATOMIC_MASSES[zi])
    return masses


def load_lnnp_from_ckpt(ckpt_path: str, device: str) -> LNNP:
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

    model.eval().to(device)
    return model


def model_energy_forces(model: LNNP, z: torch.Tensor, pos: torch.Tensor, device: str):
    """
    Returns:
      U: scalar tensor (eV) detached
      F: (N,3) tensor (eV/Å) detached
    """
    n = z.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)

    # Need grads ON for forces (dE/dx), but we detach outputs so we don't keep graphs.
    with torch.enable_grad():
        pos_req = pos.detach().clone().requires_grad_(True)
        out = model(z, pos_req, batch=batch)

        if isinstance(out, tuple) and len(out) >= 2:
            y, neg_dy = out[0], out[1]
        else:
            y, neg_dy = out, None

        if neg_dy is None:
            raise RuntimeError("Model did not return forces (neg_dy). Ensure derivative=True in the checkpoint.")

        U = y.squeeze().detach()
        F = neg_dy.detach()

    return U, F


def kinetic_energy(masses_amu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    masses_amu: (N,)
    v: (N,3) in Å/fs
    returns scalar K in eV
    """
    v2 = (v * v).sum(dim=-1)  # (N,)
    return 0.5 * AMU_A2_FS2_TO_EV * (masses_amu * v2).sum()



def remove_com_velocity(v: torch.Tensor, masses_amu: torch.Tensor) -> torch.Tensor:
    """Remove center-of-mass velocity so total linear momentum is ~0."""
    m = masses_amu.view(-1, 1).to(v.device)
    v_cm = (m * v).sum(dim=0, keepdim=True) / m.sum()
    return v - v_cm


def sample_maxwell_boltzmann_velocities(masses_amu: torch.Tensor, temperature_k: float, device: str) -> torch.Tensor:
    """
    Sample per-atom velocities from Maxwell–Boltzmann distribution.
    Returns v in Å/fs.

    For each Cartesian component:
      0.5 * m * v^2 * (AMU_A2_FS2_TO_EV) = 0.5 * kB * T
      => var(v) = kB*T / (m*AMU_A2_FS2_TO_EV)
    """
    m = masses_amu.to(device).float().view(-1, 1)  # (N,1)
    var = (K_BOLTZMANN_EV_PER_K * float(temperature_k)) / (m * AMU_A2_FS2_TO_EV)  # (N,1) (Å/fs)^2
    std = torch.sqrt(var).expand(-1, 3)  # (N,3)
    v = torch.randn((m.shape[0], 3), device=device) * std
    return remove_com_velocity(v, masses_amu.to(device))


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
):
    """
    Returns a dict with time series of energies + drift, and whether it failed.
    """
    # state
    x = x0.clone()
    v = v0.clone()

    # initial forces
    U0, F0 = model_energy_forces(model, z, x, device=device)
    a0 = (F0 * FORCE_TO_ACCEL) / masses_amu.view(-1, 1)  # (N,3) Å/fs^2

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

    # integration
    a = a0
    for step in range(1, steps + 1):
        # v(t+1/2)
        v_half = v + 0.5 * dt_fs * a
        # x(t+1)
        x = x + dt_fs * v_half

        # basic blow-up check (cheap)
        if torch.isnan(x).any() or torch.isinf(x).any() or x.abs().max().item() > 1e4:
            failed = True
            fail_reason = "position_numerical_blowup"
            break

        # forces at new x
        U, F = model_energy_forces(model, z, x, device=device)
        a_new = (F * FORCE_TO_ACCEL) / masses_amu.view(-1, 1)

        # v(t+1)
        v = v_half + 0.5 * dt_fs * a_new
        a = a_new

        if step % energy_log_stride == 0 or step == steps:
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

    return {
        "failed": failed,
        "fail_reason": fail_reason,
        "series": series,
        # summarize
        "final_step": int(series["step"][-1]) if series["step"] else 0,
        "final_drift": float(series["drift"][-1]) if series["drift"] else None,
        "max_abs_drift": float(max(abs(d) for d in series["drift"])) if series["drift"] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")
    ap.add_argument("--molecule", type=str, default="aspirin")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=0.5, help="fs")
    ap.add_argument("--n-rollouts", type=int, default=20)
    ap.add_argument("--energy-log-stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    print(f"Device: {device}")
    print(f"Loading model: {args.ckpt}")
    model = load_lnnp_from_ckpt(args.ckpt, device=device)

    # dataset split consistent with your training scripts
    full = MD17(root=args.data_root, molecules=args.molecule)

    train_size = int(0.8 * len(full))
    val_size = int(0.1 * len(full))
    test_size = len(full) - train_size - val_size
    train_data, val_data, test_data = random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # For rollouts we want consecutive frames, so we sample indices from *full dataset*,
    # but we restrict starting points to the test index set.
    # random_split gives Subset with indices attribute.
    test_indices = list(test_data.indices)
    test_indices = sorted(test_indices)

    # need t and t+1 (for v0), so filter out last index
    test_start_indices = [i for i in test_indices if (i + 1) < len(full)]
    if len(test_start_indices) == 0:
        raise RuntimeError("No valid start indices found in test set.")

    # sample starts evenly to reduce overlap
    stride = max(1, len(test_start_indices) // args.n_rollouts)
    starts = test_start_indices[::stride][: args.n_rollouts]
    if len(starts) < args.n_rollouts:
        # pad with random choices
        extra = torch.randint(0, len(test_start_indices), (args.n_rollouts - len(starts),)).tolist()
        starts += [test_start_indices[i] for i in extra]

    results = {
        "ckpt": args.ckpt,
        "molecule": args.molecule,
        "steps": args.steps,
        "dt_fs": args.dt,
        "frame_dt_fs": args.frame_dt,
        "init_vel": args.init_vel,
        "temp_K": args.temp,
        "vel_scale": args.vel_scale,
        "n_rollouts": args.n_rollouts,
        "energy_log_stride": args.energy_log_stride,
        "seed": args.seed,
        "rollouts": [],
        "summary": {},
    }

    print(f"Running {args.n_rollouts} rollouts (NVE, velocity Verlet) ...")

    n_failed = 0
    max_abs_drifts = []
    final_drifts = []

    for r, start_idx in enumerate(starts):
        s0 = full[start_idx]
        s1 = full[start_idx + 1]

        z = s0.z.to(device)
        x0 = s0.pos.to(device).float()
        x1 = s1.pos.to(device).float()

        masses = get_atomic_masses(z).to(device)

        # initial velocity from finite diff
        v0 = (x1 - x0) / args.dt  # Å/fs

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
        )

        results["rollouts"].append({
            "rollout_id": r,
            "start_idx": int(start_idx),
            **out
        })

        if out["failed"]:
            n_failed += 1
            print(f"[{r+1:02d}/{args.n_rollouts}] start={start_idx} FAILED ({out['fail_reason']}) at step {out['final_step']}")
        else:
            max_abs_drifts.append(out["max_abs_drift"])
            final_drifts.append(out["final_drift"])
            print(f"[{r+1:02d}/{args.n_rollouts}] start={start_idx} ok | max|drift|={out['max_abs_drift']:.6g} eV | final drift={out['final_drift']:.6g} eV")

    results["summary"] = {
        "failed": int(n_failed),
        "success": int(args.n_rollouts - n_failed),
        "failure_rate": float(n_failed / args.n_rollouts),
        "mean_max_abs_drift_eV": float(sum(max_abs_drifts) / len(max_abs_drifts)) if max_abs_drifts else None,
        "mean_final_drift_eV": float(sum(final_drifts) / len(final_drifts)) if final_drifts else None,
    }

    print("Summary:", json.dumps(results["summary"], indent=2))

    if args.out:
        out_path = Path(args.out)
    else:
        ck = Path(args.ckpt)
        out_path = ck.with_suffix("").with_name(ck.stem + f"_rollout_{args.steps}steps.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
