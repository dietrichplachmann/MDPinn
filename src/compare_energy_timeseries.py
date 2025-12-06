import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import MDAnalysis as mda

from model import CorrectionPotential
from checkpointing import load_checkpoint


def read_xvg_potential(xvg_path):
    """
    Read GROMACS potential.xvg file.
    Returns:
        times (np.ndarray), energies (np.ndarray)
    """
    times = []
    energies = []
    with open(xvg_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                energies.append(float(parts[1]))
    return np.array(times), np.array(energies)


def build_Z_from_universe(u):
    """
    Build atomic number array Z[N] from MDAnalysis universe.
    Assumes a water box with O and H atoms.
    """
    Z = []
    for atom in u.atoms:
        elem = atom.element
        if elem == "O":
            Z.append(8)
        elif elem == "H":
            Z.append(1)
        else:
            # fallback if MDAnalysis element is missing/unknown
            Z.append(0)
    return torch.tensor(Z, dtype=torch.long)


def load_model_from_checkpoint(ckpt_path, device):
    """
    Load model + config from a checkpoint file created by train.py
    using save_checkpoint(...).
    """
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    config = ckpt["config"]
    mp = config["model_params"]

    model = CorrectionPotential(
        emb_dim=mp["emb_dim"],
        hidden_dim=mp["hidden_dim"],
        n_rbf=mp["n_rbf"],
        cutoff=mp["cutoff"],
        max_Z=mp["max_Z"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Compare GROMACS potential vs PINN model energy over time."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint file (run_YYYYMMDD_HHMMSS.pt). "
             "If not provided, uses 'trained_model.pth' with default model params.",
    )
    parser.add_argument(
        "--tpr",
        type=str,
        default=os.path.join("gromacs", "prod.tpr"),
        help="Path to GROMACS .tpr file.",
    )
    parser.add_argument(
        "--xtc",
        type=str,
        default=os.path.join("gromacs", "prod.xtc"),
        help="Path to GROMACS .xtc file.",
    )
    parser.add_argument(
        "--xvg",
        type=str,
        default=os.path.join("gromacs", "potential.xvg"),
        help="Path to GROMACS potential.xvg file.",
    )
    parser.add_argument(
        "--savefig",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g. 'energy_compare.png').",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Check required files ----
    for p in [args.tpr, args.xtc, args.xvg]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # ---- Load GROMACS potential ----
    t_gmx, E_gmx = read_xvg_potential(args.xvg)
    print(f"Read {len(t_gmx)} energy points from {args.xvg}")

    # ---- Load trajectory ----
    print(f"Loading trajectory from {args.tpr} and {args.xtc} ...")
    u = mda.Universe(args.tpr, args.xtc)

    # Atomic numbers (assumed constant over time)
    Z = build_Z_from_universe(u).to(device)

    # ---- Load model ----
    if args.ckpt is not None:
        print(f"Loading model from checkpoint: {args.ckpt}")
        model, config = load_model_from_checkpoint(args.ckpt, device)
    else:
        # Fallback: use trained_model.pth with hard-coded model params
        state_path = "trained_model.pth"
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                "No checkpoint provided and 'trained_model.pth' not found."
            )
        print(f"Loading model from {state_path}")
        # Use the same defaults as in train.py
        model = CorrectionPotential(
            emb_dim=16,
            hidden_dim=64,
            n_rbf=32,
            cutoff=6.0,
            max_Z=100,
        ).to(device)
        model.load_state_dict(torch.load(state_path, map_location=device))
        model.eval()
        config = None

    # ---- Evaluate model energy over trajectory ----
    E_model_list = []
    t_traj = []

    print("Evaluating model on trajectory frames...")
    with torch.no_grad():
        for ts in u.trajectory:
            # positions in nm
            R = torch.tensor(ts.positions, dtype=torch.float32, device=device)
            E_pred = model(R, Z)  # scalar
            E_model_list.append(E_pred.item())
            t_traj.append(ts.time)  # ps (if default GROMACS)

    E_model = np.array(E_model_list)
    t_traj = np.array(t_traj)

    # some debug stuff
    print("First 10 model energies (raw):", E_model[:10])
    print("Min/Max model energy:", E_model.min(), E_model.max())

    print(f"Trajectory frames: {len(t_traj)}")

    # ---- Align time series lengths ----
    # simplest option: truncate all to the minimum length
    T = min(len(t_gmx), len(t_traj), len(E_model))
    t_common = t_gmx[:T]           # assuming energies and trajectory are aligned
    E_gmx_common = E_gmx[:T]
    E_model_common = E_model[:T]

    # ---- Optionally, shift energies to compare fluctuations ----
    E_gmx_shift = E_gmx_common - E_gmx_common.mean()
    E_model_shift = E_model_common - E_model_common.mean()

    # ---- Plot Overlay----
    plt.figure(figsize=(10, 6))
    plt.plot(t_common, E_gmx_shift, label="GROMACS Potential (shifted)")
    plt.plot(t_common, E_model_shift, label="Model Potential (shifted)", alpha=0.7)
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (kJ/mol, mean-shifted)")
    plt.title("GROMACS vs PINN Model Potential Energy Over Time")
    plt.legend()
    plt.tight_layout()

    # ---- Plot GROMACS ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_common, E_gmx_shift)
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (kJ/mol, shifted)")
    plt.title("GROMACS Potential Energy Over Time")

    # ---- Plot Model ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_common, E_model_shift)
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (kJ/mol, shifted)")
    plt.title("PINN Model Potential Energy Over Time")

    plt.show()

    if args.savefig is not None:
        plt.savefig(args.savefig, dpi=200)
        print(f"Saved figure to {args.savefig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
