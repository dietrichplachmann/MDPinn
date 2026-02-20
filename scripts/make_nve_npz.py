#!/usr/bin/env python

import argparse
import numpy as np
import MDAnalysis as mda


def load_traj_positions(tpr, xtc):
    """
    Load positions, box, and masses using MDAnalysis.

    tpr : GROMACS .tpr file (e.g. prod.tpr)
    xtc : GROMACS .xtc file (positions, e.g. prod.xtc)
    """
    print(f"[INFO] Loading topology+positions: {tpr}, {xtc}")
    u = mda.Universe(tpr, xtc)

    T = len(u.trajectory)
    N = u.atoms.n_atoms
    print(f"[INFO] Trajectory: {T} frames, {N} atoms")

    R = np.zeros((T, N, 3), dtype=np.float32)
    box = np.zeros((T, 3), dtype=np.float32)

    for i, ts in enumerate(u.trajectory):
        R[i] = ts.positions
        box[i] = ts.dimensions[:3]

    masses = u.atoms.masses.astype(np.float32)  # not actually needed for potential-only NVE

    return R, box, masses


def load_energy_xvg(xvg_path):
    """Load a single energy column from an .xvg file (time, energy)."""
    print(f"[INFO] Loading energy from {xvg_path}")
    energies = []
    with open(xvg_path) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            _, e = parts[:2]
            energies.append(float(e))
    U = np.array(energies, dtype=np.float32)
    print(f"[INFO] Loaded {U.shape[0]} energy samples from xvg")
    return U


def load_Z_from_train_npz(train_npz_path):
    """Load atomic numbers Z from your existing configs_train.npz."""
    print(f"[INFO] Loading Z from train npz: {train_npz_path}")
    data = np.load(train_npz_path)
    Z_all = data["Z"]  # (B_train, N)
    Z = Z_all[0].astype(np.int64)  # (N,)
    print(f"[INFO] Z shape from train npz: {Z_all.shape} -> using {Z.shape}")
    return Z


def segment_trajectory(R, U_ref, box, Z, segment_length, stride):
    """
    Slice full trajectory into segments.

    R:      (T, N, 3)
    U_ref:  (T,)
    box:    (T, 3)
    Z:      (N,)
    """
    T, N, _ = R.shape
    print(f"[INFO] Segmenting trajectory: T={T}, segment_length={segment_length}, stride={stride}")

    segments = []
    start = 0
    while start + segment_length <= T:
        end = start + segment_length
        segments.append((start, end))
        start += stride

    if not segments:
        raise RuntimeError(
            f"No segments generated. Check that segment_length <= T (T={T})."
        )

    B_seg = len(segments)
    print(f"[INFO] Generated {B_seg} segments")

    R_out = np.zeros((B_seg, segment_length, N, 3), dtype=np.float32)
    U_out = np.zeros((B_seg, segment_length), dtype=np.float32)
    Z_out = np.zeros((B_seg, N), dtype=np.int64)
    box_out = np.zeros((B_seg, 3), dtype=np.float32)

    for i, (s, e) in enumerate(segments):
        R_out[i] = R[s:e]
        U_out[i] = U_ref[s:e]
        box_out[i] = box[s:e].mean(axis=0)
        Z_out[i] = Z

    return R_out, U_out, Z_out, box_out


def main():
    parser = argparse.ArgumentParser(
        description="Generate NVE npz (potential-only) for PINN/NNP training."
    )
    parser.add_argument("--tpr", required=True, help="GROMACS .tpr file (e.g. prod.tpr)")
    parser.add_argument("--xtc", required=True, help="GROMACS .xtc file (positions, e.g. prod.xtc)")
    parser.add_argument("--xvg", required=True, help="Energy xvg from `gmx energy` (time vs potential)")
    parser.add_argument("--train-npz", required=True,
                        help="Path to configs_train.npz to reuse Z (atomic numbers)")
    parser.add_argument("--segment-length", type=int, default=40,
                        help="Number of frames per NVE segment (T)")
    parser.add_argument("--stride", type=int, default=40,
                        help="Stride between segment starts (segment_length for non-overlap)")
    parser.add_argument("--out", default="data/nve_trajs.npz",
                        help="Output npz path")

    args = parser.parse_args()

    # Load positions, box
    R, box, _ = load_traj_positions(args.tpr, args.xtc)

    # Load Z from your existing training data
    Z = load_Z_from_train_npz(args.train_npz)

    # Sanity check: same N
    T, N, _ = R.shape
    if Z.shape[0] != N:
        raise RuntimeError(
            f"Mismatch between atoms in traj (N={N}) and Z from train npz (N={Z.shape[0]})."
        )

    # Load reference potential energies along the trajectory
    U_ref = load_energy_xvg(args.xvg)
    if U_ref.shape[0] != T:
        raise RuntimeError(
            f"Mismatch between frames in traj (T={T}) and energy samples (T_energy={U_ref.shape[0]}). "
            "Ensure nstenergy matches nstxout in MD."
        )

    # Segment into smaller windows
    R_out, U_out, Z_out, box_out = segment_trajectory(
        R, U_ref, box, Z,
        segment_length=args.segment_length,
        stride=args.stride
    )

    # Save npz
    print(f"[INFO] Saving NVE dataset to {args.out}")
    np.savez(
        args.out,
        R_traj=R_out,   # (B_traj, T, N, 3)
        Z=Z_out,        # (B_traj, N)
        box_L=box_out,  # (B_traj, 3)
        U_ref=U_out,    # (B_traj, T)
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
