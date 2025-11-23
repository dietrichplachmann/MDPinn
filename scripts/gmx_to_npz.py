import numpy as np
import MDAnalysis as mda
import os

def read_xvg_potential(filename):
    """Reads a GROMACS .xvg energy file (Potential vs Time)."""
    times = []
    energies = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                energies.append(float(parts[1]))
    return np.array(times), np.array(energies)

def build_type_array(universe):
    """Creates atomic number array Z[N] based on element names."""
    Z = []
    for atom in universe.atoms:
        elem = atom.element
        if elem == 'O':
            Z.append(8)
        elif elem == 'H':
            Z.append(1)
        else:
            Z.append(0)
    return np.array(Z, dtype=np.int32)

def main():
    # Paths relative to your project root
    tpr_path = "gromacs/prod.tpr"
    xtc_path = "gromacs/prod.xtc"
    trr_path = "gromacs/prod.trr"
    xvg_path = "gromacs/potential.xvg"

    for path in [tpr_path, xtc_path, trr_path, xvg_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    print("Loading trajectories with MDAnalysis...")
    # Universe for positions (xtc)
    u_pos = mda.Universe(tpr_path, xtc_path)
    # Universe for forces (trr)
    u_for = mda.Universe(tpr_path, trr_path)

    # Build Z from the topology
    Z = build_type_array(u_pos)        # (N,)
    N = len(Z)

    positions = []
    forces = []
    box_lengths = []

    print("Reading frames from XTC (positions) and TRR (forces)...")
    # zip over both trajectories in lockstep
    for ts_pos, ts_for in zip(u_pos.trajectory, u_for.trajectory):
        # positions & box from xtc
        positions.append(ts_pos.positions.copy())           # (N, 3)
        box_lengths.append(ts_pos.dimensions[:3].copy())    # (3,)

        # forces from trr
        if ts_for.has_forces:
            forces.append(ts_for.forces.copy())             # (N, 3)
        else:
            raise RuntimeError("TRR frame has no forces; check nstfout in prod.mdp")

    positions = np.array(positions)      # (T, N, 3)
    forces = np.array(forces)            # (T, N, 3)
    box_lengths = np.array(box_lengths)  # (T, 3)

    print(f"Total synchronized frames read: {positions.shape[0]}")

    print("Reading potential energy from XVG...")
    times, pot = read_xvg_potential(xvg_path)

    T_traj = positions.shape[0]
    T_energy = pot.shape[0]
    T = min(T_traj, T_energy)
    print(f"Synchronizing with energy: using {T} matched frames")

    R = positions[:T]
    F_ref = forces[:T]
    E_ref = pot[:T]
    box_L = box_lengths[:T]
    Z_all = np.tile(Z[None, :], (T, 1))  # (T, N)

    # Shuffle and split
    idx = np.arange(T)
    np.random.shuffle(idx)

    train_frac = 0.8
    valid_frac = 0.1
    n_train = int(train_frac * T)
    n_valid = int(valid_frac * T)

    idx_train = idx[:n_train]
    idx_valid = idx[n_train:n_train + n_valid]
    idx_test = idx[n_train + n_valid:]

    os.makedirs("data", exist_ok=True)

    def save_split(name, indices):
        out_path = f"data/configs_{name}.npz"
        np.savez_compressed(
            out_path,
            R=R[indices],
            Z=Z_all[indices],
            F_ref=F_ref[indices],
            E_ref=E_ref[indices],
            box_L=box_L[indices],
        )
        print(f"Saved: {out_path}")

    save_split("train", idx_train)
    save_split("valid", idx_valid)
    save_split("test", idx_test)

    print("Done.")

if __name__ == "__main__":
    main()
