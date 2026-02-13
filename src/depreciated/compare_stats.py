import numpy as np

# 1) training energies used in loss L_E
train = np.load("data/configs_train.npz")
E_ref = train["E_ref"]        # (B,)
print("Train E_ref: mean =", E_ref.mean(), "std =", E_ref.std())

# 2) GROMACS potential from potential.xvg
E_gro = []
with open("gromacs/potential.xvg") as f:
    for line in f:
        if line.startswith(("#", "@")):
            continue
        t, e = line.split()[:2]
        E_gro.append(float(e))
E_gro = np.array(E_gro)
print("GROMACS Pot: mean =", E_gro.mean(), "std =", E_gro.std())