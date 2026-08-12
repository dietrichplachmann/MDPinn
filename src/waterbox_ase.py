#!/usr/bin/env python
"""ASE Calculator wrapper for a trained water-box TensorNet checkpoint - lets
real MD rollouts (energy conservation, RDF) reuse ASE's own periodic
dynamics/analysis machinery instead of hand-rolling velocity-Verlet with
minimum-image PBC in rollout_nve.py (which has no periodic-boundary handling
at all, and was explicitly scoped out of the water-box study for exactly that
reason - see paper/main.tex Section 5.2 (label sec:q2)).

Why ASE rather than extending rollout_nve.py: this project has already gotten
periodic-boundary handling subtly wrong on a first attempt more than once
(non-periodic bond inference, fixed early in the water-box study; the "shift
everything and wrap" false start when first designing verify_periodicity.py).
ASE's NeighborList/dynamics/RDF code already handles minimum-image convention
and is widely used and tested, so this turns "add PBC support to a
hand-rolled integrator" into "write a ~50-line Calculator adapter" - a much
smaller, lower-risk surface.

Units: ASE's internal convention (Angstrom, eV) matches this project's own
(waterbox_data.py's Bohr/Hartree -> Angstrom/eV conversion) - no additional
unit conversion needed here.

IMPORTANT - written without ase or torchmdnet installed locally (same
caveat as every other water-box script in this repo - see CLAUDE.md).
`pip install ase` on the training box before running anything that imports
this; nothing here has been executed yet.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from evaluate_waterbox import load_waterbox_checkpoint


def tensors_from_atoms(atoms, device):
    """Extract (z, pos, box) tensors from an ASE Atoms object, matching
    WaterBox's own per-sample convention exactly: box shape (1,3,3), row
    i = lattice vector i (confirmed empirically in verify_periodicity.py's
    diagonal/off-diagonal print, not assumed). Shared by TensorNetCalculator
    (live rollout stepping) and analyze_force_decomposition.py (reading back
    already-saved trajectory frames) so both go through the identical
    tensor-construction logic rather than two separately-trusted copies of
    it. pos is NOT given requires_grad here - callers that need forces
    (anything computing energy/forces) must do that themselves, since a
    read-only inspection use (e.g. just checking positions) shouldn't pay
    for/create a grad-tracked tensor it never uses.
    """
    z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    box = None
    if atoms.pbc.any():
        # A rollout genuinely carries atoms outside [0, L) as they diffuse
        # across the periodic boundary over time, unlike the single-shot
        # shift verify_periodicity.py tested - exactly the case minimum-image
        # distance handling exists for, and exactly what was verified there.
        cell = np.array(atoms.get_cell())
        box = torch.as_tensor(cell, dtype=torch.float32, device=device).unsqueeze(0)
    return z, pos, box


class TensorNetCalculator(Calculator):
    """Wraps a trained WaterLNNP/LNNP checkpoint as an ASE Calculator.

    Mirrors evaluate_waterbox.py's _predict_forces exactly (same
    requires_grad-on-pos + enable_grad pattern for the derivative=True force
    head), so every ASE-driven forward pass uses the identical code path
    every other water-box evaluation script already relies on - not a new,
    separately-trusted one.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, checkpoint_path, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_waterbox_checkpoint(checkpoint_path, device=self.device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        z, pos, box = tensors_from_atoms(atoms, self.device)
        pos = pos.clone().detach().requires_grad_(True)
        batch = torch.zeros(len(z), dtype=torch.long, device=self.device)

        with torch.enable_grad():
            energy, forces = self.model(z, pos, batch=batch, box=box)

        self.results["energy"] = float(energy.detach().squeeze().item())
        self.results["forces"] = forces.detach().cpu().numpy()


def atoms_from_waterbox_sample(sample):
    """Build an ASE Atoms object from a single WaterBox Data sample (as
    returned by waterbox_data.load_waterbox_dataset) - positions/box already
    in Angstrom. pbc=True on all three axes: this dataset's box is confirmed
    orthorhombic and periodic in every direction (verify_periodicity.py)."""
    cell = np.array(sample.box).reshape(3, 3)
    return Atoms(
        numbers=sample.z.detach().cpu().numpy(),
        positions=sample.pos.detach().cpu().numpy(),
        cell=cell,
        pbc=True,
    )
