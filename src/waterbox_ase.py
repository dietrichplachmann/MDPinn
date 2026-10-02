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


def _strain_matrix_from_voigt(strain_voigt, *, dtype, device):
    """Return a symmetric 3x3 strain tensor from ASE Voigt ordering.

    ``strain_voigt`` is ordered xx, yy, zz, yz, xz, xy.  The shear entries
    are engineering strains, so each symmetric off-diagonal tensor entry is
    half the corresponding Voigt value.  With this convention, differentiating
    energy with respect to ``strain_voigt`` returns ASE's stress convention in
    the same ordering.
    """
    strain = torch.zeros((3, 3), dtype=dtype, device=device)
    strain[0, 0] = strain_voigt[0]
    strain[1, 1] = strain_voigt[1]
    strain[2, 2] = strain_voigt[2]
    strain[1, 2] = strain[2, 1] = 0.5 * strain_voigt[3]
    strain[0, 2] = strain[2, 0] = 0.5 * strain_voigt[4]
    strain[0, 1] = strain[1, 0] = 0.5 * strain_voigt[5]
    return strain


class TensorNetCalculator(Calculator):
    """Wraps a trained WaterLNNP/LNNP checkpoint as an ASE Calculator.

    Mirrors evaluate_waterbox.py's _predict_forces exactly (same
    requires_grad-on-pos + enable_grad pattern for the derivative=True force
    head), so every ASE-driven forward pass uses the identical code path
    every other water-box evaluation script already relies on - not a new,
    separately-trusted one.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self, checkpoint_path=None, model=None, device=None,
        stress_mode="analytic", stress_fd_epsilon=0.003, **kwargs,
    ):
        """Pass exactly one of checkpoint_path (loads a fresh model from
        disk, the original/only behavior before this parameter was added)
        or model (uses an already-loaded model object directly - added for
        train_waterbox_stable.py's StABlE fine-tuning loop, sec:q4-stable-plan,
        where every replica's Calculator must share the SAME live model
        object the optimizer is updating, not its own independently-loaded
        copy - PyTorch's optimizer.step() mutates parameters in place, so a
        shared reference automatically reflects every later update with no
        extra plumbing, but only if it really is the same object, not a
        separately-loaded twin)."""
        super().__init__(**kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if stress_mode not in {"analytic", "hydrostatic_fd"}:
            raise ValueError(
                "stress_mode must be 'analytic' or 'hydrostatic_fd', "
                f"got {stress_mode!r}."
            )
        if not 0.0 < stress_fd_epsilon < 0.1:
            raise ValueError(
                f"stress_fd_epsilon must be between 0 and 0.1, got {stress_fd_epsilon}."
            )
        self.stress_mode = stress_mode
        self.stress_fd_epsilon = float(stress_fd_epsilon)
        if model is not None:
            if checkpoint_path is not None:
                raise ValueError("Pass exactly one of checkpoint_path or model, not both.")
            self.model = model
        else:
            if checkpoint_path is None:
                raise ValueError("Must pass checkpoint_path or model.")
            self.model = load_waterbox_checkpoint(checkpoint_path, device=self.device)

    def _energy_only(self, z, pos, batch, box):
        """Evaluate energy without TorchMD-Net's force derivative.

        In eval mode the force-producing forward frees the energy graph.  A
        separate energy-only pass is therefore required both for analytic
        strain differentiation and for finite-difference hydrostatic stress.
        """
        derivative_model = getattr(self.model, "model", None)
        if derivative_model is None or not hasattr(derivative_model, "derivative"):
            raise RuntimeError(
                "Cannot locate TorchMD-Net derivative flag for stress evaluation."
            )
        original_derivative = derivative_model.derivative
        try:
            derivative_model.derivative = False
            energy, _ = self.model(z, pos, batch=batch, box=box)
        finally:
            derivative_model.derivative = original_derivative
        return energy

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        z, pos, box = tensors_from_atoms(atoms, self.device)
        pos = pos.clone().detach().requires_grad_(True)
        batch = torch.zeros(len(z), dtype=torch.long, device=self.device)

        need_stress = properties is not None and "stress" in properties
        strain_voigt = None
        if need_stress and self.stress_mode == "analytic":
            if box is None:
                raise ValueError("Stress requires a periodic simulation cell.")
            strain_voigt = torch.zeros(6, dtype=pos.dtype, device=self.device, requires_grad=True)
            strain = _strain_matrix_from_voigt(strain_voigt, dtype=pos.dtype, device=self.device)
            deformation_t = torch.eye(3, dtype=pos.dtype, device=self.device) + strain.T
            # Positions are row vectors and box rows are lattice vectors.  A
            # homogeneous real-space deformation therefore right-multiplies
            # both by F^T, keeping fractional coordinates fixed.
            model_pos = pos @ deformation_t
            model_box = box @ deformation_t
        else:
            model_pos = pos
            model_box = box

        with torch.enable_grad():
            # Preserve the established force-producing inference path.  In
            # eval mode TorchMD-Net's internal autograd.grad frees the energy
            # graph after forming forces, so stress cannot safely be taken
            # from that same forward pass.
            energy, forces = self.model(z, pos, batch=batch, box=box)

            if need_stress and self.stress_mode == "analytic":
                # Perform a second, energy-only pass for the strain derivative.
                # LNNP owns the actual TorchMD_Net as `.model`; toggling only
                # its derivative flag avoids train-mode behavior while keeping
                # the graph required for dE/d(strain).  NPT therefore costs one
                # additional energy forward per step.
                stress_energy = self._energy_only(
                    z, model_pos, batch=batch, box=model_box
                )
                strain_gradient = torch.autograd.grad(
                    stress_energy.sum(), strain_voigt, retain_graph=False, create_graph=False,
                    allow_unused=False,
                )[0]
                volume = float(atoms.get_volume())
                if not np.isfinite(volume) or volume <= 0:
                    raise ValueError(f"Stress requires a finite positive cell volume, got {volume}.")
                self.results["stress"] = (strain_gradient / volume).detach().cpu().numpy()
            elif need_stress:
                if box is None:
                    raise ValueError("Stress requires a periodic simulation cell.")
                volume = float(atoms.get_volume())
                if not np.isfinite(volume) or volume <= 0:
                    raise ValueError(f"Stress requires a finite positive cell volume, got {volume}.")
                epsilon = self.stress_fd_epsilon
                # Use a wrapped coordinate copy to keep the numerical strain
                # independent of which periodic images happen to be stored as
                # atoms diffuse.  The live ASE positions are not modified.
                wrapped_pos = torch.as_tensor(
                    atoms.get_positions(wrap=True), dtype=pos.dtype, device=self.device
                )
                energy_plus = self._energy_only(
                    z, wrapped_pos * (1.0 + epsilon), batch,
                    box * (1.0 + epsilon),
                )
                energy_minus = self._energy_only(
                    z, wrapped_pos * (1.0 - epsilon), batch,
                    box * (1.0 - epsilon),
                )
                stress_trace = (energy_plus - energy_minus) / (2.0 * epsilon * volume)
                isotropic_component = float(stress_trace.detach().squeeze().item()) / 3.0
                # IsotropicMTKNPT uses only -trace(stress)/3. Returning the
                # isotropic projection is deliberate; this mode must not be
                # used with anisotropic/full-cell barostats.
                self.results["stress"] = np.asarray(
                    [isotropic_component, isotropic_component, isotropic_component, 0.0, 0.0, 0.0]
                )

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
