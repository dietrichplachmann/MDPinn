#!/usr/bin/env python
"""Bonded-exclusion variant of torchmdnet's ZBL prior, for the water-box
study (paper/main.tex sec:q4-negative-result).

Why this exists: the stock ZBL prior (torchmdnet.priors.zbl.ZBL) applies to
EVERY atom pair within its cutoff, with no concept of molecular topology -
including each water molecule's own covalent O-H bonds (~0.98 A). Because
the O-H bond length sits almost exactly in the same distance range as the
anomalous non-bonded floor this project needed ZBL to reach (~0.8-1.8 A,
diagnose_short_range_collapse.py), enabling stock ZBL injected a large
(~2.6 eV, verify_zbl_units.py), continuous, hard-to-compensate-for
correction onto every O-H bond in every molecule, all the time - and made
NVE rollout stability substantially WORSE, replicated across two training
seeds (paper/main.tex sec:q4-negative-result), not better.

Literature review (2026-08-18, see paper/literature_review_candidates.md
section 0) found no MLIP framework - checked NequIP's and MACE's actual ZBL
source directly - implements a bonded-pair exclusion for this. NequIP/MACE's
own "cutoff normalized by sum of covalent radii" envelope is a per-element-
pair ADAPTIVE CUTOFF, not a topological exclusion - it applies identically
whether a pair is bonded or not, so it would not have resolved this
project's overlap problem either. MACE-OFF (MACE's own flagship for
covalent organic chemistry) simply does not enable ZBL at all. The only
real precedent is classical MD: LAMMPS/CHARMM/AMBER all zero out nonbonded
interactions for 1-2 (bonded) and 1-3 (angle) pairs via a topology-derived
exclusion list, specifically to avoid double-counting energy already
described by bonded/angle terms - exactly the mechanism reused here.

Why "same molecule" (not "same bond") is the right exclusion for THIS
system specifically: a water molecule has exactly 3 atoms (O, H, H), so
every possible atom pair within one molecule is either a direct O-H bond
(1-2) or the H...H angle pair (1-3) - there is no distinction to make.
Excluding by molecule membership (reusing structural_metrics.
infer_molecule_groups via the SAME local_molecule_ids tensor
train_waterbox.py already builds for the momentum loss - not a second,
separately-trusted grouping) is therefore exactly equivalent to the
CHARMM/AMBER 1-2+1-3 convention here, without needing generic bond-graph
distance-2 traversal.

Registration: torchmdnet's create_prior_models looks up prior_model by name
via getattr(priors, name) - it has no way to construct an arbitrary Python
class passed in directly when reloading a saved checkpoint (LNNP(hparams)
with no explicit prior_model= argument, exactly what evaluate_waterbox.py's
load_waterbox_checkpoint and waterbox_ase.py's TensorNetCalculator both do).
register_molecular_zbl_prior() below monkey-patches this class into
torchmdnet.priors's own namespace (same monkey-patching pattern this project
already uses for WaterBox.url in waterbox_data.py and torch.load's
weights_only default in train_waterbox.py/evaluate_waterbox.py) so both
training-time construction AND every later checkpoint-reload path resolve
"MolecularZBL" identically. Call it before constructing ANY model that
might use this prior - idempotent, safe to call repeatedly.

IMPORTANT - written without torchmdnet installed locally (same caveat as
every other water-box script in this project - see CLAUDE.md). Subclassing
torchmdnet.priors.zbl.ZBL means this module cannot even be imported on this
Windows checkout; only python -m py_compile has verified it here. Run the
--smoke-test path of train_waterbox.py (see its --use-zbl-prior
--zbl-bonded-exclusion flags) on the training box before trusting anything
else below.
"""

from __future__ import annotations

import torch
from torchmdnet.models.utils import scatter
from torchmdnet.priors.zbl import ZBL


class MolecularZBL(ZBL):
    """ZBL with same-molecule pairs excluded from the repulsive correction.

    Same constructor arguments as torchmdnet.priors.zbl.ZBL, plus:

    local_molecule_ids: a length-`atoms_per_system` list of per-atom
    molecule ids for ONE system (e.g. train_waterbox.py's
    local_molecule_ids.tolist(), the exact tensor already used for the
    per-fragment momentum loss - not a second, separately-computed
    grouping). Every sample in this dataset has the same fixed atom count
    and ordering (one system, different geometries), so an edge's LOCAL
    atom index within its own system is just (global index) %
    atoms_per_system - the identical assumption/trick
    train_waterbox.py's WaterLNNP._global_molecule_ids_for_batch already
    relies on for the momentum loss; would need revisiting for a dataset
    with a variable number of atoms per sample.
    """

    def __init__(self, cutoff_distance, max_num_neighbors, local_molecule_ids,
                 atomic_number=None, distance_scale=None, energy_scale=None, dataset=None):
        super().__init__(
            cutoff_distance=cutoff_distance,
            max_num_neighbors=max_num_neighbors,
            atomic_number=atomic_number,
            distance_scale=distance_scale,
            energy_scale=energy_scale,
            dataset=dataset,
        )
        local_molecule_ids = torch.as_tensor(local_molecule_ids, dtype=torch.long)
        self.register_buffer("local_molecule_ids", local_molecule_ids)
        self.atoms_per_system = int(local_molecule_ids.shape[0])

    def get_init_args(self):
        return {
            **super().get_init_args(),
            "local_molecule_ids": self.local_molecule_ids.tolist(),
        }

    def post_reduce(self, y, z, pos, batch, box=None, extra_args=None):
        """Identical to ZBL.post_reduce (torchmdnet/priors/zbl.py), except
        every edge between two atoms in the SAME molecule is masked out of
        the energy sum before scattering - see module docstring for why
        same-molecule is the right (and, for this 3-atom-per-molecule
        system, exactly correct) exclusion criterion."""
        edge_index, distance, _ = self.distance(pos, batch, box)
        if edge_index.shape[1] == 0:
            return y

        # Same molecule-membership check the per-fragment momentum loss
        # already relies on (WaterLNNP._global_molecule_ids_for_batch) -
        # local atom index = global index modulo one system's atom count,
        # since every sample here has the same fixed atom count/ordering.
        local_i = edge_index[0] % self.atoms_per_system
        local_j = edge_index[1] % self.atoms_per_system
        same_molecule = self.local_molecule_ids[local_i] == self.local_molecule_ids[local_j]
        keep = ~same_molecule
        if not bool(keep.any()):
            return y
        edge_index = edge_index[:, keep]
        distance = distance[keep]

        atomic_number = self.atomic_number[z[edge_index]]
        # 5.29e-11 is the Bohr radius in meters. All other numbers are magic constants from the ZBL potential.
        a = (
            0.8854
            * 5.29177210903e-11
            / (atomic_number[0] ** 0.23 + atomic_number[1] ** 0.23)
        )
        d = distance * self.distance_scale / a
        f = (
            0.1818 * torch.exp(-3.2 * d)
            + 0.5099 * torch.exp(-0.9423 * d)
            + 0.2802 * torch.exp(-0.4029 * d)
            + 0.02817 * torch.exp(-0.2016 * d)
        )
        f *= self.cutoff(distance)
        # Compute the energy, converting to the dataset's units. Multiply by 0.5 because every atom pair
        # appears twice.
        energy = f * atomic_number[0] * atomic_number[1] / distance
        energy = (
            0.5
            * (2.30707755e-28 / self.energy_scale / self.distance_scale)
            * scatter(energy, batch[edge_index[0]], dim=0, reduce="sum")
        )
        if energy.shape[0] < y.shape[0]:
            energy = torch.nn.functional.pad(energy, (0, y.shape[0] - energy.shape[0]))
        energy = energy.reshape(y.shape)
        return y + energy


def register_molecular_zbl_prior():
    """Monkey-patch MolecularZBL into torchmdnet.priors's own namespace so
    create_prior_models's name-based lookup (getattr(priors, name)) resolves
    it identically whether called at training-time construction
    (train_waterbox.py) or at checkpoint-reload time
    (evaluate_waterbox.py's load_waterbox_checkpoint, used by every rollout
    step via waterbox_ase.py's TensorNetCalculator). Idempotent - safe to
    call on every import/reload path; only registers once."""
    import torchmdnet.priors as priors

    if not hasattr(priors, "MolecularZBL"):
        priors.MolecularZBL = MolecularZBL
        if "MolecularZBL" not in priors.__all__:
            priors.__all__.append("MolecularZBL")
