#!/usr/bin/env python
"""Torchmdnet-dependent half of the localized per-molecule Boltzmann
estimator (paper/main.tex sec:q4-stable-local-estimator-scope) - the fix for
the global-vs-local observable mismatch diagnosed there:
`train_waterbox_stable.py`'s original `_stacked_rdf` observable is a single
whole-box RDF per configuration, averaged over all 64 water molecules - but
Raja et al. 2025 (StABlE, arXiv:2402.13984) explicitly warn that "global
observables g(Gamma) may be insensitive to" localized failure modes
(Section 3.1, Definition 2), and use a LOCAL per-molecule estimator for
water specifically, not the global one used for aspirin/Ac-Ala3-NHMe.

The pure numpy/torch pieces (local_molecule_energies,
per_molecule_mean_oh_bond_length) live in local_molecule_geometry.py instead
of here specifically so they stay exercisable with synthetic arrays on this
Windows checkout - see that module for the per-molecule math itself and
src/verify_local_molecule_geometry.py for its synthetic test. This module
holds only what genuinely needs torchmdnet: capturing the model's own
per-atom energy tensor, and sampling a real reference dataset for the
observable's target value.

boltzmann_estimator.py's actual gradient-estimator math needs NO changes -
Equation 5 (local) is identical in form to Equation 4 (global, already
implemented and verified there); it doesn't care whether a "sample" is a
global configuration or a local per-molecule neighborhood, only the layer
that constructs g and U changes.

Capturing the per-atom energy tensor: verified directly against torchmd-net's
actual source (torchmdnet/models/model.py's TorchMD_Net.forward), not
assumed. The forward pass computes per-atom energies via
`x = self.output_model.pre_reduce(x, v, z, pos, batch, box=box)`, gives any
active prior model's OWN `pre_reduce` a chance to modify this SAME per-atom
tensor, then finally reduces to one scalar per configuration via
`self.output_model.reduce(x, batch, num_systems=...)`. torchmdnet/priors/
base.py's BasePrior.pre_reduce default is a pure identity (`return x`), so a
side-effect-only prior appended to the model's `prior_model` list (a mutable
torch.nn.ModuleList - confirmed from torchmdnet/models/model.py's
TorchMD_Net.__init__: `self.prior_model = None if prior_model is None else
torch.nn.ModuleList(prior_model)`) can capture a reference to that tensor and
return it completely unchanged - no monkey-patching of forward() itself, and
zero change to the model's real energy/force output for every other purpose.

IMPORTANT, checked directly against torchmdnet/priors/zbl.py's actual source,
not assumed: ZBL (and molecular_zbl.py's MolecularZBL) overrides only
post_reduce, not pre_reduce - its repulsive correction is computed from its
own independent edge/distance calculation and added directly to the
already-reduced scalar y, never touching the per-atom tensor x at all. This
means AtomicEnergyCapture below always sees the BARE GNN atomic energy
decomposition, with zero ZBL contribution, regardless of where in
prior_model's list it sits. Treated as correct, not a gap to work around:
Raja et al.'s local-energy formula assumes a clean per-atom decomposition,
which the bare GNN output is and ZBL's pairwise correction structurally is
not; and MolecularZBL already excludes same-molecule pairs from its
correction by design (paper/main.tex sec:q4-bonded-exclusion), so it
contributes exactly zero to any single-molecule neighborhood's energy
anyway. L_QM is completely unaffected by any of this and continues to
regularize against the model's FULL forward pass (GNN + ZBL together) - only
this local-observable capture is GNN-only, a deliberate, documented
simplification, not an oversight.

Attached transiently, in-memory, only for the duration of one StABlE
fine-tuning run (attach_atomic_energy_capture, called once right after
loading a checkpoint in train_waterbox_stable.py) - NOT part of any saved
checkpoint's hyperparameters, unlike MolecularZBL: this prior has no
parameters/buffers and changes no model output, so it needs no
re-registration at every other reload site (evaluate_waterbox.py,
waterbox_ase.py) the way MolecularZBL does.

IMPORTANT - written without torchmdnet installed locally (same caveat as
every other water-box script in this project - see CLAUDE.md). Importing
torchmdnet.priors.base means this module cannot even be imported on this
Windows checkout; only python -m py_compile has verified it here. What
remains unverified until run on the training box (see paper/main.tex
sec:q4-stable-local-estimator-scope for the full list): the exact attribute
path from a loaded LNNP wrapper to the real TorchMD_Net instance's
`.prior_model` (verified from torchmdnet/module.py's LNNP.__init__:
`self.model = create_model(...)`, so `lnnp_instance.model.prior_model` is
the path used below - not yet exercised against a running model); whether
captured per-atom energies' atom-ordering aligns with z/pos/molecule-group
ordering (expected, since both derive from the same forward call, but
unconfirmed).
"""

from __future__ import annotations

import numpy as np
import torch
from torchmdnet.priors.base import BasePrior

from local_molecule_geometry import per_molecule_mean_oh_bond_length


class AtomicEnergyCapture(BasePrior):
    """Side-effect-only prior: stashes a reference to the model's own
    per-atom energy tensor (x, immediately after output_model.pre_reduce,
    before any prior's pre_reduce or the final whole-system reduction) and
    returns it completely unchanged - a pure identity, matching BasePrior's
    own default pre_reduce exactly, so attaching this changes NOTHING about
    the model's real energy/force output for any other caller. See this
    module's docstring for why the captured tensor is deliberately
    GNN-only, with zero contribution from MolecularZBL.

    `captured` is overwritten (not accumulated) on every forward call - read
    it immediately after the model call it corresponds to, before calling
    the model again for a different configuration.
    """

    def __init__(self, dataset=None):
        super().__init__(dataset=dataset)
        self.captured = None

    def pre_reduce(self, x, z, pos, batch, extra_args):
        self.captured = x
        return x


def attach_atomic_energy_capture(lnnp_model):
    """Appends a fresh AtomicEnergyCapture to lnnp_model.model.prior_model
    and returns the capture instance so the caller can read `.captured`
    after every forward call through lnnp_model. lnnp_model is whatever
    evaluate_waterbox.load_waterbox_checkpoint returns (an LNNP instance);
    `.model` is confirmed directly from torchmdnet/module.py's
    `LNNP.__init__` (`self.model = create_model(...)`) to be the real
    TorchMD_Net instance, and `.prior_model` is confirmed from
    torchmdnet/models/model.py's `TorchMD_Net.__init__` to be either None or
    a torch.nn.ModuleList - both cases handled here. List POSITION does not
    matter for what gets captured (this module's docstring - nothing else in
    this project's prior_model touches x at the pre_reduce stage), so a
    plain append is the simplest correct choice.

    NOT saved into any checkpoint - call this fresh every time a model is
    loaded for StABlE fine-tuning specifically; do not call this for
    evaluate_waterbox.py/waterbox_ase.py's ordinary checkpoint reloads,
    which have no use for it.
    """
    capture = AtomicEnergyCapture()
    torch_model = lnnp_model.model
    if torch_model.prior_model is None:
        torch_model.prior_model = torch.nn.ModuleList([capture])
    else:
        torch_model.prior_model.append(capture)
    return capture


def sample_reference_mean_oh_bond_length(full_dataset, n_samples=200, seed=42):
    """Scalar g_target: the true (real DFT) average mean-O-H-bond-length
    across every molecule in a random sample of real reference
    configurations - the single number Raja et al.'s water-specific local
    Boltzmann estimator trains against (Section 4.4), replacing
    _sample_reference_frames + the whole-box reference RDF this observable
    supersedes. Needs torchmdnet (via full_dataset's own Data samples) -
    training-box only, same caveat as every other function here.
    """
    from diagnose_short_range_collapse import molecule_group_ids

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(full_dataset))
    indices = rng.choice(len(full_dataset), size=n, replace=False)

    all_values = []
    for idx in indices:
        sample = full_dataset[int(idx)]
        z = sample.z.detach().cpu().numpy()
        pos = sample.pos.detach().cpu().numpy()
        box_lengths = np.asarray(sample.box).reshape(3, 3).diagonal()
        group_ids = molecule_group_ids(z, pos, box_lengths)
        all_values.extend(per_molecule_mean_oh_bond_length(z, pos, box_lengths, group_ids).tolist())
    return float(np.mean(all_values))
