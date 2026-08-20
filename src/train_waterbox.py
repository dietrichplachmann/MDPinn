#!/usr/bin/env python
"""Physics-informed training on a periodic liquid-water box.

Deliberately narrower in scope than train_physics.py - see the project plan
("MDPinn: Periodic water-box study") for the full rationale:
- No delta-learning: no water analytic baseline is built (baseline_potential.py
  is aspirin-specific and its unvectorized O(N^2) nonbonded loop is exactly the
  kind of thing that made delta-learning intractable on aspirin - not worth
  reproducing at 192 atoms instead of 21).
- No NVE/rollout diagnostics: rollout_nve.py has no periodic-boundary handling
  today, so no rollout probes run during training here. The actual metric this
  study cares about - per-molecule momentum violation - is measured on static
  held-out configurations by run_waterbox_study.py's evaluation step instead.

Two conditions:
- water_absolute: momentum_weight=0 - plain supervised energy/force loss, same
  0.05/0.95 energy/force weighting convention as everywhere else in this repo.
- water_absolute+momentum: momentum_weight>0 - adds a per-INDIVIDUAL-MOLECULE
  momentum-conservation penalty (physics_losses.per_fragment_momentum_loss),
  not a per-whole-box penalty. Checking momentum conservation per molecule
  rather than per whole system is the entire point of this study: a whole
  periodic box's net force is already guaranteed ~0 by the same equivariance
  argument that made this loss redundant on a single isolated aspirin molecule,
  but an individual water molecule being pushed by its neighbors is not
  guaranteed to have zero net force/torque on it, so this is a genuinely
  different (non-redundant) test.

IMPORTANT - written without torchmdnet installed locally (see waterbox_data.py's
docstring): the WaterBox dataset's constructor/attribute names are taken from
torchmd-net's public source, not verified against an installed copy. Run the
--smoke-test path first and watch for errors before trusting anything else here.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.module import LNNP

from molecular_zbl import register_molecular_zbl_prior
from physics_losses import build_global_molecule_ids, per_fragment_momentum_loss
from structural_metrics import infer_molecule_groups, summarize_molecule_groups
from training_history import MetricHistoryCallback
from waterbox_data import load_waterbox_dataset, random_split


# PyTorch 2.7 checkpoint compatibility (matches train_physics.py/train_standard.py).
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


class WaterLNNP(LNNP):
    """Plain-supervised LNNP plus an optional per-molecule momentum penalty.

    Overrides on_validation_epoch_end (in addition to on_train_epoch_start/end)
    to log val_checkpoint_score - see that method's docstring for why
    val_total_mse_loss alone isn't a safe thing to monitor for checkpoint
    selection when weight-annealing is active. Every override calls super()
    first, so the base LNNP's own logging (val_total_mse_loss and friends)
    still works exactly as it does in the plain train_standard.py path - no
    risk of the missing-super()-call bug that broke val_total_mse_loss there
    earlier in this project.
    """

    def __init__(
        self,
        hparams,
        local_molecule_ids,
        num_molecules_per_system,
        anneal_epoch=None,
        post_anneal_energy_weight=None,
        post_anneal_force_weight=None,
        **kwargs,
    ):
        super().__init__(hparams, **kwargs)
        self.momentum_weight = float(hparams.get("momentum_weight", 0.0))
        self.register_buffer("local_molecule_ids", local_molecule_ids.clone())
        self.num_molecules_per_system = int(num_molecules_per_system)
        self.atoms_per_system = int(local_molecule_ids.shape[0])

        # Energy/force loss-weight annealing (see train_waterbox_model's
        # docstring for citations). anneal_epoch=None (default) preserves the
        # original fixed-weight behavior exactly.
        self.anneal_epoch = anneal_epoch
        self.post_anneal_energy_weight = post_anneal_energy_weight
        self.post_anneal_force_weight = post_anneal_force_weight
        self._pre_anneal_y_weight = float(self.hparams.y_weight)
        self._pre_anneal_neg_dy_weight = float(self.hparams.neg_dy_weight)
        self._annealed = False

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.anneal_epoch is None or self._annealed:
            return
        if self.current_epoch < self.anneal_epoch:
            return
        # LNNP.step() reads self.hparams.y_weight/neg_dy_weight fresh on every
        # call (not cached at construction), so mutating them here takes
        # effect starting with this epoch's very first training batch.
        self.hparams.y_weight = self.post_anneal_energy_weight
        self.hparams.neg_dy_weight = self.post_anneal_force_weight
        self._annealed = True
        print(
            f"[weight anneal] epoch {self.current_epoch}: switching "
            f"y_weight {self._pre_anneal_y_weight} -> {self.post_anneal_energy_weight}, "
            f"neg_dy_weight {self._pre_anneal_neg_dy_weight} -> {self.post_anneal_force_weight}"
        )

    def on_train_epoch_end(self):
        # MUST call super() first - LNNP's own on_train_epoch_end logs base
        # training metrics; skipping this previously broke val_total_mse_loss
        # logging elsewhere in this project (see CLAUDE.md's lessons-learned).
        super().on_train_epoch_end()
        # Logged (not just printed) so it lands in best_model_history.csv via
        # MetricHistoryCallback, making the schedule directly visible/
        # verifiable in the same place as every other training curve.
        self.log("train_y_weight", float(self.hparams.y_weight), on_step=False, on_epoch=True)
        self.log("train_neg_dy_weight", float(self.hparams.neg_dy_weight), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # MUST call super() first - see on_train_epoch_end's comment above.
        # This also means val_y_mse_loss/val_neg_dy_mse_loss are already
        # populated in trainer.callback_metrics by the time this line
        # finishes, since LNNP's own on_validation_epoch_end is what logs
        # them.
        super().on_validation_epoch_end()

        # val_total_mse_loss (what LNNP logs) is w_E*val_y_mse + w_F*val_neg_dy_mse
        # using WHATEVER (w_E, w_F) are active this epoch - fine as a training
        # diagnostic, but not safe for ModelCheckpoint/EarlyStopping to
        # monitor when annealing is active: its definition changes at
        # anneal_epoch (force-dominant 0.05/0.95 before, energy-dominant
        # 0.75/0.25 after), so a post-anneal epoch is judged by a
        # structurally different, harsher yardstick than a pre-anneal one,
        # regardless of whether the model actually improved - confirmed
        # empirically, every water-box cell's "best" epoch landed pre-anneal.
        # val_checkpoint_score fixes this by always using the run's original
        # (pre-anneal) weights, so "best" means the same thing on every
        # epoch of the run, annealed or not. For a non-annealed run
        # (anneal_epoch=None), this is identical to val_total_mse_loss, since
        # _pre_anneal_y_weight/neg_dy_weight are just the run's one constant
        # weight pair - so this is a no-op change for existing behavior.
        y_mse = self.trainer.callback_metrics.get("val_y_mse_loss")
        neg_dy_mse = self.trainer.callback_metrics.get("val_neg_dy_mse_loss")
        if y_mse is not None and neg_dy_mse is not None:
            checkpoint_score = self._pre_anneal_y_weight * y_mse + self._pre_anneal_neg_dy_weight * neg_dy_mse
            self.log("val_checkpoint_score", checkpoint_score, on_step=False, on_epoch=True)

    def _global_molecule_ids_for_batch(self, batch):
        n_atoms = batch.z.shape[0]
        # Every WaterBox sample has the same fixed atom count/order (one system,
        # different geometries), and torch_geometric's default collation
        # concatenates same-sized graphs' nodes in order without reordering
        # within a graph - so atom i's position within its own graph is
        # i % atoms_per_system. This would need revisiting for a dataset with a
        # variable number of atoms per sample.
        local_atom_idx = torch.arange(n_atoms, device=batch.z.device) % self.atoms_per_system
        local_ids = self.local_molecule_ids.to(batch.z.device)[local_atom_idx]
        return build_global_molecule_ids(batch.batch, local_ids, self.num_molecules_per_system)

    def step(self, batch, loss_fn_list, stage):
        total_loss = super().step(batch, loss_fn_list, stage)

        # Gated entirely behind momentum_weight > 0, unlike train_physics.py's
        # PhysicsInformedLNNP (which pays for an extra forward pass every batch
        # regardless of momentum_weight) - water_absolute (momentum_weight=0)
        # should cost nothing beyond the base supervised loss.
        if stage != "train" or self.momentum_weight <= 0:
            return total_loss

        try:
            batch.pos = batch.pos.clone().detach().requires_grad_(True)
            _, neg_dy = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
            )
            molecule_ids = self._global_molecule_ids_for_batch(batch)
            num_molecules_in_batch = int(molecule_ids.max().item()) + 1
            loss_momentum = per_fragment_momentum_loss(batch.pos, neg_dy, molecule_ids, num_molecules_in_batch)

            weighted_momentum = self.momentum_weight * loss_momentum
            self.log("train_loss_momentum_per_molecule", loss_momentum, on_step=False, on_epoch=True)
            self.log("train_loss_momentum_per_molecule_weighted", weighted_momentum, on_step=False, on_epoch=True)
            total_loss = total_loss + weighted_momentum
            self.log("train_total_with_momentum", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as exc:
            print(f"Warning: per-molecule momentum loss failed: {exc}")
            traceback.print_exc()

        return total_loss


def _build_local_molecule_ids(full_dataset):
    """Infer per-atom molecule ids once from a representative frame, and sanity
    check the grouping before trusting it anywhere (see the project plan: a
    silent mis-grouping wouldn't crash, it would just make the momentum loss
    meaningless)."""
    sample = full_dataset[0]
    group_ids = infer_molecule_groups(sample.z, sample.pos, box=getattr(sample, "box", None))
    summary = summarize_molecule_groups(sample.z, group_ids)

    bad_groups = [c for c in summary["compositions"] if c != {8: 1, 1: 2}]
    print(f"infer_molecule_groups: {summary['n_groups']} groups from {sample.z.shape[0]} atoms")
    if bad_groups:
        raise ValueError(
            f"Expected every group to be exactly {{O: 1, H: 2}} (atomic numbers 8/1) for a "
            f"water box; got {len(bad_groups)} group(s) that don't match, e.g. {bad_groups[:3]}. "
            "Do not proceed - the momentum loss would be meaningless with a wrong grouping."
        )

    return group_ids, summary["n_groups"]


#  Bohr radius -> meters and eV -> Joules are the two factors ZBL needs to
# bridge its own native-unit formula to this project's Angstrom/eV convention
# (waterbox_data.py's positions/energies). Verified empirically, not just
# taken on faith, via verify_zbl_units.py - this project has been burned
# before by trusting a unit-conversion factor without checking it against
# physical magnitudes first (the Bohr/Hartree mixup in waterbox_data.py; see
# CLAUDE.md's Lessons). ZBL_CUTOFF_DISTANCE=2.0 A was chosen the same way:
# verify_zbl_units.py shows it gives a large repulsive correction right at
# the empirical short-range-collapse floors diagnose_short_range_collapse.py
# found (paper/main.tex sec:q4), while staying near-zero at real equilibrium
# non-bonded separations (O-O/H-H exactly 0, O-H hydrogen-bond distance only
# 0.003 eV) - cutoffs of 2.5 A or more start leaking a non-negligible
# correction into the hydrogen-bond distance itself (up to 0.30 eV at 5.0 A,
# comparable to a real H-bond's own energy scale), which would fight the
# bulk-water physics the network already learned correctly rather than only
# fixing the missing short-range repulsion. A nonzero ZBL contribution AT the
# covalent O-H bond length is a real, blanket pairwise addition with no
# concept of "this pair is a real bond" (same as stock NequIP/MACE usage) -
# and, per paper/main.tex sec:q4-negative-result, this is exactly what made
# enabling stock ZBL substantially WORSEN rollout stability at two
# independent training seeds, not fix it: the network never learned to
# compensate for a ~2.6 eV correction dumped onto every O-H bond,
# continuously, throughout the whole simulation. zbl_bonded_exclusion=True
# (see molecular_zbl.py) fixes this by excluding same-molecule pairs from
# the correction entirely, reusing the CHARMM/AMBER 1-2/1-3 nonbonded-
# exclusion convention (paper/literature_review_candidates.md section 0) -
# no MLIP framework checked (NequIP, MACE, GRACE) implements this; MACE-OFF
# (MACE's own flagship for covalent organic chemistry) simply doesn't
# enable ZBL at all.
_ZBL_ANGSTROM_TO_METER = 1e-10
_ZBL_EV_TO_JOULE = 1.602176634e-19
ZBL_CUTOFF_DISTANCE = 2.0
ZBL_MAX_NUM_NEIGHBORS = 128


def train_waterbox_model(
    data_root="./data",
    batch_size=32,
    num_epochs=20,
    lr=1e-4,
    model_type="tensornet",
    save_dir="checkpoints/waterbox",
    log_dir="logs/waterbox",
    force_weight=0.95,
    energy_weight=0.05,
    momentum_weight=0.0,
    embedding_dimension=256,
    num_layers=6,
    num_rbf=64,
    checkpoint_name="best_model",
    train_loss="mse_loss",
    train_loss_arg=None,
    weight_decay=0.0,
    lr_patience=15,
    lr_min=1e-7,
    lr_factor=0.8,
    early_stop_patience=30,
    num_workers=4,
    seed=42,
    anneal_epoch=None,
    post_anneal_energy_weight=None,
    post_anneal_force_weight=None,
    use_zbl_prior=False,
    zbl_cutoff_distance=ZBL_CUTOFF_DISTANCE,
    zbl_max_num_neighbors=ZBL_MAX_NUM_NEIGHBORS,
    zbl_bonded_exclusion=False,
    trainer_callbacks=None,
    trainer_kwargs=None,
):
    """Train a water-box model (absolute-mode only - no delta-learning here).

    anneal_epoch/post_anneal_energy_weight/post_anneal_force_weight: optional
    energy/force loss-weight schedule. anneal_epoch=None (default) keeps
    energy_weight/force_weight fixed for the whole run, unchanged from before.
    When set, training starts at (energy_weight, force_weight) as usual, then
    switches to (post_anneal_energy_weight, post_anneal_force_weight) at
    anneal_epoch and stays there. This mirrors the force-then-energy annealing
    schedules reported for MACE and NequIP (force-dominant early to fix the
    local force/gradient shape, energy-dominant later to fix up absolute
    energy calibration) - see run_waterbox_study.py for the specific schedule
    and citations used here.

    momentum_weight=0.0 is the "water_absolute" condition; momentum_weight>0 is
    "water_absolute+momentum" - see module docstring for what that term checks.

    use_zbl_prior=False (default) preserves every existing run's behavior
    exactly (prior_model stays None, unchanged). When True, adds torchmdnet's
    ZBL short-range repulsive prior (paper/main.tex sec:q4 - the fix for the
    both-conditions rollout instability, contingent on
    diagnose_short_range_collapse.py's finding that missing short-range
    repulsion is at least a partial cause). zbl_cutoff_distance/
    zbl_max_num_neighbors default to the empirically-checked values in
    verify_zbl_units.py - see that script and the ZBL_CUTOFF_DISTANCE
    constant's comment above for why 2.0 A, not the model's own 5.0 A cutoff.

    Stock ZBL (zbl_bonded_exclusion=False, the default when use_zbl_prior is
    True) is confirmed to substantially WORSEN rollout stability rather than
    fix it (paper/main.tex sec:q4-negative-result) - kept as the default only
    for backward compatibility with that already-completed comparison, not
    because it's recommended. zbl_bonded_exclusion=True switches to
    molecular_zbl.MolecularZBL, which excludes same-molecule atom pairs from
    the ZBL correction (see that module's docstring for why "same molecule"
    is the exactly-correct exclusion criterion for this 3-atom-per-molecule
    system, and paper/literature_review_candidates.md section 0 for why this
    is the standard classical-force-field convention rather than something
    invented for this project). Ignored when use_zbl_prior is False.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Water-Box Training")
    print("=" * 70)
    print(f"model={model_type}, momentum_weight={momentum_weight}")

    pl.seed_everything(seed, workers=True)

    full_dataset = load_waterbox_dataset(data_root=data_root)
    print(f"Loaded WaterBox: {len(full_dataset)} configurations")
    train_data, val_data, test_data = random_split(full_dataset, seed=seed)

    local_molecule_ids, num_molecules_per_system = _build_local_molecule_ids(full_dataset)

    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    prior_args = {
        "cutoff_distance": zbl_cutoff_distance,
        "max_num_neighbors": zbl_max_num_neighbors,
        # WaterBox's z field already stores literal atomic numbers (8/1),
        # not a compacted 0-based type index (confirmed by every other
        # z==8/z==1 check throughout this project - structural_metrics.py,
        # rollout_waterbox_ase.py's ELEMENT_PAIRS) - so ZBL's own
        # model-type-index -> real-atomic-number lookup table is just the
        # identity, sized to match max_z below so every index the model
        # can embed is covered.
        "atomic_number": list(range(100)),
        # Angstrom -> meters / eV -> Joules: this project's own position/
        # energy convention (waterbox_data.py), bridged to ZBL's native
        # SI-unit formula - see the module-level comment above
        # ZBL_CUTOFF_DISTANCE for why these specific values, verified via
        # verify_zbl_units.py rather than taken on faith.
        "distance_scale": _ZBL_ANGSTROM_TO_METER,
        "energy_scale": _ZBL_EV_TO_JOULE,
    } if use_zbl_prior else None

    if use_zbl_prior and zbl_bonded_exclusion:
        # Registers "MolecularZBL" into torchmdnet.priors's own namespace so
        # create_prior_models's name-based lookup resolves it - must happen
        # before WaterLNNP(...) below constructs the model. See
        # molecular_zbl.py's module docstring for why this same registration
        # also has to happen at checkpoint-reload time
        # (evaluate_waterbox.py's load_waterbox_checkpoint), not just here.
        register_molecular_zbl_prior()
        prior_model_name = "MolecularZBL"
        prior_args["local_molecule_ids"] = local_molecule_ids.tolist()
    elif use_zbl_prior:
        prior_model_name = "ZBL"
    else:
        prior_model_name = None

    model_args = {
        "model": model_type,
        "prior_model": prior_model_name,
        "prior_args": prior_args,
        "output_model": "Scalar",
        "load_model": None,
        "remove_ref_energy": False,
        "train_loss": train_loss,
        "train_loss_arg": train_loss_arg,
        "charge": False,
        "spin": False,
        "precision": 32,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "embedding_dimension": int(embedding_dimension),
        "num_layers": int(num_layers),
        "num_rbf": int(num_rbf),
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "max_z": 100,
        "max_num_neighbors": 128,
        "derivative": True,
        "lr": lr,
        "lr_patience": lr_patience,
        "lr_min": lr_min,
        "lr_factor": lr_factor,
        "lr_warmup_steps": 0,
        "weight_decay": weight_decay,
        "y_weight": energy_weight,
        "neg_dy_weight": force_weight,
        "ema_alpha_y": 1.0,
        "ema_alpha_neg_dy": 1.0,
        "momentum_weight": momentum_weight,
        # Left unset (None): WaterBox provides box vectors per sample already
        # (batch.box), which per TorchMD-Net's own docs is an alternative to
        # setting a single fixed box_vecs hparam for the whole dataset - NOT
        # verified locally (no torchmdnet installed here), confirm on the
        # training box that periodicity is actually active (e.g. check that
        # neighbor search respects the box) before trusting results.
        "box_vecs": None,
        "atom_filter": -1,
        "reduce_op": "add",
        "equivariance_invariance_group": "O(3)",
        "check_errors": True,
        "static_shapes": False,
        "vector_cutoff": False,
        "aggr": "add",
        "neighbor_embedding": True,
        "attn_activation": "silu",
        "num_heads": 8,
        "distance_influence": "both",
    }

    model = WaterLNNP(
        model_args,
        local_molecule_ids=local_molecule_ids,
        num_molecules_per_system=num_molecules_per_system,
        anneal_epoch=anneal_epoch,
        post_anneal_energy_weight=post_anneal_energy_weight,
        post_anneal_force_weight=post_anneal_force_weight,
    )

    # Both conditions are absolute-mode only (no delta-learning), so
    # val_checkpoint_score means the same thing in both - no confound between
    # water_absolute and water_absolute+momentum the way there would be if one
    # used a residualized target and the other didn't.
    #
    # Monitors val_checkpoint_score (WaterLNNP.on_validation_epoch_end),
    # NOT val_total_mse_loss - the latter is weighted by whatever (y_weight,
    # neg_dy_weight) are active that epoch, which the anneal schedule changes
    # partway through the run. Monitoring it directly would judge post-anneal
    # epochs by a structurally different, harsher yardstick than pre-anneal
    # ones, biasing "best" toward pre-anneal regardless of actual model
    # quality - confirmed empirically on the first annealed run, where every
    # cell's best epoch landed pre-anneal. val_checkpoint_score always uses
    # the run's original weights, so it means the same thing on every epoch.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_checkpoint_score",
        dirpath=save_dir,
        filename=checkpoint_name,
        save_top_k=1,
        mode="min",
    )
    # Dedicated true-last-epoch checkpoint, separate from the monitored
    # best-tracker above. save_last=True on a monitored ModelCheckpoint only
    # updates last.ckpt "whenever a checkpoint file gets saved" (Lightning's
    # own docs) - i.e. only on improvement, same gating as the best-tracker
    # itself. If the best epoch stops improving early (confirmed happening
    # here - e.g. one water_absolute seed's global-minimum val_total_mse_loss
    # was epoch 3 of 18, never beaten again), last.ckpt silently freezes at
    # that same epoch instead of tracking the actual final epoch, making the
    # two checkpoints indistinguishable. monitor=None + save_top_k=0 removes
    # the improvement gate entirely, so this one unconditionally overwrites
    # last.ckpt every epoch regardless of validation performance.
    last_epoch_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor=None,
        save_top_k=0,
        save_last=True,
    )
    # Same reasoning as checkpoint_callback above - monitor the anneal-
    # invariant score, not the raw weighted total, so "no improvement in N
    # epochs" isn't tripped by the metric's own scale jumping at anneal_epoch.
    early_stop = EarlyStopping(monitor="val_checkpoint_score", patience=early_stop_patience, mode="min", strict=False)
    history_callback = MetricHistoryCallback(save_dir, checkpoint_name)
    logger = TensorBoardLogger(save_dir=log_dir, name="waterbox")
    trainer_callbacks = list(trainer_callbacks or [])
    trainer_kwargs = dict(trainer_kwargs or {})

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, last_epoch_callback, early_stop, history_callback, *trainer_callbacks],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1000.0,
        inference_mode=False,
        **trainer_kwargs,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    fit_metrics = {key: float(value.item()) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")}

    print("Testing best checkpoint...")
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    test_callback_metrics = {
        key: float(value.item()) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")
    }
    best_model_score = checkpoint_callback.best_model_score
    best_model_score = float(best_model_score.item()) if best_model_score is not None else None
    best_model_path = checkpoint_callback.best_model_path or str(Path(save_dir) / f"{checkpoint_name}.ckpt")
    val_metrics = dict(fit_metrics)
    val_metrics.update({f"post_test.{key}": value for key, value in test_callback_metrics.items()})
    if best_model_score is not None:
        # checkpoint_callback now monitors val_checkpoint_score, not
        # val_total_mse_loss (see checkpoint_callback's own comment) - label
        # this with the metric it actually is, not the old one.
        val_metrics.setdefault("val_checkpoint_score", best_model_score)

    config = {
        "model_args": model_args,
        "training": {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "seed": seed,
            "weight_decay": weight_decay,
            "train_loss": train_loss,
            "train_loss_arg": train_loss_arg,
            "momentum_weight": momentum_weight,
            "use_zbl_prior": use_zbl_prior,
            "zbl_cutoff_distance": zbl_cutoff_distance if use_zbl_prior else None,
            "zbl_max_num_neighbors": zbl_max_num_neighbors if use_zbl_prior else None,
            "zbl_bonded_exclusion": zbl_bonded_exclusion if use_zbl_prior else None,
        },
        "num_molecules_per_system": num_molecules_per_system,
        "validation_metrics": val_metrics,
        "history_paths": {
            "json": str(Path(save_dir) / f"{checkpoint_name}_history.json"),
            "csv": str(Path(save_dir) / f"{checkpoint_name}_history.csv"),
            "plot": str(Path(save_dir) / f"{checkpoint_name}_history.png"),
            "plot_no_epoch0": str(Path(save_dir) / f"{checkpoint_name}_history_no_epoch0.png"),
        },
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "test_results": test_results[0] if test_results else None,
    }

    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Model: {save_dir}/{checkpoint_name}.ckpt")
    return {
        "trainer": trainer,
        "model": model,
        "test_results": test_results,
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "validation_metrics": val_metrics,
        "config": config,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum-weight", type=float, default=0.0)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--checkpoint-name", type=str, default="best_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-zbl-prior", action="store_true",
        help="Add torchmdnet's ZBL short-range repulsive prior (paper/main.tex sec:q4). "
        "Default off, so existing runs stay reproducible unless explicitly requested. "
        "Stock ZBL (without --zbl-bonded-exclusion) is confirmed to substantially WORSEN "
        "rollout stability (paper/main.tex sec:q4-negative-result) - pass "
        "--zbl-bonded-exclusion too unless deliberately reproducing that already-completed "
        "negative-result comparison.",
    )
    parser.add_argument("--zbl-cutoff-distance", type=float, default=ZBL_CUTOFF_DISTANCE)
    parser.add_argument("--zbl-max-num-neighbors", type=int, default=ZBL_MAX_NUM_NEIGHBORS)
    parser.add_argument(
        "--zbl-bonded-exclusion", action="store_true",
        help="Use molecular_zbl.MolecularZBL instead of stock ZBL - excludes same-molecule "
        "atom pairs from the repulsive correction (see molecular_zbl.py). Only takes effect "
        "with --use-zbl-prior.",
    )
    args = parser.parse_args()

    train_waterbox_model(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        use_zbl_prior=args.use_zbl_prior,
        zbl_cutoff_distance=args.zbl_cutoff_distance,
        zbl_max_num_neighbors=args.zbl_max_num_neighbors,
        zbl_bonded_exclusion=args.zbl_bonded_exclusion,
        momentum_weight=args.momentum_weight,
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        checkpoint_name=args.checkpoint_name,
        seed=args.seed,
    )
