# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Same training seed/condition, before (before_finetune=checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt) vs. after (after_finetune=checkpoints/waterbox_study_zbl_bonded_ext70_stable/water_absolute/seed1/stable_final.ckpt) StABlE fine-tuning. velocity-axis replicates at dt=0.1, matching results/waterbox_rollout_study_zbl_bonded_ext70_seed1's own DATA_SEED/test_config_index exactly for direct comparability.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| after_finetune | 5 | -0.02564 +/- 0.035 | -1.64e-05 +/- 2.2e-05 | 768.2 +/- 19 | 29.63 +/- 2.1 |
| before_finetune | 5 | 0.0775 +/- 0.19 | 4.958e-05 +/- 0.00012 | 772.1 +/- 20 | 28.59 +/- 1.6 |
