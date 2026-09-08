# Water-box rollout stability study summary (mean +/- std across replicates)

n=5 replicates per condition. Same training seed/condition, before (before_finetune=checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt) vs. after (after_finetune=checkpoints/waterbox_study_zbl_bonded_ext70_stable_v6_diag/water_absolute/seed1/stable_final.ckpt) StABlE fine-tuning. velocity-axis replicates at dt=0.1, matching results/waterbox_rollout_study_zbl_bonded_ext70_seed1's own DATA_SEED/test_config_index exactly for direct comparability.

| condition | n_replicates | drift_ev_per_atom_mev | drift_fraction_pct | plateau_temperature_mean | plateau_temperature_std |
| --- | --- | --- | --- | --- | --- |
| after_finetune | 5 | 0.01698 +/- 0.039 | 1.086e-05 +/- 2.5e-05 | 768.5 +/- 11 | 27.85 +/- 4.1 |
| before_finetune | 5 | 0.003051 +/- 0.033 | 1.952e-06 +/- 2.1e-05 | 775.2 +/- 27 | 31.52 +/- 2.5 |
