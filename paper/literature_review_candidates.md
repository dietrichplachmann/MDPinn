# Literature review candidate pool

Compiled from six parallel research passes (2026-08-18). Every source below was checked against a primary record (arXiv abstract/HTML, publisher DOI page, or — for the ZBL implementation claims — raw GitHub source fetched directly) by the reviewing agent, not accepted from a search-engine summary alone. Items explicitly flagged as unverified are marked **[UNVERIFIED — check before citing]** and should not be added to `main.tex` without independent confirmation.

This is a candidate pool, not a final citation list — pull from it, don't dump all of it into the paper. Existing bibliography entries (`md17`, `cheng2019water`, `tensornet`, `torchmdnet2`, `mace`, `mlip-validation`, `mlip-error-metrics`, `nequip`, `adaptive-weighting`, `ase`, `fu2022forces`, `zbl1985`, `shortrangeprior2025`, `nequip-code`) are not repeated here except where a review found a *better* or *additional* source for a point currently backed only by one of these.

---

## 0. PRIORITY: Bonded-exclusion ZBL (directly informs the planned implementation)

**Headline finding: no MLIP framework checked implements bonded-pair exclusion in its ZBL prior.** Checked NequIP's and MACE's actual source directly:

- **NequIP** (`nequip/nn/pair_potential.py`, class `ZBL`) — uses a `PolynomialCutoff` applied to distance *normalized by the sum of covalent radii*, plus the ZBL screening function. No exclusion list, no bonded-pair/topology concept anywhere.
- **MACE** (`mace/modules/radial.py`, class `ZBLBasis`) — identical pattern: covalent-radii-normalized cutoff envelope, no `exclude_types`, no connectivity awareness.
- **MACE-OFF** (Kovács et al., JACS 2025, arXiv:2312.15211) — MACE's own flagship model for covalent *organic* molecules. Full text checked: **no mention of ZBL/pair_repulsion at all.** The team that owns MACE's ZBL implementation does not enable it for exactly this class of system.
- **DP-ZBL / NEP-ZBL** (Yan, Fan, Zhu 2025, arXiv:2504.15925 — already cited as `shortrangeprior2025`) — confirmed tested only on LLZO, an ionic ceramic solid electrolyte with no covalent-bond network to collide with.
- GRACE has a `ZBLPotential` class but its exclusion behavior (if any) could not be confirmed from available source.

**Key implication for our diagnosis**: NequIP/MACE's covalent-radii-normalized envelope is a *per-element-pair adaptive cutoff*, not a *topological* exclusion — it applies identically whether a pair is bonded or not. Given our own finding that the O–H bond length (~0.98 Å) sits inside the same range as the anomalous non-bonded floor (~0.8–1.8 Å), that mechanism would not have resolved our overlap problem either.

**Classical-MD precedent (the actual fix)**: LAMMPS `special_bonds` documentation confirms CHARMM and AMBER both use a weighting factor of **0.0 for both 1-2 and 1-3 nonbonded pairs** (AMBER: 0.5/0.833 for 1-4), built from an exclusion list derived from bond topology, specifically to avoid double-counting energy already described by bonded/angle terms. This is the direct, 60-year-old precedent for what a bonded-exclusion ZBL should do.

**No prior report of this failure mode found.** No paper documents ZBL-on-a-bonded-system backfiring the way we observed. Circumstantial support that it's a real, under-reported gap: MACE-OFF avoids ZBL for covalent organics entirely; DP-ZBL/NEP-ZBL is unvalidated outside ionic/metallic systems; no warning found in NequIP/MACE/DeePMD docs. **This appears to be a genuinely underexplored failure mode, worth stating as such.**

**Smooth alternatives to a hard exclusion list** (secondary/later refinement, not the starting point):
- **PFP/Matlantis** (Takamoto et al., *Nat. Commun.* 2022) uses a **Morse**-style two-body correction instead of ZBL — Morse has an attractive minimum near the bond distance (built to look like a bond) rather than a pure repulsive wall, so it structurally can't have our exact failure mode. Targets organic/molecular systems.
- **MLBOP** (arXiv:2501.11297, 2025) — ML-learned bond order smoothly modulates a two-body Abell-Tersoff-style term instead of a hard exclusion list. From-scratch design, not a drop-in ZBL patch.
- ~~"Smooth Dynamic Cutoffs for MLIPs" (arXiv:2601.21147)~~ — **[UNVERIFIED — an early search summary claimed this paper states MACE/NEP "may remain unstable with ZBL potential"; the agent fetched the actual text and could not find that passage. Do not cite for that claim.]**

**Recommended implementation path**: subclass `torchmdnet/priors/zbl.py`, precompute a static boolean exclusion mask over atom-index pairs from the bond inference already trusted elsewhere in this project (`structural_metrics.infer_bonds`), zero the ZBL contribution for 1-2 (and consider 1-3) pairs — a direct, low-risk port of the CHARMM/AMBER convention. A distance-threshold heuristic is *not* an alternative here — that's precisely the mechanism that just failed, since bond length and anomaly floor overlap in distance space for this system; only topology can separate them.

---

## 1. Physics-informed / conservation-law soft-constraint losses (the paper's actual core topic — currently almost uncited)

**PINN framing, general:**
- Raissi, Perdikaris, Karniadakis — "Physics-informed neural networks..." *J. Comput. Phys.* 378, 686–707 (2019). The original PINN paper — the loss-term-penalizing-a-physical-law methodology the project's momentum loss is a direct instance of.
- Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang — "Physics-informed machine learning." *Nature Reviews Physics* 3, 422–440 (2021). DOI: 10.1038/s42254-021-00314-5. Foundational review.

**Directly NNIP-relevant:**
- Takamoto, Zaverkin, Niepert — "Physics-Informed Weakly Supervised Learning for Interatomic Potentials." arXiv:2408.05215 (2024), ICML 2025. Adds physics-informed loss terms (energy-extrapolation + conservative-force consistency) to MLIP training — close methodological cousin.
- Zhang, Chigaev, Isayev, Messerly, Lubbers — "Including Physics-Informed Atomization Constraints in Neural Networks for Reactive Chemistry." *J. Chem. Inf. Model.* 65(9), 4367–4380 (2025). DOI: 10.1021/acs.jcim.5c00341. Hard-bakes an atomization-energy constraint into HIP-NN/ANI; reports the constraint helps in some regimes and never hurts — mirrors our own "modest, real, but confounded" finding.
- Baez, Zhang, Ma, Das, Nguyen, Daniel — "Guaranteeing Conservation Laws with Projection in Physics-Informed Neural Networks." arXiv:2410.17445 (2024). Shows a *soft* penalty-based conservation loss (ordinary PINN) fails to reliably conserve momentum; proposes hard projection instead — directly supports our redundancy argument (soft loss terms are a weak lever vs. architectural guarantees).
- Sharma, Fink — "A physics-informed graph neural network conserving linear and angular momentum for dynamical systems" (Dynami-CAL GraphNet). *Nat. Commun.* (2026); arXiv:2501.07373. Bakes momentum conservation into a GNN via equivariant pairwise reference frames rather than a loss penalty.

**Equivariance ⇒ conservation "for free" (theory + architecture lineage):**
- "The principles behind equivariant neural networks for physics and chemistry." *PNAS* (Oct. 2025), DOI: 10.1073/pnas.2415656122. Clean recent theoretical treatment — best single citation for the general argument.
- Thomas, Smidt, Kearnes, Yang, Li, Kohlhoff, Riley — "Tensor Field Networks..." arXiv:1802.08219 (2018). Root of the spherical-harmonic/Clebsch–Gordan machinery TensorNet/NequIP/MACE build on.
- Fuchs, Worrall, Fischer, Welling — "SE(3)-Transformers..." arXiv:2006.10503 (2020), NeurIPS 2020.
- Anderson, Hy, Kondor — "Cormorant: Covariant Molecular Neural Networks." NeurIPS 2019.
- Schütt, Unke, Gastegger — "Equivariant message passing..." (PaiNN). arXiv:2102.03150 (2021), ICML 2021.

**Adjacent field, multi-body momentum conservation (worth citing as the closest existing precedent for the whole-vs-subunit question):**
- Prantl, Ummenhofer, Koltun, Thuerey — "Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics" (DMCF). arXiv:2210.06036 (2022), NeurIPS 2022. Tests momentum conservation (hard constraint) in a multi-particle/multi-body learned-simulation setting; shows measurable improvement — the clearest adjacent-field precedent for "does enforcing momentum conservation in a multi-body system change behavior," though continuum/SPH, not atomistic.

**Architecturally-guaranteed vs. must-be-trained-for properties (general framing):**
- Unke, Chmiela, Sauceda, Gastegger, Poltavsky, Schütt, Tkatchenko, Müller — "Machine Learning Force Fields." *Chem. Rev.* 121(16), 10142–10186 (2021). DOI: 10.1021/acs.chemrev.0c01111. The standard comprehensive review; explicitly separates architecturally-enforced properties from trained-for ones.
- "Six Open Questions in Machine-Learned Interatomic Potential Foundation Models." arXiv:2606.07327 (2026). Recent field-wide perspective on this exact distinction.

**Energy-force consistency (closest cousin of momentum consistency):**
- Chmiela, Tkatchenko, Sauceda, Poltavsky, Schütt, Müller — "Machine learning of accurate energy-conserving molecular force fields" (GDML/sGDML). *Sci. Adv.* 3(5), e1603015 (2017); arXiv:1611.04678. [Note: this is the same underlying dataset lineage as `md17`, already cited — but this specific paper is about the conservative-forces design principle, not the dataset.]
- Bigi, Langer, Ceriotti — "The dark side of the forces: assessing non-conservative force models for atomistic machine learning." arXiv:2412.11569 (2024), ICML 2025 oral. Documents concrete failure modes (bad geometry optimization, MD instability) from broken energy-force consistency — structurally parallel to our momentum-consistency story.

**Honest gap, worth stating rather than papering over**: no paper found running our exact experiment (comparing whole-system vs. per-subunit momentum-conservation loss on an NNIP, isolated-molecule or periodic/multi-molecule). This looks like a legitimate, defensible novelty claim, not a search failure.

---

## 2. Delta-learning and the CHARMM36/CGenFF baseline (currently uncited entirely)

**Δ-ML foundations — well covered:**
- Ramakrishnan, Dral, Rupp, von Lilienfeld — "Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach." *J. Chem. Theory Comput.* 2015, arXiv:1503.04987. The foundational paper.
- Huang, Hou, Dral — "Active delta-learning for fast construction of interatomic potentials and stable molecular dynamics simulations." arXiv:2505.08195, DOI: 10.1088/2632-2153/adeb46. Extends Δ-learning to ANI-type NNIPs; ~10x fewer points needed, and delta-learned models notably *more stable in MD* than direct-learned ones.
- Bowman, Qu, Conte, Nandi, Houston, Yu — "Δ-Machine Learned Potential Energy Surfaces and Force Fields." *J. Chem. Theory Comput.* 2022, DOI: 10.1021/acs.jctc.2c01034 (Perspective).

**Delta-learning against a *classical* (not cheaper-QM) baseline — thinner, no exact match to our setup:**
- Jiang, Chen, Zeng, Cao, et al. — "Deep Residual Learning for Molecular Force Fields" (ResFF). *Nat. Commun.* 2026, DOI: 10.1038/s41467-026-74983-0. Closest match: hybrid model, learnable MM covalent terms + NN residual, jointly optimized.
- **Doerr, Majewski, Pérez, Krämer, Clementi, Noé, Giorgino, De Fabritiis — TorchMD.** *J. Chem. Theory Comput.* 2021, DOI: 10.1021/acs.jctc.0c01343, arXiv:2012.12106. U = U_θ(NN) + U_λ(prior); prior has bonded harmonic + nonbonded repulsive terms subtracted before NN training ("delta forces"). **Closest published architectural analog to our aborted approach** — from the same TorchMD family this project already builds on.
- Karwounopoulos, Wu, Tkaczyk, Wang, Baskerville, Ranasinghe, Langer, Wood, Wieder, Boresch — "Insights and Challenges in Correcting Force Field Based Solvation Free Energies Using a Neural Network Potential." *J. Phys. Chem. B* 2024, DOI: 10.1021/acs.jpcb.4c01417. **The one genuine classical-FF-baseline comparison found**: ANI-2x residual correction over CGenFF/OpenFF gave a statistically *insignificant* improvement across 589 molecules — good literature-backed caution that our abandoned delta-learning direction wasn't necessarily leaving something on the table.

**CHARMM36 / CGenFF primary citations — fully covered:**
- Best, Zhu, Shim, Lopes, Mittal, Feig, MacKerell Jr. — CHARMM36 protein FF. *J. Chem. Theory Comput.* 2012, DOI: 10.1021/ct300400x.
- **Vanommeslaeghe, Hatcher, Acharya, Kundu, Zhong, Shim, Darian, Guvench, Lopes, Vorobyov, MacKerell Jr. — "CHARMM general force field..." (CGenFF).** *J. Comput. Chem.* 2010, 31:671, DOI: 10.1002/jcc.21367. **This is the one to cite for the aspirin baseline specifically.**
- Vanommeslaeghe, MacKerell Jr. — CGenFF automation Parts I & II. *J. Chem. Inf. Model.* 2012, 52:3144 (DOI: 10.1021/ci300363c) and 52:3155 (DOI: 10.1021/ci3003649).

**O(N²) nonbonded cost and why it need not be fatal (context for why our unvectorized implementation, not the algorithm, was the real problem):**
- Verlet — "Computer 'Experiments' on Classical Fluids I." *Phys. Rev.* 159:98 (1967). Origin of the neighbor list.
- Darden, York, Pedersen — Particle Mesh Ewald. *J. Chem. Phys.* 98:10089 (1993), DOI: 10.1063/1.464397.
- Cheng — "Latent Ewald summation for machine learning of long-range interactions." arXiv:2408.15165 (2025). Explicitly notes Ewald summation's own scaling isn't the real MLIP bottleneck (~2x a short-range-only model) — supports the point that a vectorized baseline would likely have made delta-learning tractable.

---

## 3. MLIP rollout-stability and RDF evaluation methodology (beyond Fu et al. 2022 / Morrow-Gardner-Deringer, already cited)

**More rollout-stability benchmarks, current through 2026:**
- Stocker, Gasteiger, Becker, Günnemann, Margraf — "How robust are modern graph neural network potentials in long and hot molecular dynamics simulations?" *Machine Learning: Science and Technology* 3, 045010 (2022). DOI: 10.1088/2632-2153/ac9955. 280 ns GemNet MD; low static error doesn't predict stability; pathologies emerge after 100s of ps. Strong companion to Fu et al.
- Ranasinghe, Baskerville, Wood, König — "Basic stability tests of machine learning potentials for molecular simulations in computational drug discovery." *J. Chem. Inf. Model.* 65(17) (2025), arXiv:2503.11537. Proposes a standardized stability-test protocol; **also directly reports that ANI-2x and an in-house MACE model fail to reproduce liquid water's structure, forming an amorphous solid instead** — direct precedent that this project's finding (water-MLIP dynamical instability) is not unique to this project.
- "Crash testing machine learning force fields..." (TEA Challenge 2023), two parts, *Chemical Science* (2025), DOI: 10.1039/D4SC06529H and D4SC06530A.
- Riebesell, Goodall, Benner, Chiang, Deng, Ceder, Asta, Lee, Jain, Persson — Matbench Discovery. arXiv:2308.14920 (2023); live leaderboard now scores an MD task against RDF/ADF/vDOS/pressure — a rollout-stability-as-metric leaderboard in the spirit this project wants.
- Yuan, Head-Gordon — "Teachers that teach the irrelevant: Pre-training machine learned interaction potentials with classical force fields for robust molecular dynamics simulations." arXiv:2509.14205 (2025). Frames the exact problem this project hit and tests liquid water specifically.

**RDF as a validation metric for liquid water:**
- Soper — "The radial distribution functions of water and ice from 220 to 673 K..." *Chem. Phys.* 258:121 (2000). Canonical experimental RDF reference.
- Omranpour, Montero De Hijes, Behler, Dellago — "Perspective: Atomistic Simulations of Water and Aqueous Systems with Machine Learning Potentials." arXiv:2401.17875 (2024). Field-wide review, strong general citation.
- Montero de Hijes, Dellago, Jinnouchi, Schmiedmayer, Kresse — "Comparing machine learning potentials for water..." *J. Chem. Phys.* 160, 114107 (2024), arXiv:2312.15213. Cross-architecture RDF comparison; argues RMSE alone is insufficient; also reports *decreased* stability for bulk water vs. single-molecule test systems across the potentials compared — another independent instability data point.

**Other ML water potentials for context (beyond MACE, already cited on the same dataset):**
- Morawietz, Singraber, Dellago, Behler — "How van der Waals interactions determine the unique properties of water." *PNAS* 113(30), 8368 (2016). Landmark Behler-Parrinello water NNP.
- Yu, Qu, Houston, Nandi, Pandey, Conte, Bowman — "A Status Report on 'Gold Standard' Machine-Learned Potentials for Water." *J. Phys. Chem. Lett.* 14(36), 8077 (2023).

**Periodic boundary handling in GNN potentials (thinnest angle — mostly implementation docs, not standalone papers):**
- Pelaez, Simeon, Galvelis, Mirarchi, Eastman, Doerr, Thölke, Markland, De Fabritiis — TorchMD-Net 2.0. arXiv:2402.17660 (2024) [note: this project already cites `torchmdnet2` for TorchMD-Net 2.0 — this arXiv ID documents the specific periodic neighbor-search implementation and the half-box-width cutoff constraint, worth citing at the specific point about periodicity implementation rather than only generically].
- e3nn documentation, "Point inputs with periodic boundary conditions" (docs.e3nn.org) — not a paper, but the clearest reference for how equivariant GNNs must add an edge_shift/lattice correction to respect PBC.

**Energy/temperature drift as NVE integrator diagnostics (classical numerics — directly supports the timestep-vs-ZBL-stiffness finding):**
- Engle, Skeel, Drees — "Monitoring energy drift with shadow Hamiltonians." *J. Comput. Phys.* 206(2), 432 (2005). Key reference for distinguishing genuine instability from expected symplectic-integrator drift.
- "Energy drift in molecular dynamics simulations." *BIT Numerical Mathematics* (2007), Springer. ΔE ~ Δt² scaling — directly supports the project's own timestep-too-coarse-for-ZBL finding.

---

## 4. Seed variance and statistical rigor in ML comparisons (backs three of the project's own recurring empirical findings)

**General DL seed-variance/reproducibility — strong, well-verified coverage:**
- Henderson, Islam, Bachman, Pineau, Precup, Meger — "Deep Reinforcement Learning that Matters." arXiv:1709.06560, AAAI 2018. Confirmed real (6 authors). The canonical "seed choice alone can flip which method looks better" citation.
- Picard — "torch.manual_seed(3407) is all you need." arXiv:2109.08203 (2021). Scans up to 10,000 seeds on CIFAR-10; finds outlier seeds performing far above/below average — close analogue to this project's "one noisy outlier seed reversed the n=3 result."
- Bethard — "We need to talk about random seeds." arXiv:2210.13393 (2022). Surveys 85 ACL papers; >50% use seeds in statistically risky ways.
- Bouthillier, Laurent, Vincent, et al. — "Accounting for Variance in Machine Learning Benchmarks." arXiv:2103.03098, MLSys 2021.
- Reimers, Gurevych — "Reporting Score Distributions Makes a Difference..." arXiv:1707.09861, EMNLP 2017. Seed choice alone produces p<10⁻⁴ significant differences.

**MLIP-specific (thinner — the honest gap, not padded):**
- Tan, Urata, Goldman, Dietschreit, Gómez-Bombarelli — "Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles." arXiv:2305.01754, *npj Computational Materials* 9, 225 (2023). Closest MLIP-specific support: compares ensemble-of-seeds vs. single-model UQ across rMD17 and other systems.
- Fu et al. 2022 (already cited) is again relevant here: the dynamics-vs-static-metric divergence is arguably the closest existing analogue to this project's own finding (b) about rollout comparisons not tracking static ones.
- **Honest assessment (from the reviewing agent): no paper found whose central claim is "we retrained an MLIP with different seeds and the benchmark ranking flipped," analogous to Henderson et al. for RL.** The project's own escalating-sample-size-reversal finding may be a genuinely novel empirical contribution in the MLIP space specifically, even though the general phenomenon is well established in DL broadly.

**How many seeds is "enough" (statistical power):**
- Colas, Sigaud, Oudeyer — "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments." arXiv:1806.08295 (2018). Derives via power analysis how many seeds are needed; shows typical practice (3–5 seeds) is often underpowered.
- Agarwal, Schwarzer, Castro, Courville, Bellemare — "Deep Reinforcement Learning at the Edge of the Statistical Precipice." arXiv:2108.13264, NeurIPS 2021 (outstanding paper award). Argues small run counts are common and inadequate; proposes interval estimates.

**Noisy validation curves / checkpoint selection under noise:**
- Izmailov, Podoprikhin, Garipov, Vetrov, Wilson — "Averaging Weights Leads to Wider Optima and Better Generalization" (SWA). arXiv:1803.05407, UAI 2018.
- Wang, Teng, Perdikaris — "Understanding and mitigating gradient pathologies in physics-informed neural networks." arXiv:2001.04536 (2020). Diagnoses imbalanced gradients across composite PINN loss terms causing training instability — directly on-topic given this project's own PINN-style loss.
- Prechelt — "Early Stopping — But When?" in *Neural Networks: Tricks of the Trade*, Springer (1998). The classic methodological reference against naive min-val-loss checkpoint selection.

**Replication-crisis framing (angle 5):**
- Leech, Vazquez, Kupper, Yagudin, Aitchison — "Questionable practices in machine learning." arXiv:2407.12220 (2024). Catalogs 44 questionable practices including seed-related ones.
- Gundersen, Kjensmo — "State of the Art: Reproducibility in Artificial Intelligence." AAAI 32(1) (2018). Peer-reviewed survey of 400 IJCAI/AAAI papers on reproducibility documentation.

---

## 5. Loss weighting and checkpoint selection under annealing (backs the anneal schedule and the anneal-invariant checkpoint fix)

**General multi-task loss balancing — strong coverage:**
- Kendall, Gal, Cipolla — "Multi-Task Learning Using Uncertainty to Weigh Losses..." CVPR 2018, arXiv:1705.07115. Confirmed real.
- Chen, Badrinarayanan, Lee, Rabinovich — "GradNorm..." ICML 2018, arXiv:1711.02257.
- Sener, Koltun — "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018, arXiv:1810.04650.

**NNIP-specific energy/force weighting — a possibly better canonical citation than treating this as MACE/NequIP folklore:**
- **Vita, Schwalbe-Koda — "Data efficiency and extrapolation trends in neural network interatomic potentials."** *Machine Learning: Science and Technology* 4, 035031 (2023), DOI: 10.1088/2632-2153/acf115. Explicitly and generally (architecture-independent) recommends the exact force-heavy-then-anneal-to-energy-heavy strategy this project implemented, citing a specific 1:10 → 1000:1 schedule. **Consider citing this instead of, or alongside, MACE/NequIP as the canonical reference for the annealing idea.**
- Devereux, Yang, Martí, Zádor, Eldred, Najm — "Force Training Neural Network Potential Energy Surface Models." arXiv:2311.07910 (2023). Dedicated study of the force:energy loss ratio's effect, with loss-scale normalization discussion.

**Curriculum learning / loss-weight annealing lineage (a genuinely different-subfield precedent worth citing for the underlying idea):**
- Bengio, Louradour, Collobert, Weston — "Curriculum Learning." ICML 2009. Foundational.
- **Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio — "Generating Sentences from a Continuous Space."** CoNLL 2016, arXiv:1511.06349. The original "KL cost annealing" paper — ramps a loss term's weight from 0 over training to fix a training pathology. The closest general-DL precedent for the *loss-weight-curriculum* concept (as opposed to data-curriculum), predating MACE/NequIP's own schedules.
- Fu, Li, Liu, Gao, Celikyilmaz, Carin — "Cyclical Annealing Schedule..." NAACL 2019, arXiv:1903.10145. Follow-up making the annealing cyclical.

**Contrast citations — worth noting the technique is not universal (itself informative):**
- Allegro (Musaelian et al., *Nat. Commun.* 14:579, 2023) — confirmed to use **equal, fixed** energy/force weighting, no annealing.
- CHGNet (Deng et al., *Nat. Mach. Intell.* 2023, arXiv:2302.14231) — fixed weighted sum, no confirmed annealing.

**Checkpoint selection under a metric that changes definition mid-training (thinnest angle, honestly reported):**
- Apicella, Isgrò, Pollastro, Prevete — "Don't stop me now: Rethinking Validation Criteria for Model Parameter Selection." arXiv:2602.22107 (2026). Closest general methodology paper found; does not address curriculum/annealing-induced metric drift specifically.
- **Agent's honest note: no paper found addressing the exact scenario (a loss-weight-annealing schedule biasing "best checkpoint" selection) head-on.** The anneal-invariant checkpoint-selection fix may be best presented as this project's own contribution, loosely supported by the KL-annealing lineage's implicit acknowledgment of the same "yardstick moves" problem, rather than forced onto a citation that doesn't quite fit.

---

## Cross-cutting notes

- Three sources recur across multiple sections and are worth citing once each in the most relevant place, not repeatedly: **Fu et al. 2022** (already cited), **the TorchMD paper** (delta-learning architecture + general TorchMD-family context), and **Unke et al.'s *Chem. Rev.* MLFF review** (general background for both the physics-informed-loss and architecturally-guaranteed-properties discussions).
- Two items were explicitly flagged as unverified by the reviewing agents and should not be cited without independent confirmation: the "Smooth Dynamic Cutoffs" ZBL-instability claim (§0), and a MOB-ML size-extensivity item originally surfaced only through secondary summaries (§1, omitted above after the flag — see agent transcript if needed).
- A few citations (the TIP4P-2005/SWM4-DP/BK3 water-model comparison papers in §3) had titles/URLs confirmed but full author lists not independently re-verified — do a quick confirmation pass before formal citation.
