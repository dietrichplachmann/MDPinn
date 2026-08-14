#!/usr/bin/env python
"""Empirical sanity check for the ZBL prior's unit-bridging arguments
(distance_scale, energy_scale, cutoff_distance) before trusting them in an
actual training run - this project's own history (the Bohr/Hartree mixup in
waterbox_data.py) is reason enough not to trust a unit-conversion step
without checking it against known physics first (paper/main.tex sec:q4).

Standalone reimplementation of torchmdnet.priors.zbl.ZBL.post_reduce's exact
per-pair math (screening function, cosine cutoff envelope, unit-conversion
constants - pulled directly from torchmd-net's public source, same approach
as waterbox_data.py's own "constructor signature taken from public source,
not verified against an installed copy" caveat) for a SINGLE scalar pair
distance, not the batched/scattered/neighbor-list version - lets this run
without torchmdnet installed (this checkout has neither, see CLAUDE.md), by
sidestepping OptimizedDistance/scatter entirely and just checking the pure
physics formula at specific distances of interest.

Why these specific distances: diagnose_short_range_collapse.py already
established, empirically, how close a real ~300K liquid-water configuration
ever gets (the "floor" values) and roughly where the RDF first-neighbor
peaks sit - the two regimes ZBL needs to tell apart. A good
(cutoff_distance, and implicitly distance_scale/energy_scale) choice should
show:
  - a LARGE repulsive correction right at/below the empirical anomaly
    floors (this is the whole point - supplying the missing short-range
    signal the network's own training data never sampled)
  - a NEGLIGIBLE correction at normal equilibrium non-bonded separations
    (so ZBL doesn't distort physics the network already learned correctly)
  - some correction at the intramolecular O-H covalent bond length
    (~0.96-1.0 A) is EXPECTED and fine - ZBL is a blanket pairwise addition
    with no concept of "this pair is a real bond", exactly as used in
    NequIP/MACE; the network is expected to learn to compensate for it
    during training, the same way it already learns the rest of the energy
    landscape. What would be a problem is bond-distance ZBL energy so large
    it destabilizes early training, not that it's nonzero.

Usage (runs anywhere - no torchmdnet/ase needed):
    python src/verify_zbl_units.py
"""

from __future__ import annotations

import math

# Constants exactly as hardcoded inside torchmdnet.priors.zbl.ZBL.post_reduce
# (verified against torchmd-net's public source on GitHub) - not redefined
# independently, to avoid a second, possibly-inconsistent copy of the same
# magic numbers.
_BOHR_RADIUS_M = 5.29177210903e-11
_ZBL_ENERGY_CONSTANT = 2.30707755e-28  # units: J*m (Coulomb-type prefactor)

# This project's own unit convention (waterbox_data.py): positions in
# Angstrom, energies in eV - so distance_scale/energy_scale bridge those to
# ZBL's required meters/Joules.
ANGSTROM_TO_METER = 1e-10
EV_TO_JOULE = 1.602176634e-19

ATOMIC_NUMBER = {"O": 8, "H": 1}


def cosine_cutoff(distance_angstrom: float, cutoff_upper_angstrom: float) -> float:
    """torchmdnet.models.utils.CosineCutoff with cutoff_lower=0.0 (ZBL's own
    usage: CosineCutoff(cutoff_upper=cutoff_distance), no cutoff_lower
    passed) - the else-branch of that class's forward(), reproduced exactly:
    0.5*(cos(pi*r/r_cut)+1) for r < r_cut, else 0."""
    if distance_angstrom >= cutoff_upper_angstrom:
        return 0.0
    return 0.5 * (math.cos(distance_angstrom * math.pi / cutoff_upper_angstrom) + 1.0)


def zbl_pair_energy_ev(
    distance_angstrom: float,
    z1: int,
    z2: int,
    cutoff_distance_angstrom: float,
    distance_scale: float = ANGSTROM_TO_METER,
    energy_scale: float = EV_TO_JOULE,
) -> float:
    """Single-pair ZBL repulsion energy in eV, replicating
    torchmdnet.priors.zbl.ZBL.post_reduce's math exactly for one pair (no
    neighbor list, no scatter, no factor-of-0.5 double-counting correction -
    that 0.5 in the original only exists because it sums over every ordered
    (i,j)/(j,i) pair; a single unordered pair here needs none of it)."""
    a = 0.8854 * _BOHR_RADIUS_M / (z1**0.23 + z2**0.23)
    distance_m = distance_angstrom * distance_scale
    d = distance_m / a
    f = (
        0.1818 * math.exp(-3.2 * d)
        + 0.5099 * math.exp(-0.9423 * d)
        + 0.2802 * math.exp(-0.4029 * d)
        + 0.02817 * math.exp(-0.2016 * d)
    )
    f *= cosine_cutoff(distance_angstrom, cutoff_distance_angstrom)
    # distance here is intentionally the DATASET-unit (Angstrom) distance,
    # not distance_m - mirrors the original exactly, where `distance`
    # (dataset units) is divided into, and the unit-bridging happens only in
    # the leading constant below.
    energy_raw = f * z1 * z2 / distance_angstrom
    energy_ev = (_ZBL_ENERGY_CONSTANT / energy_scale / distance_scale) * energy_raw
    return energy_ev


# Empirical reference points, in Angstrom:
# - "floor_*": diagnose_short_range_collapse.py's p1.0 empirical floors
#   (results/short_range_diagnostic/summary.md, corrected run) - the
#   closest a real ~300K liquid-water configuration's own minimum
#   cross-molecule approach gets.
# - "equilibrium_*": typical real liquid-water separations (O-O RDF first
#   peak ~2.8 A; O-H hydrogen-bond distance ~1.8-2.0 A; H-H ~2.3-2.5 A;
#   values are standard liquid-water literature figures, used here only to
#   pick a cutoff that stays out of the way of normal chemistry, not as a
#   claim requiring the same "check against real data" discipline as an
#   actual training assumption).
# - "bond_OH": the covalent O-H bond length (~0.96-1.0 A) - see module
#   docstring on why a nonzero ZBL contribution here is fine, not a bug.
REFERENCE_DISTANCES_ANGSTROM = {
    ("O", "O"): {"floor": 1.52, "equilibrium": 2.8},
    ("O", "H"): {"floor": 0.95, "bond": 0.98, "equilibrium_hbond": 1.9},
    ("H", "H"): {"floor": 0.79, "bond_pair_intramolecular": 1.51, "equilibrium": 2.4},
}


def report(cutoff_distance_angstrom: float):
    print(f"\n=== cutoff_distance = {cutoff_distance_angstrom} A ===")
    print(f"{'pair':6s} {'regime':28s} {'r (A)':8s} {'ZBL energy (eV)':>16s}")
    for (elem1, elem2), distances in REFERENCE_DISTANCES_ANGSTROM.items():
        z1, z2 = ATOMIC_NUMBER[elem1], ATOMIC_NUMBER[elem2]
        for regime, r in distances.items():
            e = zbl_pair_energy_ev(r, z1, z2, cutoff_distance_angstrom)
            print(f"{elem1}-{elem2:4s} {regime:28s} {r:8.3f} {e:16.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cutoff-distances", type=str, default="2.0,2.5,3.0,4.0,5.0",
        help="Comma-separated candidate cutoff_distance values (Angstrom) to compare.",
    )
    args = parser.parse_args()

    for cutoff in [float(x) for x in args.cutoff_distances.split(",")]:
        report(cutoff)

    print(
        "\nLook for: a cutoff where 'floor' rows show a clearly non-negligible (repulsive, "
        "positive-energy) correction, 'equilibrium'/'equilibrium_hbond' rows are near-zero "
        "(ZBL should stay out of the way of normal chemistry), and 'bond'/'bond_pair_"
        "intramolecular' rows are nonzero but not so large they'd look like a training-"
        "destabilizing outlier next to the model's own eV-scale energies."
    )
