#!/usr/bin/env python
"""
Analytic baseline potentials used by delta-learning.

Primary mode:
- Aspirin baseline parsed from the uploaded GROMACS/CHARMM36+CGenFF files.

Fallback mode:
- Simple Lennard-Jones 12-6 with cutoff for unsupported molecules.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
ASPIRIN_TOP = DATA_ROOT / "aspirin_gmx.top"
CHARMM36_DIR = DATA_ROOT / "charmm36.ff"
FF_BONDED = CHARMM36_DIR / "ffbonded.itp"
FF_NONBONDED = CHARMM36_DIR / "ffnonbonded.itp"
ASP_FF_BONDED = CHARMM36_DIR / "asp_ffbonded.itp"
ASPIRIN_OFFSET = DATA_ROOT / "aspirin_reference_offset.json"
OFFSET_CACHE = DATA_ROOT / "reference_offset_cache.json"
KJMOL_TO_EV = 0.01036427230133138
ASPIRIN_DEFAULT_ENERGY_OFFSET_EV = 406757.03125
_ATOM_ORDER_CACHE: dict[tuple[int, ...], list[int]] = {}


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


def _read_gmx_sections(path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current = None
    with open(path, "r") as handle:
        for raw_line in handle:
            line = _strip_comment(raw_line)
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line.strip("[]").strip().lower()
                sections.setdefault(current, [])
                continue
            if current is None:
                continue
            sections[current].append(line.split())
    return sections


def _append_section_entries(target: dict[str, list[list[str]]], path: Path):
    parsed = _read_gmx_sections(path)
    for section, entries in parsed.items():
        target.setdefault(section, []).extend(entries)


def _periodic_table() -> dict[str, int]:
    return {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "P": 15,
        "S": 16,
        "CL": 17,
        "BR": 35,
        "I": 53,
    }


def _element_from_atomtype(atomtype: str) -> int:
    token = atomtype.upper()
    if token.startswith("CL"):
        return 17
    if token.startswith("BR"):
        return 35
    if token.startswith("NA"):
        return 11
    if token.startswith("MG"):
        return 12
    if token.startswith("CA") and token not in {"CA", "CAD", "CAI", "CAP", "CAL"}:
        return 20
    symbol = token[0]
    return _periodic_table().get(symbol, 0)


def _covalent_radius_by_z(z: int) -> float:
    return {
        1: 0.31,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        9: 0.57,
        15: 1.07,
        16: 1.05,
        17: 1.02,
    }.get(z, 0.75)


def _max_valence_by_z(z: int) -> int:
    return {
        1: 1,
        6: 4,
        7: 4,
        8: 2,
        9: 1,
        15: 5,
        16: 6,
        17: 1,
    }.get(z, 4)


def _bond_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _angle_key(a: str, b: str, c: str) -> tuple[str, str, str]:
    forward = (a, b, c)
    reverse = (c, b, a)
    return forward if forward <= reverse else reverse


def _match_dihedral(entry_types: tuple[str, ...], query: tuple[str, ...]) -> bool:
    return all(e == q or e.upper() == "X" for e, q in zip(entry_types, query))


def _dihedral_score(entry_types: tuple[str, ...], query: tuple[str, ...]) -> int:
    return sum(1 for e, q in zip(entry_types, query) if e == q)


@lru_cache(maxsize=None)
def load_reference_energy_offset_eV(molecule: str | None = None) -> float:
    if (molecule or "").lower() != "aspirin":
        return 0.0
    if not ASPIRIN_OFFSET.exists():
        return ASPIRIN_DEFAULT_ENERGY_OFFSET_EV
    with open(ASPIRIN_OFFSET, "r") as handle:
        payload = json.load(handle)
    return float(payload.get("energy_offset_eV", ASPIRIN_DEFAULT_ENERGY_OFFSET_EV))


def _load_offset_cache() -> dict:
    if not OFFSET_CACHE.exists():
        return {}
    with open(OFFSET_CACHE, "r") as handle:
        return json.load(handle)


def _write_offset_cache(payload: dict):
    OFFSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(OFFSET_CACHE, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _offset_cache_key(
    molecule: str,
    dataset: str,
    epsilon_eV: float,
    sigma_A: float,
    r_cut_A: float,
    sample_count: int,
    train_frac: float,
    val_frac: float,
) -> str:
    return "|".join(
        [
            f"molecule={molecule.lower()}",
            f"dataset={dataset}",
            f"eps={epsilon_eV:.12g}",
            f"sigma={sigma_A:.12g}",
            f"cutoff={r_cut_A:.12g}",
            f"samples={sample_count}",
            f"train_frac={train_frac:.6f}",
            f"val_frac={val_frac:.6f}",
        ]
    )


def calibrate_reference_energy_offset_eV(
    molecule: str,
    dataset: str = "MD17",
    data_root: str | Path = "./data",
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
    sample_count: int = 2048,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    force_recompute: bool = False,
) -> float:
    """Estimate a parameter-specific constant energy zero for delta-learning.

    The delta target is `E_true - E_ref`. For a baseline with the correct energy
    zero, the mean residual over the train split should be near zero. We therefore
    estimate the needed offset as mean(raw_baseline_energy - reference_energy).

    The estimate is cached on disk because this is expensive and may be reused by
    Optuna reruns that revisit the same baseline hyperparameters.
    """
    if (molecule or "").lower() != "aspirin":
        return load_reference_energy_offset_eV(molecule)
    if dataset != "MD17":
        raise NotImplementedError(
            f"Offset calibration currently supports aspirin on MD17 only (got dataset='{dataset}')."
        )

    cache_key = _offset_cache_key(
        molecule=molecule,
        dataset=dataset,
        epsilon_eV=float(epsilon_eV),
        sigma_A=float(sigma_A),
        r_cut_A=float(r_cut_A),
        sample_count=int(sample_count),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
    )
    cache = _load_offset_cache()
    if not force_recompute and cache_key in cache:
        return float(cache[cache_key]["energy_offset_eV"])

    from torchmdnet.datasets import MD17
    from data_splits import contiguous_split

    full_dataset = MD17(root=str(data_root), molecules=molecule)
    train_data, _, _ = contiguous_split(full_dataset, train_frac=train_frac, val_frac=val_frac)
    train_indices = list(getattr(train_data, "indices", []))
    if not train_indices:
        raise RuntimeError("Could not build a non-empty train split for offset calibration.")

    requested_sample_count = max(1, min(int(sample_count), len(train_indices)))
    if requested_sample_count == len(train_indices):
        sample_indices = train_indices
    else:
        step = (len(train_indices) - 1) / float(requested_sample_count - 1) if requested_sample_count > 1 else 0.0
        sample_indices = [train_indices[int(round(i * step))] for i in range(requested_sample_count)]
        sample_indices = list(dict.fromkeys(sample_indices))
    actual_sample_count = len(sample_indices)

    raw_minus_true = []
    for dataset_idx in sample_indices:
        sample = full_dataset[int(dataset_idx)]
        y_true = sample.y
        y_true = float(y_true.item() if hasattr(y_true, "item") else y_true[0])
        raw_energy, _ = reference_energy_forces(
            z=sample.z,
            pos=sample.pos,
            molecule=molecule,
            box_l=getattr(sample, "box", None),
            epsilon_eV=float(epsilon_eV),
            sigma_A=float(sigma_A),
            r_cut_A=float(r_cut_A),
            energy_offset_eV=0.0,
        )
        raw_minus_true.append(float(raw_energy.item()) - y_true)

    offset = float(sum(raw_minus_true) / len(raw_minus_true))
    cache[cache_key] = {
        "energy_offset_eV": offset,
        "molecule": molecule,
        "dataset": dataset,
        "epsilon_eV": float(epsilon_eV),
        "sigma_A": float(sigma_A),
        "r_cut_A": float(r_cut_A),
        "sample_count": int(actual_sample_count),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
    }
    _write_offset_cache(cache)
    return offset


@lru_cache(maxsize=1)
def _load_aspirin_forcefield() -> dict:
    if not ASPIRIN_TOP.exists():
        raise FileNotFoundError(f"Missing aspirin topology: {ASPIRIN_TOP}")

    top_sections = _read_gmx_sections(ASPIRIN_TOP)
    ff_sections: dict[str, list[list[str]]] = {}
    _append_section_entries(ff_sections, FF_BONDED)
    _append_section_entries(ff_sections, ASP_FF_BONDED)
    _append_section_entries(ff_sections, FF_NONBONDED)

    atoms = []
    atomtypes = []
    charges = []
    masses = []
    z_numbers = []
    for entry in top_sections.get("atoms", []):
        atomtype = entry[1]
        atomtypes.append(atomtype)
        charges.append(float(entry[6]))
        masses.append(float(entry[7]))
        atoms.append(entry[4])
        z_numbers.append(_element_from_atomtype(atomtype))

    bonds = [(int(e[0]) - 1, int(e[1]) - 1, int(e[2])) for e in top_sections.get("bonds", [])]
    pairs = {tuple(sorted((int(e[0]) - 1, int(e[1]) - 1))) for e in top_sections.get("pairs", [])}
    angles = [(int(e[0]) - 1, int(e[1]) - 1, int(e[2]) - 1, int(e[3])) for e in top_sections.get("angles", [])]
    molecule_section = top_sections.get("moleculetype", [])
    nrexcl = int(molecule_section[0][1]) if molecule_section else 3

    proper_dihedrals = []
    improper_dihedrals = []
    for entry in top_sections.get("dihedrals", []):
        item = (int(entry[0]) - 1, int(entry[1]) - 1, int(entry[2]) - 1, int(entry[3]) - 1, int(entry[4]))
        if item[4] == 9:
            proper_dihedrals.append(item)
        elif item[4] == 2:
            improper_dihedrals.append(item)

    bondtypes = {}
    for entry in ff_sections.get("bondtypes", []):
        bondtypes[_bond_key(entry[0], entry[1])] = {
            "func": int(entry[2]),
            "b0_nm": float(entry[3]),
            "kb_kjmol_nm2": float(entry[4]),
        }

    angletypes = {}
    for entry in ff_sections.get("angletypes", []):
        angletypes[_angle_key(entry[0], entry[1], entry[2])] = {
            "func": int(entry[3]),
            "theta0_deg": float(entry[4]),
            "k_theta_kjmol_rad2": float(entry[5]),
            "ub0_nm": float(entry[6]) if len(entry) > 6 else 0.0,
            "kub_kjmol_nm2": float(entry[7]) if len(entry) > 7 else 0.0,
        }

    proper_types = []
    improper_types = []
    for entry in ff_sections.get("dihedraltypes", []):
        record = {
            "types": tuple(entry[:4]),
            "func": int(entry[4]),
            "phi0_deg": float(entry[5]),
            "k_phi_kjmol": float(entry[6]),
        }
        if record["func"] == 9:
            record["mult"] = int(entry[7])
            proper_types.append(record)
        elif record["func"] == 2:
            improper_types.append(record)

    nonbonded = {}
    for entry in ff_sections.get("atomtypes", []):
        nonbonded[entry[0]] = {
            "sigma_nm": float(entry[5]),
            "epsilon_kjmol": float(entry[6]),
        }

    z_signature = tuple(z_numbers)
    return {
        "z": z_signature,
        "atom_names": atoms,
        "atom_types": atomtypes,
        "charges": charges,
        "masses": masses,
        "nrexcl": nrexcl,
        "bonds": bonds,
        "pairs": pairs,
        "angles": angles,
        "proper_dihedrals": proper_dihedrals,
        "improper_dihedrals": improper_dihedrals,
        "bondtypes": bondtypes,
        "angletypes": angletypes,
        "proper_types": proper_types,
        "improper_types": improper_types,
        "nonbonded": nonbonded,
    }


def _infer_bonds_from_positions(z: tuple[int, ...], pos: torch.Tensor) -> list[tuple[int, int]]:
    n_atoms = len(z)
    candidates = []
    for i in range(n_atoms):
        zi = z[i]
        for j in range(i + 1, n_atoms):
            zj = z[j]
            threshold = 1.25 * (_covalent_radius_by_z(zi) + _covalent_radius_by_z(zj))
            dist = float(torch.linalg.norm(pos[i] - pos[j]).item())
            if dist <= threshold:
                candidates.append((dist, i, j))

    candidates.sort(key=lambda item: item[0])
    degrees = [0 for _ in range(n_atoms)]
    bonds = []
    for _, i, j in candidates:
        if degrees[i] >= _max_valence_by_z(z[i]) or degrees[j] >= _max_valence_by_z(z[j]):
            continue
        bonds.append((i, j))
        degrees[i] += 1
        degrees[j] += 1
    return bonds


def _neighbors_from_bonds(n_atoms: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    neighbors = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    for nbrs in neighbors:
        nbrs.sort()
    return neighbors


def _wl_labels(z_signature: tuple[int, ...], bonds: list[tuple[int, int]], rounds: int = 4) -> list[tuple]:
    neighbors = _neighbors_from_bonds(len(z_signature), bonds)
    labels = [(z_signature[idx],) for idx in range(len(z_signature))]
    for _ in range(rounds):
        labels = [
            (labels[idx], tuple(sorted(labels[nbr] for nbr in neighbors[idx])))
            for idx in range(len(z_signature))
        ]
    return labels


def _build_order_mapping(ff: dict, z: torch.Tensor, pos: torch.Tensor) -> list[int]:
    input_signature = tuple(int(v) for v in z.detach().cpu().tolist())
    cache_key = input_signature
    if cache_key in _ATOM_ORDER_CACHE:
        return _ATOM_ORDER_CACHE[cache_key]

    ref_labels = _wl_labels(ff["z"], [(i, j) for i, j, _ in ff["bonds"]])
    inferred_bonds = _infer_bonds_from_positions(input_signature, pos.detach().cpu())
    input_labels = _wl_labels(input_signature, inferred_bonds)

    buckets: dict[tuple, list[int]] = {}
    for idx, label in enumerate(input_labels):
        buckets.setdefault(label, []).append(idx)

    ref_to_input = [-1] * len(ff["z"])
    for ref_idx, label in enumerate(ref_labels):
        candidates = buckets.get(label, [])
        if not candidates:
            raise ValueError("Unable to map MD17 aspirin atom order onto CHARMM topology order.")
        ref_to_input[ref_idx] = candidates.pop(0)

    _ATOM_ORDER_CACHE[cache_key] = ref_to_input
    return ref_to_input


def _neighbors_from_topology(ff: dict) -> list[list[int]]:
    neighbors = [[] for _ in range(len(ff["atom_types"]))]
    for i, j, _ in ff["bonds"]:
        neighbors[i].append(j)
        neighbors[j].append(i)
    for nbrs in neighbors:
        nbrs.sort()
    return neighbors


def _make_nrexcl_exclusions(ff: dict) -> set[tuple[int, int]]:
    nrexcl = int(ff.get("nrexcl", 3))
    neighbors = _neighbors_from_topology(ff)
    excluded: set[tuple[int, int]] = set()
    for start in range(len(neighbors)):
        frontier = {start}
        visited = {start}
        for _ in range(nrexcl):
            next_frontier = set()
            for node in frontier:
                for nbr in neighbors[node]:
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    excluded.add(tuple(sorted((start, nbr))))
            frontier = next_frontier
            if not frontier:
                break
    return excluded


def _bond_energy(pos_nm: torch.Tensor, ff: dict) -> torch.Tensor:
    energy = torch.zeros((), device=pos_nm.device, dtype=pos_nm.dtype)
    atom_types = ff["atom_types"]
    for i, j, _ in ff["bonds"]:
        params = ff["bondtypes"][_bond_key(atom_types[i], atom_types[j])]
        dist = torch.linalg.norm(pos_nm[i] - pos_nm[j]).clamp(min=1e-12)
        energy = energy + 0.5 * params["kb_kjmol_nm2"] * (dist - params["b0_nm"]) ** 2
    return energy


def _angle_value(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    cos_theta = torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2)).clamp(min=1e-12)
    return torch.acos(torch.clamp(cos_theta, -1.0, 1.0))


def _angle_energy(pos_nm: torch.Tensor, ff: dict) -> torch.Tensor:
    energy = torch.zeros((), device=pos_nm.device, dtype=pos_nm.dtype)
    atom_types = ff["atom_types"]
    for i, j, k, _ in ff["angles"]:
        params = ff["angletypes"][_angle_key(atom_types[i], atom_types[j], atom_types[k])]
        theta = _angle_value(pos_nm[i] - pos_nm[j], pos_nm[k] - pos_nm[j])
        theta0 = torch.deg2rad(torch.tensor(params["theta0_deg"], device=pos_nm.device, dtype=pos_nm.dtype))
        energy = energy + 0.5 * params["k_theta_kjmol_rad2"] * (theta - theta0) ** 2
        if params["kub_kjmol_nm2"] > 0 and params["ub0_nm"] > 0:
            ub = torch.linalg.norm(pos_nm[i] - pos_nm[k]).clamp(min=1e-12)
            energy = energy + 0.5 * params["kub_kjmol_nm2"] * (ub - params["ub0_nm"]) ** 2
    return energy


def _dihedral_angle(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / torch.linalg.norm(b1).clamp(min=1e-12)
    v = b0 - torch.dot(b0, b1n) * b1n
    w = b2 - torch.dot(b2, b1n) * b1n
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1n, v, dim=0), w)
    return torch.atan2(y, x)


def _match_terms(type_list: list[dict], query: tuple[str, str, str, str], func: int) -> list[dict]:
    candidates = []
    for entry in type_list:
        if entry["func"] != func:
            continue
        if _match_dihedral(entry["types"], query):
            candidates.append((entry, _dihedral_score(entry["types"], query)))
    if not candidates:
        return []
    best_score = max(score for _, score in candidates)
    return [entry for entry, score in candidates if score == best_score]


def _proper_dihedral_energy(pos_nm: torch.Tensor, ff: dict) -> torch.Tensor:
    energy = torch.zeros((), device=pos_nm.device, dtype=pos_nm.dtype)
    atom_types = ff["atom_types"]
    for i, j, k, l, func in ff["proper_dihedrals"]:
        query = (atom_types[i], atom_types[j], atom_types[k], atom_types[l])
        terms = _match_terms(ff["proper_types"], query, func)
        if not terms:
            terms = _match_terms(ff["proper_types"], tuple(reversed(query)), func)
        if not terms:
            continue
        phi = _dihedral_angle(pos_nm[i], pos_nm[j], pos_nm[k], pos_nm[l])
        for term in terms:
            phi0 = torch.deg2rad(torch.tensor(term["phi0_deg"], device=pos_nm.device, dtype=pos_nm.dtype))
            energy = energy + term["k_phi_kjmol"] * (1.0 + torch.cos(term["mult"] * phi - phi0))
    return energy


def _improper_dihedral_energy(pos_nm: torch.Tensor, ff: dict) -> torch.Tensor:
    energy = torch.zeros((), device=pos_nm.device, dtype=pos_nm.dtype)
    atom_types = ff["atom_types"]
    for i, j, k, l, func in ff["improper_dihedrals"]:
        query = (atom_types[i], atom_types[j], atom_types[k], atom_types[l])
        terms = _match_terms(ff["improper_types"], query, func)
        if not terms:
            terms = _match_terms(ff["improper_types"], tuple(reversed(query)), func)
        if not terms:
            continue
        phi = _dihedral_angle(pos_nm[i], pos_nm[j], pos_nm[k], pos_nm[l])
        for term in terms:
            phi0 = torch.deg2rad(torch.tensor(term["phi0_deg"], device=pos_nm.device, dtype=pos_nm.dtype))
            energy = energy + 0.5 * term["k_phi_kjmol"] * (phi - phi0) ** 2
    return energy


def _minimum_image(delta_nm: torch.Tensor, box_nm: torch.Tensor | None) -> torch.Tensor:
    if box_nm is None:
        return delta_nm
    return delta_nm - box_nm * torch.round(delta_nm / box_nm.clamp(min=1e-12))


def _nonbonded_energy(pos_nm: torch.Tensor, ff: dict, box_l=None) -> torch.Tensor:
    energy = torch.zeros((), device=pos_nm.device, dtype=pos_nm.dtype)
    atom_types = ff["atom_types"]
    charges = torch.as_tensor(ff["charges"], device=pos_nm.device, dtype=pos_nm.dtype)
    excluded = ff.get("excluded_pairs")
    if excluded is None:
        excluded = _make_nrexcl_exclusions(ff)
        ff["excluded_pairs"] = excluded
    pair_interactions = ff["pairs"]
    box_nm = None
    if box_l is not None:
        box_nm = torch.as_tensor(box_l, device=pos_nm.device, dtype=pos_nm.dtype) * 0.1

    n_atoms = pos_nm.shape[0]
    for i in range(n_atoms):
        params_i = ff["nonbonded"][atom_types[i]]
        for j in range(i + 1, n_atoms):
            key = tuple(sorted((i, j)))
            if key in excluded and key not in pair_interactions:
                continue
            params_j = ff["nonbonded"][atom_types[j]]
            delta = _minimum_image(pos_nm[i] - pos_nm[j], box_nm)
            dist = torch.linalg.norm(delta).clamp(min=1e-12)
            sigma = 0.5 * (params_i["sigma_nm"] + params_j["sigma_nm"])
            epsilon = (params_i["epsilon_kjmol"] * params_j["epsilon_kjmol"]) ** 0.5
            sr6 = (sigma / dist) ** 6
            energy = energy + 4.0 * epsilon * (sr6 * sr6 - sr6)
            energy = energy + 138.935456 * charges[i] * charges[j] / dist
    return energy


def _aspirin_component_energies(pos_nm: torch.Tensor, ff: dict, box_l=None) -> dict[str, torch.Tensor]:
    return {
        "bond_kjmol": _bond_energy(pos_nm, ff),
        "angle_kjmol": _angle_energy(pos_nm, ff),
        "proper_kjmol": _proper_dihedral_energy(pos_nm, ff),
        "improper_kjmol": _improper_dihedral_energy(pos_nm, ff),
        "nonbonded_kjmol": _nonbonded_energy(pos_nm, ff, box_l=box_l),
    }


def _aspirin_reference_energy_forces(
    pos: torch.Tensor,
    z: torch.Tensor,
    box_l=None,
    energy_offset_eV: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ff = _load_aspirin_forcefield()
    ref_to_input = _build_order_mapping(ff, z, pos)
    input_to_ref = [0] * len(ref_to_input)
    for ref_idx, input_idx in enumerate(ref_to_input):
        input_to_ref[input_idx] = ref_idx

    pos_ordered = pos[ref_to_input]
    pos_req = pos_ordered.detach().clone().requires_grad_(True)
    pos_nm = pos_req * 0.1
    components = _aspirin_component_energies(pos_nm, ff, box_l=box_l)
    energy_kjmol = sum(components.values())
    energy_ev = energy_kjmol * KJMOL_TO_EV - torch.as_tensor(
        float(energy_offset_eV),
        device=pos_req.device,
        dtype=pos_req.dtype,
    )
    forces_ref = -torch.autograd.grad(energy_ev, pos_req, create_graph=False, retain_graph=False)[0]
    forces_input = torch.zeros_like(pos)
    for input_idx, ref_idx in enumerate(input_to_ref):
        forces_input[input_idx] = forces_ref[ref_idx]
    return energy_ev.detach(), forces_input.detach()


def debug_aspirin_reference_components(
    pos: torch.Tensor,
    z: torch.Tensor,
    box_l=None,
    energy_offset_eV: float | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-term aspirin baseline energies on one graph in input atom order."""
    ff = _load_aspirin_forcefield()
    _ = _build_order_mapping(ff, z, pos)
    pos_ordered = pos[_ATOM_ORDER_CACHE[tuple(int(v) for v in z.detach().cpu().tolist())]]
    pos_nm = pos_ordered.detach().clone() * 0.1
    components = _aspirin_component_energies(pos_nm, ff, box_l=box_l)
    components_ev = {key.replace("_kjmol", "_ev"): value * KJMOL_TO_EV for key, value in components.items()}
    raw_total_ev = sum(components_ev.values())
    if energy_offset_eV is None:
        energy_offset_eV = load_reference_energy_offset_eV("aspirin")
    components_ev["raw_total_ev"] = raw_total_ev
    components_ev["energy_offset_ev"] = torch.as_tensor(
        float(energy_offset_eV),
        device=raw_total_ev.device,
        dtype=raw_total_ev.dtype,
    )
    components_ev["total_ev"] = raw_total_ev - components_ev["energy_offset_ev"]
    return {key: value.detach() for key, value in components_ev.items()}


def lj_energy_forces(
    pos: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    dtype = pos.dtype
    n_atoms = pos.shape[0]
    if n_atoms <= 1:
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    rij = pos[:, None, :] - pos[None, :, :]
    r2 = (rij * rij).sum(dim=-1)
    triu = torch.triu(torch.ones((n_atoms, n_atoms), device=device, dtype=torch.bool), diagonal=1)
    r = torch.sqrt(torch.clamp(r2, min=1e-12))
    mask = triu & (r < float(r_cut_A))
    if not mask.any():
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    r_sel = r[mask]
    rij_sel = rij[mask]
    sig = torch.as_tensor(float(sigma_A), device=device, dtype=dtype)
    eps = torch.as_tensor(float(epsilon_eV), device=device, dtype=dtype)
    inv_r = 1.0 / r_sel
    sr = sig * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    u_pairs = 4.0 * eps * (sr12 - sr6)
    u_total = u_pairs.sum()

    coef = -24.0 * eps * (2.0 * sr12 - sr6) * (inv_r * inv_r)
    fij = coef[:, None] * rij_sel
    idx = mask.nonzero(as_tuple=False)
    forces = torch.zeros_like(pos)
    forces.index_add_(0, idx[:, 0], fij)
    forces.index_add_(0, idx[:, 1], -fij)
    return u_total, forces


def reference_energy_forces(
    z: torch.Tensor,
    pos: torch.Tensor,
    molecule: str | None = None,
    box_l=None,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
    energy_offset_eV: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (molecule or "").lower() == "aspirin" and ASPIRIN_TOP.exists():
        if energy_offset_eV is None:
            energy_offset_eV = load_reference_energy_offset_eV(molecule)
        return _aspirin_reference_energy_forces(pos, z, box_l=box_l, energy_offset_eV=energy_offset_eV)
    return lj_energy_forces(pos, epsilon_eV=epsilon_eV, sigma_A=sigma_A, r_cut_A=r_cut_A)


def reference_energy_forces_batched(
    z: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    molecule: str | None = None,
    box_l=None,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
    energy_offset_eV: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    dtype = pos.dtype
    unique_graphs = torch.unique(batch)
    u_per_graph = torch.zeros((int(unique_graphs.numel()),), device=device, dtype=dtype)
    f_all_atoms = torch.zeros_like(pos)

    for graph_idx, graph_id in enumerate(unique_graphs.tolist()):
        mask = batch == graph_id
        local_box = None
        if box_l is not None:
            if isinstance(box_l, torch.Tensor) and box_l.dim() == 2 and box_l.shape[1] == 3:
                local_box = box_l[graph_idx]
            else:
                local_box = box_l
        u_graph, f_graph = reference_energy_forces(
            z=z[mask],
            pos=pos[mask],
            molecule=molecule,
            box_l=local_box,
            epsilon_eV=epsilon_eV,
            sigma_A=sigma_A,
            r_cut_A=r_cut_A,
            energy_offset_eV=energy_offset_eV,
        )
        u_per_graph[graph_idx] = u_graph
        f_all_atoms[mask] = f_graph

    return u_per_graph, f_all_atoms


def lj_energy_forces_batched(
    z: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return reference_energy_forces_batched(
        z=z,
        pos=pos,
        batch=batch,
        molecule=None,
        epsilon_eV=epsilon_eV,
        sigma_A=sigma_A,
        r_cut_A=r_cut_A,
    )
