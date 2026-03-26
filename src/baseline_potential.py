#!/usr/bin/env python
"""
Analytic baseline potentials used by delta-learning.

Two modes exist:
- Aspirin reference mode: CHARMM-like bonded + typed LJ + fixed-charge Coulomb.
- Fallback mode: simple Lennard-Jones 12-6 with cutoff.

The aspirin reference is derived once from the first observed aspirin graph and
cached under `data/aspirin_topology_cache.json` so it plugs into the existing
training/evaluation loops without a separate preprocessing step.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASPIRIN_REFERENCE_PATH = PROJECT_ROOT / "data" / "aspirin_charmm_reference.json"
ASPIRIN_TOPOLOGY_CACHE_PATH = PROJECT_ROOT / "data" / "aspirin_topology_cache.json"
KCAL_MOL_TO_EV = 0.0433641153
COULOMB_CONSTANT_EV_A = 14.3996454784255

_TOPOLOGY_CACHE = {}


def _to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


@lru_cache(maxsize=4)
def _load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


@lru_cache(maxsize=1)
def _aspirin_reference_library() -> dict:
    return _load_json(str(ASPIRIN_REFERENCE_PATH))


def _fallback_if_needed(molecule: str | None) -> bool:
    return (molecule or "").lower() != "aspirin" or not ASPIRIN_REFERENCE_PATH.exists()


def _pairwise_distances(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rij = pos[:, None, :] - pos[None, :, :]
    distances = torch.sqrt(torch.clamp((rij * rij).sum(dim=-1), min=1e-12))
    return rij, distances


def _infer_bonds(z: torch.Tensor, pos: torch.Tensor, library: dict) -> list[tuple[int, int]]:
    radii = {int(k): float(v) for k, v in library["covalent_radii_A"].items()}
    max_valence = {int(k): int(v) for k, v in library["max_valence"].items()}
    scale = float(library.get("bond_detection_scale", 1.25))

    _, distances = _pairwise_distances(pos)
    n_atoms = int(z.numel())
    candidates = []
    for i in range(n_atoms):
        zi = int(z[i].item())
        for j in range(i + 1, n_atoms):
            zj = int(z[j].item())
            threshold = scale * (radii.get(zi, 0.7) + radii.get(zj, 0.7))
            d_ij = float(distances[i, j].item())
            if d_ij <= threshold:
                candidates.append((d_ij, i, j))

    candidates.sort(key=lambda item: item[0])
    degrees = [0 for _ in range(n_atoms)]
    bonds = []
    for _, i, j in candidates:
        if degrees[i] >= max_valence.get(int(z[i].item()), 4):
            continue
        if degrees[j] >= max_valence.get(int(z[j].item()), 4):
            continue
        bonds.append((i, j))
        degrees[i] += 1
        degrees[j] += 1
    return bonds


def _build_neighbors(n_atoms: int, bonds: list[tuple[int, int]]) -> dict[int, list[int]]:
    neighbors = {idx: [] for idx in range(n_atoms)}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors


def _find_carbonyl_carbons(z: torch.Tensor, neighbors: dict[int, list[int]]) -> set[int]:
    carbonyl = set()
    for idx in range(int(z.numel())):
        if int(z[idx].item()) != 6:
            continue
        oxygen_neighbors = [nbr for nbr in neighbors[idx] if int(z[nbr].item()) == 8]
        if len(oxygen_neighbors) >= 2:
            carbonyl.add(idx)
    return carbonyl


def _assign_atom_types(z: torch.Tensor, neighbors: dict[int, list[int]]) -> tuple[list[str], list[float]]:
    carbonyl_carbons = _find_carbonyl_carbons(z, neighbors)
    atom_types = ["" for _ in range(int(z.numel()))]
    charges = [0.0 for _ in range(int(z.numel()))]

    for idx in range(int(z.numel())):
        zi = int(z[idx].item())
        nbrs = neighbors[idx]
        nbr_z = [int(z[nbr].item()) for nbr in nbrs]

        if zi == 1:
            heavy = next((nbr for nbr in nbrs if int(z[nbr].item()) != 1), None)
            if heavy is None:
                atom_types[idx] = "H"
                charges[idx] = 0.0
            elif int(z[heavy].item()) == 8:
                atom_types[idx] = "H_OH"
                charges[idx] = 0.42
            elif int(z[heavy].item()) == 6 and len(neighbors[heavy]) == 4:
                atom_types[idx] = "H_CT"
                charges[idx] = 0.09
            else:
                atom_types[idx] = "H_CA"
                charges[idx] = 0.115
            continue

        if zi == 8:
            if any(int(z[nbr].item()) == 1 for nbr in nbrs):
                atom_types[idx] = "O_OH"
                charges[idx] = -0.65
            elif any(nbr in carbonyl_carbons for nbr in nbrs) and len(nbrs) == 1:
                atom_types[idx] = "O_CO"
                charges[idx] = -0.55
            else:
                atom_types[idx] = "O_ES"
                charges[idx] = -0.32
            continue

        if zi == 6:
            if idx in carbonyl_carbons:
                if any(
                    int(z[nbr].item()) == 8 and any(int(z[n2].item()) == 1 for n2 in neighbors[nbr])
                    for nbr in nbrs
                ):
                    atom_types[idx] = "C_CARBOXY"
                    charges[idx] = 0.76
                else:
                    atom_types[idx] = "C_ESTER"
                    charges[idx] = 0.74
            elif nbr_z.count(1) == 3:
                atom_types[idx] = "C_CT3"
                charges[idx] = -0.27
            else:
                atom_types[idx] = "C_AR"
                if any(int(z[nbr].item()) == 8 for nbr in nbrs):
                    charges[idx] = 0.18
                elif any(nbr in carbonyl_carbons for nbr in nbrs):
                    charges[idx] = 0.16
                else:
                    charges[idx] = -0.12
            continue

        atom_types[idx] = "X"
        charges[idx] = 0.0

    return atom_types, charges


def _enumerate_angles(neighbors: dict[int, list[int]]) -> list[tuple[int, int, int]]:
    angles = set()
    for center, nbrs in neighbors.items():
        nbrs_sorted = sorted(nbrs)
        for i_idx in range(len(nbrs_sorted)):
            for k_idx in range(i_idx + 1, len(nbrs_sorted)):
                angles.add((nbrs_sorted[i_idx], center, nbrs_sorted[k_idx]))
    return sorted(angles)


def _enumerate_propers(neighbors: dict[int, list[int]]) -> list[tuple[int, int, int, int]]:
    torsions = set()
    for j, k_nbrs in neighbors.items():
        for k in k_nbrs:
            if j > k:
                continue
            left = [i for i in neighbors[j] if i != k]
            right = [l for l in neighbors[k] if l != j]
            for i in left:
                for l in right:
                    torsion = (i, j, k, l)
                    reverse = (l, k, j, i)
                    torsions.add(min(torsion, reverse))
    return sorted(torsions)


def _enumerate_impropers(z: torch.Tensor, neighbors: dict[int, list[int]]) -> list[tuple[int, int, int, int]]:
    impropers = []
    for center, nbrs in neighbors.items():
        if len(nbrs) != 3:
            continue
        z_center = int(z[center].item())
        if z_center == 6:
            impropers.append((center, nbrs[0], nbrs[1], nbrs[2]))
    return sorted(impropers)


def _make_exclusions(neighbors: dict[int, list[int]]) -> set[tuple[int, int]]:
    excluded = set()
    for i, nbrs in neighbors.items():
        for j in nbrs:
            excluded.add(tuple(sorted((i, j))))
        for j in nbrs:
            for k in neighbors[j]:
                if k == i:
                    continue
                excluded.add(tuple(sorted((i, k))))
        for j in nbrs:
            for k in neighbors[j]:
                if k == i:
                    continue
                for l in neighbors[k]:
                    if l in (j, i):
                        continue
                    excluded.add(tuple(sorted((i, l))))
    return excluded


def _parameter_table_by_key(table: list[dict], value_key: str) -> dict[tuple, dict]:
    result = {}
    for entry in table:
        key = tuple(entry["types"])
        result[key] = entry
        result[tuple(reversed(key))] = entry
    result["_value_key"] = value_key
    return result


def _lookup_pair_param(table: dict, t1: str, t2: str, fallback: dict) -> dict:
    return table.get((t1, t2), fallback)


def _lookup_triple_param(table: dict, t1: str, t2: str, t3: str, fallback: dict) -> dict:
    return table.get((t1, t2, t3), fallback)


def _lookup_quad_param(table: dict, t1: str, t2: str, t3: str, t4: str, fallback: dict) -> dict:
    return table.get((t1, t2, t3, t4), fallback)


def _derive_aspirin_topology(z: torch.Tensor, pos: torch.Tensor) -> dict:
    library = _aspirin_reference_library()
    bonds = _infer_bonds(z, pos, library)
    neighbors = _build_neighbors(int(z.numel()), bonds)
    atom_types, charges = _assign_atom_types(z, neighbors)
    topology = {
        "atom_types": atom_types,
        "charges": charges,
        "bonds": bonds,
        "angles": _enumerate_angles(neighbors),
        "propers": _enumerate_propers(neighbors),
        "impropers": _enumerate_impropers(z, neighbors),
        "exclusions": sorted(list(_make_exclusions(neighbors))),
    }
    payload = {
        "z": z.detach().cpu().tolist(),
        **topology,
    }
    _write_json(ASPIRIN_TOPOLOGY_CACHE_PATH, payload)
    return topology


def _load_or_create_aspirin_topology(z: torch.Tensor, pos: torch.Tensor) -> dict:
    key = ("aspirin", tuple(int(v) for v in z.detach().cpu().tolist()))
    if key in _TOPOLOGY_CACHE:
        return _TOPOLOGY_CACHE[key]

    if ASPIRIN_TOPOLOGY_CACHE_PATH.exists():
        cached = _load_json(str(ASPIRIN_TOPOLOGY_CACHE_PATH))
        if tuple(int(v) for v in cached.get("z", [])) == key[1]:
            topology = {
                "atom_types": cached["atom_types"],
                "charges": cached["charges"],
                "bonds": [tuple(item) for item in cached["bonds"]],
                "angles": [tuple(item) for item in cached["angles"]],
                "propers": [tuple(item) for item in cached["propers"]],
                "impropers": [tuple(item) for item in cached["impropers"]],
                "exclusions": {tuple(item) for item in cached["exclusions"]},
            }
            _TOPOLOGY_CACHE[key] = topology
            return topology

    topology = _derive_aspirin_topology(z, pos)
    topology["exclusions"] = {tuple(item) for item in topology["exclusions"]}
    _TOPOLOGY_CACHE[key] = topology
    return topology


def _get_box_lengths(box_l, device, dtype):
    if box_l is None:
        return None
    if not isinstance(box_l, torch.Tensor):
        box_l = torch.as_tensor(box_l, device=device, dtype=dtype)
    return box_l.to(device=device, dtype=dtype)


def _minimum_image(delta: torch.Tensor, box_l: torch.Tensor | None) -> torch.Tensor:
    if box_l is None:
        return delta
    box = box_l.view(1, 1, 3)
    return delta - box * torch.round(delta / box.clamp(min=1e-12))


def _bond_energy(pos: torch.Tensor, topology: dict, library: dict) -> torch.Tensor:
    if not topology["bonds"]:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)

    param_table = _parameter_table_by_key(library["bond_params"], "k_bond_kcal_mol_A2")
    fallback = library["fallbacks"]["bond"]
    energy = torch.zeros((), device=pos.device, dtype=pos.dtype)
    atom_types = topology["atom_types"]
    for i, j in topology["bonds"]:
        delta = pos[i] - pos[j]
        dist = torch.sqrt(torch.clamp((delta * delta).sum(), min=1e-12))
        params = _lookup_pair_param(param_table, atom_types[i], atom_types[j], fallback)
        k_bond = float(params["k_bond_kcal_mol_A2"]) * KCAL_MOL_TO_EV
        r0 = float(params["r0_A"])
        energy = energy + 0.5 * k_bond * (dist - r0) ** 2
    return energy


def _angle_value(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    cos_theta = torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2)).clamp(min=1e-12)
    return torch.acos(torch.clamp(cos_theta, -1.0, 1.0))


def _angle_energy(pos: torch.Tensor, topology: dict, library: dict) -> torch.Tensor:
    if not topology["angles"]:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)

    param_table = _parameter_table_by_key(library["angle_params"], "k_angle_kcal_mol_rad2")
    fallback = library["fallbacks"]["angle"]
    energy = torch.zeros((), device=pos.device, dtype=pos.dtype)
    atom_types = topology["atom_types"]
    for i, j, k in topology["angles"]:
        theta = _angle_value(pos[i] - pos[j], pos[k] - pos[j])
        params = _lookup_triple_param(param_table, atom_types[i], atom_types[j], atom_types[k], fallback)
        k_theta = float(params["k_angle_kcal_mol_rad2"]) * KCAL_MOL_TO_EV
        theta0 = torch.deg2rad(torch.tensor(float(params["theta0_deg"]), device=pos.device, dtype=pos.dtype))
        energy = energy + 0.5 * k_theta * (theta - theta0) ** 2
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


def _torsion_energy(pos: torch.Tensor, terms: list[tuple[int, int, int, int]], topology: dict, table_key: str, fallback_key: str, library: dict) -> torch.Tensor:
    if not terms:
        return torch.zeros((), device=pos.device, dtype=pos.dtype)

    param_table = _parameter_table_by_key(library[table_key], "k_phi_kcal_mol")
    fallback = library["fallbacks"][fallback_key]
    atom_types = topology["atom_types"]
    energy = torch.zeros((), device=pos.device, dtype=pos.dtype)
    for i, j, k, l in terms:
        phi = _dihedral_angle(pos[i], pos[j], pos[k], pos[l])
        params = _lookup_quad_param(param_table, atom_types[i], atom_types[j], atom_types[k], atom_types[l], fallback)
        k_phi = float(params["k_phi_kcal_mol"]) * KCAL_MOL_TO_EV
        periodicity = int(params["periodicity"])
        phase = torch.deg2rad(torch.tensor(float(params["phase_deg"]), device=pos.device, dtype=pos.dtype))
        energy = energy + k_phi * (1.0 + torch.cos(periodicity * phi - phase))
    return energy


def _nonbonded_energy(pos: torch.Tensor, topology: dict, library: dict, box_l=None) -> torch.Tensor:
    atom_types = topology["atom_types"]
    charges = torch.as_tensor(topology["charges"], device=pos.device, dtype=pos.dtype)
    nb_params = library["nonbonded_params"]
    exclusions = topology["exclusions"]

    energy = torch.zeros((), device=pos.device, dtype=pos.dtype)
    n_atoms = pos.shape[0]
    box = _get_box_lengths(box_l, pos.device, pos.dtype)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if tuple(sorted((i, j))) in exclusions:
                continue
            delta = pos[i] - pos[j]
            if box is not None:
                delta = delta - box * torch.round(delta / box.clamp(min=1e-12))
            dist = torch.sqrt(torch.clamp((delta * delta).sum(), min=1e-12))
            params_i = nb_params[atom_types[i]]
            params_j = nb_params[atom_types[j]]
            sigma = 0.5 * (float(params_i["sigma_A"]) + float(params_j["sigma_A"]))
            epsilon = (float(params_i["epsilon_eV"]) * float(params_j["epsilon_eV"])) ** 0.5
            if epsilon > 0:
                sr6 = (sigma / dist) ** 6
                energy = energy + 4.0 * epsilon * (sr6 * sr6 - sr6)
            qi = charges[i]
            qj = charges[j]
            energy = energy + COULOMB_CONSTANT_EV_A * qi * qj / dist
    return energy


def _aspirin_reference_energy_and_forces(pos: torch.Tensor, z: torch.Tensor, box_l=None) -> tuple[torch.Tensor, torch.Tensor]:
    library = _aspirin_reference_library()
    topology = _load_or_create_aspirin_topology(z, pos)
    pos_req = pos.detach().clone().requires_grad_(True)

    total_energy = (
        _bond_energy(pos_req, topology, library)
        + _angle_energy(pos_req, topology, library)
        + _torsion_energy(pos_req, topology["propers"], topology, "proper_torsion_params", "proper_torsion", library)
        + _torsion_energy(pos_req, topology["impropers"], topology, "improper_torsion_params", "improper_torsion", library)
        + _nonbonded_energy(pos_req, topology, library, box_l=box_l)
    )
    forces = -torch.autograd.grad(total_energy, pos_req, create_graph=False, retain_graph=False)[0]
    return total_energy.detach(), forces.detach()


def lj_energy_forces(
    pos: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise LJ energy and forces for one molecule (no PBC)."""
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
    i_idx = idx[:, 0]
    j_idx = idx[:, 1]

    forces = torch.zeros_like(pos)
    forces.index_add_(0, i_idx, fij)
    forces.index_add_(0, j_idx, -fij)
    return u_total, forces


def reference_energy_forces(
    z: torch.Tensor,
    pos: torch.Tensor,
    molecule: str | None = None,
    box_l=None,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return analytic baseline energy/forces for one molecule."""
    if _fallback_if_needed(molecule):
        return lj_energy_forces(pos, epsilon_eV=epsilon_eV, sigma_A=sigma_A, r_cut_A=r_cut_A)
    return _aspirin_reference_energy_and_forces(pos, z, box_l=box_l)


def reference_energy_forces_batched(
    z: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    molecule: str | None = None,
    box_l=None,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the analytic reference to each graph in a PyG batch."""
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
    """Backward-compatible alias used by the existing training/eval code."""
    return reference_energy_forces_batched(
        z=z,
        pos=pos,
        batch=batch,
        molecule=None,
        epsilon_eV=epsilon_eV,
        sigma_A=sigma_A,
        r_cut_A=r_cut_A,
    )
