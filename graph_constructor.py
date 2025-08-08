import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
try:
    from torch_cluster import radius_graph  # optional, only needed for complex graphs
except Exception:  # pragma: no cover
    radius_graph = None
import numpy as np
from updated_parser import Protein, Ligand  
from config import (
    ATOM_ELEMENT_TO_INDEX,
    ATOM_CLASS_UNK,
    NUM_ATOM_CLASSES,
)

def one_hot(indices, num_classes):
    """
    Convert integer indices to one-hot encodings.
    """
    return F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes)


def build_protein_graph(prot_dict) -> Data:
    """
    Build a PyG Data for a protein:
      - prot_x: [N, 20+1] residue one-hot (20 aa) + hydrophobicity
      - pos:   [N,3]
      - edge_index, edge_attr (distance, type=0/1, bond_order=0)
    """
    residues = prot_dict['residues']              # [N]
    pos = torch.tensor(prot_dict['pos_CA'], dtype=torch.float)
    hydro = torch.tensor(prot_dict['hydrophobicity'], dtype=torch.float).unsqueeze(1)

    # Always 20 canonical amino acids
    num_aa_types = 20
    onehot = one_hot(residues, num_aa_types).float()
    prot_x = torch.cat([onehot, hydro], dim=1)    # [N,21]

    edge_index = torch.tensor(prot_dict['edge_index'], dtype=torch.long)
    dist = torch.tensor(prot_dict['edge_distance'], dtype=torch.float)
    e_type = torch.tensor(prot_dict['edge_type'], dtype=torch.float)

    # edge_attr: distance, type, bond_order=0
    edge_attr = torch.stack([dist, e_type, torch.zeros_like(dist)], dim=1)

    # also expose features under standard name `x`
    return Data(x=prot_x, prot_x=prot_x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)


def build_ligand_graph(lig_dict) -> Data:
    """
    Build a PyG Data for a ligand:
      - lig_x: [M, E + 1 + H + 1] element one-hot + charge + hybrid one-hot + aromatic
      - pos:   [M,3]
      - edge_index, edge_attr (distance, type=2, bond_order)
    """
    elem = lig_dict['element']                    # [M]
    pos = torch.tensor(lig_dict['pos'], dtype=torch.float)
    charge = torch.tensor(lig_dict['charge'], dtype=torch.float).unsqueeze(1)
    hybrid = lig_dict['hybridization']            # [M]
    aromatic = torch.tensor(lig_dict['aromaticity'], dtype=torch.float).unsqueeze(1)

    # Map element atomic number to fixed class id with UNK fallback
    elem_indices = []
    for z in elem.tolist():
        elem_indices.append(ATOM_ELEMENT_TO_INDEX.get(int(z), ATOM_CLASS_UNK))
    elem_indices = torch.tensor(elem_indices, dtype=torch.long)
    elem_oh = F.one_hot(elem_indices, num_classes=NUM_ATOM_CLASSES).float()
    # One-hot hybridization by max state in sample (kept as auxiliary features)
    max_hyb = int(hybrid.max()) + 1 if hybrid.numel() > 0 else 1
    hyb_oh = one_hot(hybrid, max_hyb).float()

    lig_x = torch.cat([elem_oh, charge, hyb_oh, aromatic], dim=1)

    bond_index = torch.tensor(lig_dict['bond_index'], dtype=torch.long)
    bond_order = torch.tensor(lig_dict['bond_type'], dtype=torch.float)

    # compute distances
    coords = pos
    i, j = bond_index
    dist = (coords[i] - coords[j]).norm(dim=1)

    # edge_attr: distance, type=2, bond_order
    edge_attr = torch.stack([dist, torch.full_like(dist, 2.0), bond_order], dim=1)

    # also expose features under standard name `x` and keep element class indices for labels
    return Data(x=lig_x, lig_x=lig_x, pos=pos, edge_index=bond_index, edge_attr=edge_attr,
                atom_class=elem_indices)


def build_complex_graph(prot_data: Data, lig_data: Data, cross_cutoff: float = 6.0) -> Data:
    """
    Merge protein & ligand Data into one complex Data:
      - prot_x, lig_x retained separately
      - pos: concatenated positions
      - edges: protein(0/1), ligand(2), cross(3)
      - node_type, num_prot for model embedding
    """
    # counts
    n_prot = prot_data.prot_x.size(0)
    n_lig = lig_data.lig_x.size(0)

    # combine positions
    pos = torch.cat([prot_data.pos, lig_data.pos], dim=0)

    # combine edges
    p_ei = prot_data.edge_index
    l_ei = lig_data.edge_index + n_prot
    edge_index = torch.cat([p_ei, l_ei], dim=1)
    edge_attr = torch.cat([prot_data.edge_attr, lig_data.edge_attr], dim=0)

    # cross edges via radius_graph
    if radius_graph is None:
        raise ImportError("torch_cluster is required for building complex cross-edges. Please install torch-cluster.")
    rg = radius_graph(pos, r=cross_cutoff, loop=False)
    mask = ((rg[0] < n_prot) & (rg[1] >= n_prot)) | ((rg[0] >= n_prot) & (rg[1] < n_prot))
    ce = rg[:, mask]
    d = (pos[ce[0]] - pos[ce[1]]).norm(dim=1)
    cross_attr = torch.stack([d, torch.full_like(d, 3.0), torch.zeros_like(d)], dim=1)

    edge_index = torch.cat([edge_index, ce], dim=1)
    edge_attr = torch.cat([edge_attr, cross_attr], dim=0)

    # node type & metadata
    node_type = torch.cat([torch.zeros(n_prot, dtype=torch.long), torch.ones(n_lig, dtype=torch.long)])

    data = Data(
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        num_prot=n_prot,
        prot_x=prot_data.prot_x,
        lig_x=lig_data.lig_x
    )
    return data


if __name__ == '__main__':
    protein_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_pocket.pdb'
    ligand_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_ligand.mol2'
    prot = Protein(protein_path)
    lig = Ligand(ligand_path)
    if prot.is_valid and lig.is_valid:
        prot_data = build_protein_graph(prot.to_dict())
        lig_data = build_ligand_graph(lig.to_dict())
        print(f'Protein graph: nodes={prot_data.x.size(0)}, edges={prot_data.edge_index.size(1)}')
        print(f'Ligand  graph: nodes={lig_data.x.size(0)}, edges={lig_data.edge_index.size(1)}')
        if radius_graph is not None:
            try:
                complex_graph = build_complex_graph(prot_data, lig_data)
                print('Built complex graph:', complex_graph)
            except Exception as e:
                print('Skipping complex graph (optional):', e)
    else:
        print('Failed to parse example structures.')
