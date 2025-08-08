import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_cluster import radius_graph
import numpy as np
from updated_parser import Protein, Ligand  

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

    return Data(prot_x=prot_x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)


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

    # One-hot element types by max atomic number present
    max_elem = int(elem.max()) + 1
    elem_oh = one_hot(elem, max_elem).float()
    # One-hot hybridization by max hybrid state
    max_hyb = int(hybrid.max()) + 1
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

    return Data(lig_x=lig_x, pos=pos, edge_index=bond_index, edge_attr=edge_attr)


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


protein_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_pocket.pdb'
ligand_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_ligand.mol2'
prot_data = build_protein_graph(Protein(protein_path).to_dict())
lig_data = build_ligand_graph(Ligand(ligand_path).to_dict())
complex_graph = build_complex_graph(prot_data, lig_data)
print(complex_graph)
