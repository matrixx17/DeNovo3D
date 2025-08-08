import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import numpy as np
from updated_parser import Protein, Ligand  

def one_hot(indices, num_classes):
    """
    Utility to convert integer indices to one-hot encodings.
    """
    return F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes)


def build_protein_graph(prot_dict, num_aa_types: int = 20) -> Data:
    """
    Constructs a PyG Data object for a protein using parsed dict:
      - Node features: residue type one-hot + hydrophobicity
      - Edge attributes: [distance, edge_type, bond_order=0]
    """
    residues = prot_dict['residues']
    pos = torch.tensor(prot_dict['pos_CA'], dtype=torch.float)
    hydro = prot_dict['hydrophobicity']

    # One-hot encode residue identities
    res_onehot = one_hot(residues, num_aa_types).float()
    hydro_tensor = torch.tensor(hydro, dtype=torch.float).unsqueeze(1)
    x = torch.cat([res_onehot, hydro_tensor], dim=1)

    edge_index = torch.tensor(prot_dict['edge_index'], dtype=torch.long)
    dist = prot_dict['edge_distance']
    e_type = prot_dict['edge_type']

    # Pack edge attributes: distance, edge_type, bond_order (0 for protein)
    edge_attr = torch.stack([
        torch.tensor(dist, dtype=torch.float),
        torch.tensor(e_type, dtype=torch.float),
        torch.zeros(len(dist), dtype=torch.float)
    ], dim=1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def build_ligand_graph(lig_dict, max_atomic_num: int = 100) -> Data:
    """
    Constructs a PyG Data object for a ligand using parsed dict:
      - Node features: element one-hot + charge + hybridization one-hot + aromatic flag
      - Edge attributes: [distance, edge_type=2, bond_order]
    """
    elem = lig_dict['element']
    pos = torch.tensor(lig_dict['pos'], dtype=torch.float)
    charge = lig_dict['charge']
    hybrid = lig_dict['hybridization']
    aromatic = lig_dict['aromaticity']
    bond_index = lig_dict['bond_index']
    bond_order = lig_dict['bond_type']

    # Element one-hot
    elem_onehot = one_hot(elem, max_atomic_num).float()
    charge_tensor = torch.tensor(charge, dtype=torch.float).unsqueeze(1)
    hybrid_onehot = one_hot(hybrid, int(max(hybrid))+1).float()
    aromatic_tensor = torch.tensor(aromatic, dtype=torch.float).unsqueeze(1)
    x = torch.cat([elem_onehot, charge_tensor, hybrid_onehot, aromatic_tensor], dim=1)

    # Compute bond distances
    coords = pos
    i, j = bond_index
    dist = (coords[i] - coords[j]).norm(dim=1)

    edge_index = torch.tensor(bond_index, dtype=torch.long)
    edge_attr = torch.stack([
        dist,
        torch.full((dist.size(0),), 2.0, dtype=torch.float),
        torch.tensor(bond_order, dtype=torch.float)
    ], dim=1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def build_complex_graph(prot_data: Data, lig_data: Data, cross_cutoff: float = 6.0) -> Data:
    """
    Merges protein and ligand graphs into one Data object with cross-edges:
      - Node feature concatenation and node_type indicator
      - Existing edges + ligand-offset edges + cross edges within cutoff
      - Edge attributes: distance, edge_type {0,1,2,3}, bond_order
    """
    # Node counts
    n_prot = prot_data.x.size(0)
    n_lig = lig_data.x.size(0)

    # Combine node features and positions
    x = torch.cat([prot_data.x, lig_data.x], dim=0)
    pos = torch.cat([prot_data.pos, lig_data.pos], dim=0)

    # Combine existing edges
    prot_ei = prot_data.edge_index
    lig_ei = lig_data.edge_index + n_prot
    edge_index = torch.cat([prot_ei, lig_ei], dim=1)
    edge_attr = torch.cat([prot_data.edge_attr, lig_data.edge_attr], dim=0)

    # Build cross-edges with radius_graph and filter
    # radius_graph returns edges for all nodes within cutoff
    rg = radius_graph(pos, r=cross_cutoff, loop=False)
    # filter edges connecting protein<->ligand
    mask = ((rg[0] < n_prot) & (rg[1] >= n_prot)) | ((rg[0] >= n_prot) & (rg[1] < n_prot))
    cross_ei = rg[:, mask]
    # Compute distances for cross edges
    cross_dist = (pos[cross_ei[0]] - pos[cross_ei[1]]).norm(dim=1)
    cross_attr = torch.stack([
        cross_dist,
        torch.full((cross_dist.size(0),), 3.0, dtype=torch.float),
        torch.zeros(cross_dist.size(0), dtype=torch.float)
    ], dim=1)

    # Append cross edges
    edge_index = torch.cat([edge_index, cross_ei], dim=1)
    edge_attr = torch.cat([edge_attr, cross_attr], dim=0)

    # Node-type flag: 0 for protein, 1 for ligand
    node_type = torch.cat([torch.zeros(n_prot, dtype=torch.long), torch.ones(n_lig, dtype=torch.long)])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, node_type=node_type)


protein_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_pocket.pdb'
ligand_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_ligand.mol2'
prot_data = build_protein_graph(Protein(protein_path).to_dict())
lig_data = build_ligand_graph(Ligand(ligand_path).to_dict())
complex_graph = build_complex_graph(prot_data, lig_data)
print(complex_graph)
