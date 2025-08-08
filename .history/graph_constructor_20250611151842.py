import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from pdbbind_parser import Protein, Ligand  # Assumes your earlier parsing code is in pdbbind_parser

# def create_protein_graph(data_dict):
#     # Convert lists/arrays into PyTorch tensors.
#     residue_types = torch.tensor(data_dict['residues'], dtype=torch.long).unsqueeze(-1)
#     hydrophobicity_scores = torch.tensor(data_dict['hydrophobicity_scores'], dtype=torch.float).unsqueeze(-1)
#     positions = torch.tensor(data_dict['pos_CA'], dtype=torch.float)
    
#     # Process edge attributes: compute the inverse of the distance.
#     distances = torch.tensor(data_dict['distances'], dtype=torch.float)
#     inverse_distances = 1.0 / distances.unsqueeze(-1)
    
#     # Concatenate node features: residue type, 3D position, and hydrophobicity score.
#     node_features = torch.cat([residue_types, positions, hydrophobicity_scores], dim=1)
    
#     # Convert the connections into a tensor for edge_index.
#     edge_index = torch.tensor(data_dict['connections'], dtype=torch.long)
    
#     # Ensure the graph is undirected (duplicates will be removed if already bidirectional).
#     edge_index, inverse_distances = to_undirected(edge_index, edge_attr=inverse_distances)
    
#     # Create and return the PyG Data object.
#     graph_data = Data(x=node_features, edge_index=edge_index.contiguous(), edge_attr=inverse_distances)
#     return graph_data


# def create_ligand_graph(data_dict):
#     # Convert atom-level data into tensors.
#     elements = torch.tensor(data_dict['element'], dtype=torch.float).unsqueeze(-1)
#     pos = torch.tensor(data_dict['pos'], dtype=torch.float)
#     hybridization = torch.tensor(data_dict['hybridization'], dtype=torch.float).unsqueeze(-1)
#     atom_features = torch.tensor(data_dict['atom_features'], dtype=torch.float)
    
#     # Bond orders are used as edge (bond) weights.
#     bond_weights = torch.tensor(data_dict['bond_type'], dtype=torch.float).unsqueeze(-1)
    
#     # Concatenate atom features: atomic number, position, hybridization, and other atom features.
#     node_features = torch.cat([elements, pos, hybridization, atom_features], dim=1)
    
#     # Convert bond indices into a tensor.
#     edge_index = torch.tensor(data_dict['bond_index'], dtype=torch.long)
    
#     # Ensure the bond graph is undirected.
#     edge_index, bond_weights = to_undirected(edge_index, edge_attr=bond_weights)
    
#     # Create and return the PyG Data object.
#     graph_data = Data(x=node_features, edge_index=edge_index.contiguous(), edge_attr=bond_weights)
#     return graph_data

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
    residues = prot_dict['residues']               # [N]
    pos = torch.tensor(prot_dict['pos_CA'], dtype=torch.float)  # [N,3]
    hydro = prot_dict['hydrophobicity']             # [N]

    # One-hot encode residue identities
    res_onehot = one_hot(residues, num_aa_types).float()  # [N, num_aa_types]
    hydro_tensor = torch.tensor(hydro, dtype=torch.float).unsqueeze(1)
    x = torch.cat([res_onehot, hydro_tensor], dim=1)      # [N, num_aa_types+1]

    edge_index = torch.tensor(prot_dict['edge_index'], dtype=torch.long)  # [2, E]
    dist = prot_dict['edge_distance']                                    # [E]
    e_type = prot_dict['edge_type']                                      # [E]

    # Pack edge attributes: distance, edge_type, bond_order (0 for protein)
    edge_attr = torch.stack([
        torch.tensor(dist, dtype=torch.float),
        torch.tensor(e_type, dtype=torch.float),
        torch.zeros(len(dist), dtype=torch.float)
    ], dim=1)  # [E,3]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def build_ligand_graph(lig_dict, max_atomic_num: int = 100) -> Data:
    """
    Constructs a PyG Data object for a ligand using parsed dict:
      - Node features: element one-hot + charge + hybridization one-hot + aromatic flag
      - Edge attributes: [distance, edge_type=2, bond_order]
    """
    elem = lig_dict['element']               # [M]
    pos = torch.tensor(lig_dict['pos'], dtype=torch.float)     # [M,3]
    charge = lig_dict['charge']              # [M]
    hybrid = lig_dict['hybridization']       # [M]
    aromatic = lig_dict['aromaticity']       # [M]
    bond_index = lig_dict['bond_index']      # [2, B]
    bond_order = lig_dict['bond_type']       # [B]

    # Element one-hot
    elem_onehot = one_hot(elem, max_atomic_num).float()  # [M, max_atomic_num]
    charge_tensor = torch.tensor(charge, dtype=torch.float).unsqueeze(1)
    hybrid_onehot = one_hot(hybrid, int(max(hybrid))+1).float()  # [M, H]
    aromatic_tensor = torch.tensor(aromatic, dtype=torch.float).unsqueeze(1)
    x = torch.cat([elem_onehot, charge_tensor, hybrid_onehot, aromatic_tensor], dim=1)

    # Compute bond distances
    coords = pos.numpy()
    i, j = bond_index
    dist = np.linalg.norm(coords[i] - coords[j], axis=1)

    edge_index = torch.tensor(bond_index, dtype=torch.long)
    # Pack edge attributes: distance, edge_type=2 for ligand-ligand, bond_order
    edge_attr = torch.stack([
        torch.tensor(dist, dtype=torch.float),
        torch.full((len(dist),), 2.0, dtype=torch.float),
        torch.tensor(bond_order, dtype=torch.float)
    ], dim=1)  # [B,3]

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
    prot_ei, lig_ei = prot_data.edge_index, lig_data.edge_index + n_prot
    edge_index = torch.cat([prot_ei, lig_ei], dim=1)
    edge_attr = torch.cat([prot_data.edge_attr, lig_data.edge_attr], dim=0)

    # Build cross-edges between prot CA and ligand atoms
    prot_coords = prot_data.pos.numpy()
    lig_coords = lig_data.pos.numpy()
    tree = KDTree(prot_coords)
    neighbors = tree.query_ball_point(lig_coords, r=cross_cutoff)

    cross_i, cross_j = [], []
    for j, prot_list in enumerate(neighbors):
        for i in prot_list:
            cross_i.append(i)
            cross_j.append(n_prot + j)
    if cross_i:
        # add both directions
        all_i = cross_i + cross_j
        all_j = cross_j + cross_i
        ce_index = torch.tensor([all_i, all_j], dtype=torch.long)
        # distances and attributes
        prot_pts = prot_coords[cross_i]
        lig_pts = lig_coords[[j - n_prot for j in cross_j[:len(cross_i)]]]
        d = np.linalg.norm(prot_pts - lig_pts, axis=1)
        # duplicate distances for both directions
        d_all = np.concatenate([d, d])
        ea = torch.stack([
            torch.tensor(d_all, dtype=torch.float),
            torch.full((len(d_all),), 3.0, dtype=torch.float),  # edge_type=3
            torch.zeros(len(d_all), dtype=torch.float)
        ], dim=1)
        edge_index = torch.cat([edge_index, ce_index], dim=1)
        edge_attr = torch.cat([edge_attr, ea], dim=0)

    # Node-type: 0 for protein, 1 for ligand
    node_type = torch.cat([torch.zeros(n_prot), torch.ones(n_lig)]).long()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, node_type=node_type)




'if __name__ == "__main__":'
protein_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_pocket.pdb'
ligand_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_ligand.mol2'
protein = Protein(protein_path).to_dict()
ligand = Ligand(ligand_path).to_dict()

print(create_protein_graph(protein))
print(create_ligand_graph(ligand))
