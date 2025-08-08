import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from pdbbind_parser import Protein, Ligand  # Assumes your earlier parsing code is in pdbbind_parser

def create_protein_graph(data_dict):
    # Convert lists/arrays into PyTorch tensors.
    residue_types = torch.tensor(data_dict['residues'], dtype=torch.long).unsqueeze(-1)
    hydrophobicity_scores = torch.tensor(data_dict['hydrophobicity_scores'], dtype=torch.float).unsqueeze(-1)
    positions = torch.tensor(data_dict['pos_CA'], dtype=torch.float)
    
    # Process edge attributes: compute the inverse of the distance.
    distances = torch.tensor(data_dict['distances'], dtype=torch.float)
    inverse_distances = 1.0 / distances.unsqueeze(-1)
    
    # Concatenate node features: residue type, 3D position, and hydrophobicity score.
    node_features = torch.cat([residue_types, positions, hydrophobicity_scores], dim=1)
    
    # Convert the connections into a tensor for edge_index.
    edge_index = torch.tensor(data_dict['connections'], dtype=torch.long)
    
    # Ensure the graph is undirected (duplicates will be removed if already bidirectional).
    edge_index, inverse_distances = to_undirected(edge_index, edge_attr=inverse_distances)
    
    # Create and return the PyG Data object.
    graph_data = Data(x=node_features, edge_index=edge_index.contiguous(), edge_attr=inverse_distances)
    return graph_data


def create_ligand_graph(data_dict):
    # Convert atom-level data into tensors.
    elements = torch.tensor(data_dict['element'], dtype=torch.float).unsqueeze(-1)
    pos = torch.tensor(data_dict['pos'], dtype=torch.float)
    hybridization = torch.tensor(data_dict['hybridization'], dtype=torch.float).unsqueeze(-1)
    atom_features = torch.tensor(data_dict['atom_features'], dtype=torch.float)
    
    # Bond orders are used as edge (bond) weights.
    bond_weights = torch.tensor(data_dict['bond_type'], dtype=torch.float).unsqueeze(-1)
    
    # Concatenate atom features: atomic number, position, hybridization, and other atom features.
    node_features = torch.cat([elements, pos, hybridization, atom_features], dim=1)
    
    # Convert bond indices into a tensor.
    edge_index = torch.tensor(data_dict['bond_index'], dtype=torch.long)
    
    # Ensure the bond graph is undirected.
    edge_index, bond_weights = to_undirected(edge_index, edge_attr=bond_weights)
    
    # Create and return the PyG Data object.
    graph_data = Data(x=node_features, edge_index=edge_index.contiguous(), edge_attr=bond_weights)
    return graph_data




'if __name__ == "__main__":'
protein_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_pocket.pdb'
ligand_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_ligand.mol2'
protein = Protein(protein_path).to_dict()
ligand = Ligand(ligand_path).to_dict()

print(create_protein_graph(protein))
print(create_ligand_graph(ligand))
