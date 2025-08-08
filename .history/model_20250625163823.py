# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, global_mean_pool
# from torch_geometric.data import Data


# class Encoder(nn.Module):
#     # Encodes protein graph into single vector
#     def __init__(self, num_node_features, out_dim):
#         super(Encoder, self).__init__()

#         # First GATConv layer
#         self.conv1 = GATConv(num_node_features, 128)
#         # Apply batch normalization and dropout for regularization
#         self.bn1 = nn.BatchNorm1d(128)
#         self.dropout1 = nn.Dropout(0.2)

#         # Second GATConv layer
#         self.conv2 = GATConv(128, 128)
#         # Apply batch normalization and dropout for regularization
#         self.bn2 = nn.BatchNorm1d(128)
#         self.dropout2 = nn.Dropout(0.2)

#         # Third GATConv layer for output dimension
#         self.conv3 = GATConv(128, out_dim)
#         # Global mean pooling to aggregate features across all nodes
#         self.pool = global_mean_pool

#     def forward(self, x, edge_index, batch):
#         # Forward pass through first GATConv layer followed by batch norm and dropout
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.bn1(x)
#         x = self.dropout1(x)

#         # Forward pass through second GATConv layer followed by batch norm and dropout
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.bn2(x)
#         x = self.dropout2(x)

#         # Forward pass through third GATConv layer
#         x = self.conv3(x, edge_index)

#         # Pool to get graph-level representation
#         x = self.pool(x, batch)

#         return x


# class Generator(nn.Module):
#     # Initializes the Generator part of the model
#     def __init__(self, encoded_protein_dim, node_feature_dim, max_nodes):
#         super(Generator, self).__init__()

#         self.node_feature_dim = node_feature_dim
#         self.max_nodes = max_nodes

#         # Linear layer to predict node type from encoded protein
#         self.node_type_predictor = nn.Linear(
#             encoded_protein_dim, node_feature_dim)

#         # Sequential model to predict 3D positions for nodes
#         self.position_predictor = nn.Sequential(
#             nn.Linear(encoded_protein_dim + node_feature_dim, 128), nn.ReLU(),
#             nn.Linear(128, 3))

#         # Sequential model to predict edge types using softmax for classification
#         self.edge_predictor = nn.Sequential(
#             nn.Linear(node_feature_dim * 2, 128), nn.ReLU(),
#             nn.Linear(128, 4), nn.Softmax(dim=-1))

#     def forward(self, encoded_protein):
#         device = encoded_protein.device
#         generated_nodes = []  # Store generated node features
#         generated_positions = []  # Store generated positions
#         edge_indices = []  # Store edges

#         for i in range(self.max_nodes):
#             # Generate node features and positions for each node
#             node_feature = self.node_type_predictor(
#                 encoded_protein).unsqueeze(0)
#             position = self.position_predictor(
#                 torch.cat((encoded_protein, node_feature), dim=-1))

#             generated_nodes.append(node_feature)
#             generated_positions.append(position)

#             # Generate edges for new node with existing nodes
#             if i > 0:
#                 edge_data = torch.cat((torch.cat(generated_nodes, dim=0).repeat(
#                     i, 1), node_feature.repeat(i, 1)), dim=1)
#                 bond_types = self.edge_predictor(edge_data)
#                 for j in range(i):
#                     bond_type_idx = torch.argmax(bond_types[j]).item()
#                     if bond_type_idx > 0:  # Exclude bonds with no type
#                         edge_indices.append((j, i))

#         # Compile all generated features and edges into tensors
#         generated_nodes = torch.cat(generated_nodes, dim=0)
#         generated_positions = torch.cat(generated_positions, dim=0)
#         edge_index = torch.tensor(
#             edge_indices, dtype=torch.long, device=device).t().contiguous()
#         # Placeholder for edge attributes
#         edge_attr = torch.tensor(
#             [1] * edge_index.size(1), dtype=torch.long, device=device)

#         # Return PyG Data object representing the generated graph
#         generated_graph = Data(x=generated_nodes, pos=generated_positions,
#                                edge_index=edge_index, edge_attr=edge_attr)
#         return generated_graph


# class DeNovo3D(nn.Module):
#     # Complete model implementation of Encoder and Generator
#     def __init__(self, num_node_features, encoded_protein_dim, node_feature_dim, max_nodes, out_dim):
#         super(DeNovo3D, self).__init__()
#         # Initialize Encoder
#         self.encoder = Encoder(num_node_features, out_dim)
#         self.generator = Generator(
#             encoded_protein_dim, node_feature_dim, max_nodes)  # Initialize Generator

#     def forward(self, protein_graph):
#         # Encode protein graph to get latent representation
#         encoded_protein = self.encoder(
#             protein_graph.x, protein_graph.edge_index, protein_graph.batch)
#         # Generate ligand graph from encoded protein
#         generated_graph = self.generator(encoded_protein)
#         return generated_graph
