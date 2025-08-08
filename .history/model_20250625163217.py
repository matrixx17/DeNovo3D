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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing, global_mean_pool

class ProteinEncoder(nn.Module):
    """
    Encodes protein nodes into hidden embeddings.
    Input: prot_x [N_prot x F_prot], edge_index, edge_attr
    Output: h_prot [N_prot x D]
    """
    def __init__(self, in_channels, hidden_dim, num_edge_features):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=num_edge_features)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    def forward(self, prot_x, edge_index, edge_attr):
        x = F.relu(self.bn1(self.conv1(prot_x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        return x

class LigandEncoder(nn.Module):
    """
    Encodes ligand nodes into hidden embeddings.
    Input: lig_x [N_lig x F_lig], edge_index, edge_attr
    Output: h_lig [N_lig x D]
    """
    def __init__(self, in_channels, hidden_dim, num_edge_features):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=num_edge_features)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    def forward(self, lig_x, edge_index, edge_attr, offset):
        # offset edges indices by n_prot handled outside
        x = F.relu(self.bn1(self.conv1(lig_x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        return x

class ComplexEncoder(nn.Module):
    """
    Aggregates protein + ligand embeddings into latent z.
    Implements a conditional VAE encoder.
    """
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)
    def forward(self, h_prot, h_lig, prot_mask):
        # global pooling
        gp = global_mean_pool(h_prot, prot_mask)
        gl = global_mean_pool(h_lig, 1 - prot_mask)
        h = torch.cat([gp, gl], dim=-1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class LigandDecoder(MessagePassing):
    """
    Graph-based decoder: autoregressively add nodes & edges.
    Conditioned on z (latent) and protein embeddings.
    """
    def __init__(self, latent_dim, hidden_dim, num_edge_types):
        super().__init__(aggr='add')
        self.fc_node = nn.Linear(latent_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_edge_types)
        )
    def forward(self, z, max_nodes):
        # Initialize node embeddings from z
        h = self.fc_node(z).unsqueeze(1).repeat(1, max_nodes, 1)
        # placeholder: actual edge creation logic here
        # return node_features, edge_index, edge_attr
        raise NotImplementedError

class DeNovo3D(nn.Module):
    """
    Conditional VAE for de novo ligand generation.
    """
    def __init__(self, prot_feat_dim, lig_feat_dim, hidden_dim, latent_dim, max_lig_nodes, edge_feature_dim):
        super().__init__()
        self.prot_encoder = ProteinEncoder(prot_feat_dim, hidden_dim, edge_feature_dim)
        self.lig_encoder  = LigandEncoder(lig_feat_dim, hidden_dim, edge_feature_dim)
        self.complex_encoder = ComplexEncoder(hidden_dim, latent_dim)
        self.decoder = LigandDecoder(latent_dim + hidden_dim, hidden_dim, num_edge_types=4)
        self.max_lig_nodes = max_lig_nodes
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, prot_data, lig_data):
        # Encode subgraphs
        h_prot = self.prot_encoder(prot_data.prot_x, prot_data.edge_index, prot_data.edge_attr)
        h_lig = self.lig_encoder(lig_data.lig_x, lig_data.edge_index + prot_data.prot_x.size(0), lig_data.edge_attr)
        # form latent
        mu, logvar = self.complex_encoder(h_prot, h_lig, prot_mask=torch.ones(h_prot.size(0), dtype=torch.long))
        z = self.reparameterize(mu, logvar)
        # Condition decoder on both z and prot summary
        prot_summary = global_mean_pool(h_prot, batch=torch.zeros(h_prot.size(0), dtype=torch.long))
        cond = torch.cat([z, prot_summary], dim=-1)
        # Decode ligand
        node_feats, edge_index, edge_attr = self.decoder(cond, self.max_lig_nodes)
        return node_feats, edge_index, edge_attr, mu, logvar
