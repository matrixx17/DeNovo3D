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
    def __init__(self, in_channels, hidden_dim, edge_dim):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr)))
        return h

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

class LigandDecoder(nn.Module):
    """
    Autoregressive graph decoder based on GraphRNN principles:
      - At each step, predicts stop token, atom type, 3D position
      - Predicts edges to previous nodes via Gumbel-Softmax for differentiability
    """
    def __init__(self, latent_dim, prot_summary_dim, hidden_dim, atom_types, bond_types, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        # Fuse latent + protein context
        self.init_mlp = nn.Sequential(
            nn.Linear(latent_dim + prot_summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Stop token predictor per timestep
        self.stop_mlp = nn.Linear(hidden_dim, 1)
        # Atom type predictor
        self.atom_mlp = nn.Linear(hidden_dim, atom_types)
        # Position predictor
        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        # Edge existence & bond type predictor
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bond_types)
        )

    def forward(self, z, prot_summary):
        batch_size = z.size(0)
        device = z.device
        # Initialize hidden state for nodes
        h_prev = self.init_mlp(torch.cat([z, prot_summary], dim=-1))
        node_feats = []
        pos_preds = []
        edge_preds = []  # list of [batch, i, bond_logits]
        stop_logits = []

        for t in range(self.max_nodes):
            # stop prediction
            stop_logit = self.stop_mlp(h_prev)
            stop_logits.append(stop_logit)
            # atom type prediction
            atom_logit = self.atom_mlp(h_prev)
            node_feats.append(atom_logit)
            # position prediction
            pos_pred = self.pos_mlp(h_prev)
            pos_preds.append(pos_pred)
            # edges to previous nodes
            if t > 0:
                # compute pairwise for each new node vs all prev
                h_repeat_i = h_prev.unsqueeze(1).repeat(1, t, 1)
                h_repeat_j = torch.stack(node_feats[:-1], dim=1)  # [B, t, T]
                h_pair = torch.cat([h_repeat_i, h_repeat_j], dim=-1)
                bond_logits = self.edge_mlp(h_pair)  # [B, t, bond_types]
                edge_preds.append(bond_logits)
            # update h_prev (for simplicity, reuse same for all steps)
            # In practice, could use an RNN to update hidden state.
        # Stack outputs
        stop_logits = torch.stack(stop_logits, dim=1)  # [B, max_nodes, 1]
        atom_logits = torch.stack(node_feats, dim=1)   # [B, max_nodes, atom_types]
        pos_preds = torch.stack(pos_preds, dim=1)      # [B, max_nodes, 3]
        edge_preds = torch.stack(edge_preds, dim=1) if edge_preds else None
        return stop_logits, atom_logits, pos_preds, edge_preds

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