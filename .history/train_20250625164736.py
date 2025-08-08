import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class ProteinEncoder(nn.Module):
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

# Loss functions

def loss_function(stop_logits, atom_logits, pos_preds, edge_preds,
                  true_stop, true_atom, true_pos, true_edges,
                  mu, logvar, beta=1.0):
    """
    Compute total VAE loss:
      - Stop token: BCE
      - Atom type: CE
      - Position: MSE
      - Edge: CE over bond types (including no bond)
      - KL divergence (beta-weighted)
    """
    # Stop loss
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits.squeeze(-1), true_stop.float()
    )
    # Atom type loss
    B, N, A = atom_logits.size()
    atom_loss = F.cross_entropy(
        atom_logits.view(B * N, A), true_atom.view(B * N)
    )
    # Position loss
    pos_loss = F.mse_loss(pos_preds, true_pos)
    # Edge loss
    if edge_preds is not None:
        # flatten batch & time & prev-nodes dims
        B, T, P, E = edge_preds.size()
        edge_loss = F.cross_entropy(
            edge_preds.view(B * T * P, E), true_edges.view(B * T * P)
        )
    else:
        edge_loss = torch.tensor(0.0, device=mu.device)
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = stop_loss + atom_loss + pos_loss + edge_loss + beta * kl_loss
    return total, {'stop': stop_loss, 'atom': atom_loss,
                   'pos': pos_loss, 'edge': edge_loss,
                   'kl': kl_loss}
