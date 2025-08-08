import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from config import NUM_ATOM_CLASSES, DEFAULT_HIDDEN_DIM


class ProteinEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = DEFAULT_HIDDEN_DIM, edge_dim: int = 3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr)))
        g = global_mean_pool(h, batch)
        return g


class LigandSetDecoder(nn.Module):
    """
    Non-autoregressive ligand generator conditioned on protein embedding.
    Predicts per-node existence (mask), atom class, 3D position, and dense bond logits.
    """
    def __init__(self, protein_dim: int, max_nodes: int, hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 num_atom_classes: int = NUM_ATOM_CLASSES, num_bond_classes: int = 5):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_atom_classes = num_atom_classes
        self.num_bond_classes = num_bond_classes

        self.init_mlp = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Per-node predictors
        self.node_exist = nn.Linear(hidden_dim, 1)
        self.atom_head = nn.Linear(hidden_dim, num_atom_classes)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )
        # Bond predictor via pairwise feature fusion
        self.bond_pair = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_bond_classes)
        )

    def forward(self, protein_emb: torch.Tensor):
        # protein_emb: [B, D]
        B, D = protein_emb.shape
        h = self.init_mlp(protein_emb)  # [B, H]
        # expand to N nodes
        h_nodes = h.unsqueeze(1).repeat(1, self.max_nodes, 1)  # [B, N, H]

        exist_logits = self.node_exist(h_nodes).squeeze(-1)  # [B, N]
        atom_logits = self.atom_head(h_nodes)                # [B, N, A]
        pos_pred = self.pos_head(h_nodes)                    # [B, N, 3]

        # pairwise bond logits
        hi = h_nodes.unsqueeze(2).repeat(1, 1, self.max_nodes, 1)
        hj = h_nodes.unsqueeze(1).repeat(1, self.max_nodes, 1, 1)
        pair = torch.cat([hi, hj], dim=-1)
        bond_logits = self.bond_pair(pair)                  # [B, N, N, E]
        return exist_logits, atom_logits, pos_pred, bond_logits


class DeNovo3D_MVP(nn.Module):
    def __init__(self, prot_feat_dim: int, max_lig_nodes: int,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM, num_atom_classes: int = NUM_ATOM_CLASSES,
                 num_bond_classes: int = 5):
        super().__init__()
        self.prot_enc = ProteinEncoder(prot_feat_dim, hidden_dim, edge_dim=3)
        self.decoder = LigandSetDecoder(hidden_dim, max_lig_nodes, hidden_dim,
                                        num_atom_classes, num_bond_classes)

    def forward(self, prot_batch):
        # prot_batch must have x, edge_index, edge_attr, batch
        g = self.prot_enc(prot_batch.x, prot_batch.edge_index, prot_batch.edge_attr, prot_batch.batch)
        return self.decoder(g)


