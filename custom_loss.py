import torch
import torch.nn.functional as F


def mvp_generation_loss(
    exist_logits, atom_logits, pos_pred, bond_logits,
    true_atom, true_pos, true_bonds,
    node_mask,
    w_exist: float = 0.5,
    w_atom: float = 1.0,
    w_pos: float = 1.0,
    w_bond: float = 1.0,
):
    """
    Minimal viable loss for non-autoregressive ligand generation.
    Inputs (batched):
      - atom_logits: [B, N, A]
      - pos_pred:    [B, N, 3]
      - bond_logits: [B, N, N, E] (last dim is bond classes)
      - true_atom:   [B, N] (class ids)
      - true_pos:    [B, N, 3]
      - true_bonds:  [B, N, N] (class ids; 0 is no-bond)
      - node_mask:   [B, N] (1 if node exists)
    """
    B, N, A = atom_logits.shape
    # Node existence BCE
    exist_loss = F.binary_cross_entropy_with_logits(exist_logits, node_mask.float())
    # Atom type CE with mask
    atom_loss = F.cross_entropy(
        atom_logits.view(B * N, A), true_atom.view(B * N), reduction='none'
    )
    atom_loss = (atom_loss.view(B, N) * node_mask.float()).sum() / (node_mask.float().sum() + 1e-6)

    # Position MSE with mask on nodes
    pos_mse = F.mse_loss(pos_pred, true_pos, reduction='none').sum(dim=-1)
    pos_loss = (pos_mse * node_mask.float()).sum() / (node_mask.float().sum() + 1e-6)

    # Bond CE with mask on node pairs (mask outer product)
    E = bond_logits.size(-1)
    pair_mask = node_mask.float().unsqueeze(2) * node_mask.float().unsqueeze(1)
    bond_loss = F.cross_entropy(
        bond_logits.view(B * N * N, E), true_bonds.view(B * N * N), reduction='none'
    ).view(B, N, N)
    bond_loss = (bond_loss * pair_mask).sum() / (pair_mask.sum() + 1e-6)

    total = w_exist * exist_loss + w_atom * atom_loss + w_pos * pos_loss + w_bond * bond_loss
    return total, {'exist': exist_loss, 'atom': atom_loss, 'pos': pos_loss, 'bond': bond_loss}


def combined_loss_fn(predicted_graph, target_graph, node_loss_weight=0.5, edge_loss_weight=0.5, position_loss_weight=0.5):
    """
    Calculates a hybrid loss for ligand graphs, considering:
    - Categorical classification loss for non-positional node features.
    - MSE loss for positional node features (3D atom positions).
    - Cross-entropy loss for edge attributes (bond types).
    """

    # First node feature is categorical (element type)
    categorical_features_pred = predicted_graph.x[:, :1]
    categorical_features_target = target_graph.x[:, :1]

    # Indices 1-3 are continuous (atom positions)
    positions_pred = predicted_graph.x[:, 1:4]  # Slice for positional features
    positions_target = target_graph.x[:, 1:4]

    # All other categorical features
    other_categorical_features_pred = predicted_graph.x[:, 4:]
    other_categorical_features_target = target_graph.x[:, 4:]

    # Compute MSE loss for positions
    position_loss = F.mse_loss(
        positions_pred, positions_target) * position_loss_weight

    # Compute cross-entropy loss for categorical node features
    categorical_loss = F.cross_entropy(categorical_features_pred, categorical_features_target.long()) + \
        F.cross_entropy(other_categorical_features_pred,
                        other_categorical_features_target.long())
    categorical_loss *= node_loss_weight

    # Compute cross-entropy loss for edge attributes
    edge_feature_loss = F.cross_entropy(
        predicted_graph.edge_attr, target_graph.edge_attr.long()) * edge_loss_weight

    # Calculate total loss
    total_loss = position_loss + categorical_loss + edge_feature_loss

    return total_loss
