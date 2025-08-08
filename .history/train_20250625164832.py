import torch
from torch_geometric.data import DataLoader
from torch.optim import Adam
from denovo3d_model import DeNovo3D, loss_function  # your VAE model and loss


def train(model: DeNovo3D, train_loader: DataLoader, optimizer: Adam, device: torch.device):
    """
    Single-epoch training loop for conditional VAE.
    Expects train_loader to yield (prot_graph, lig_graph) batches.
    """
    model.train()
    total_loss = 0.0

    for prot_batch, lig_batch in train_loader:
        # Move graphs to device
        prot_batch = prot_batch.to(device)
        lig_batch = lig_batch.to(device)

        optimizer.zero_grad()
        # Forward pass: returns stop_logits, atom_logits, pos_preds, edge_preds, mu, logvar
        outputs = model(prot_batch, lig_batch)

        # Extract ground-truth labels for teacher forcing
        # Assumes lig_batch includes these tensors:
        #   lig_batch.stop_tokens: [B, max_nodes] (0/1)
        #   lig_batch.atom_labels: [B, max_nodes] (int atom types)
        #   lig_batch.pos_labels: [B, max_nodes, 3]
        #   lig_batch.edge_labels: [B, max_nodes, max_nodes_prev] (int bond types)
        true_stop = lig_batch.stop_tokens
        true_atom = lig_batch.atom_labels
        true_pos  = lig_batch.pos_labels
        true_edges= lig_batch.edge_labels

        # Compute VAE loss (reconstruction + KL)
        loss, loss_dict = loss_function(
            outputs[0], outputs[1], outputs[2], outputs[3],
            true_stop, true_atom, true_pos, true_edges,
            outputs[4], outputs[5]
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training avg loss: {avg_loss:.4f}")


def evaluate(model: DeNovo3D, eval_loader: DataLoader, device: torch.device):
    """
    Single-epoch evaluation loop (no gradient updates).
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for prot_batch, lig_batch in eval_loader:
            prot_batch = prot_batch.to(device)
            lig_batch  = lig_batch.to(device)

            outputs = model(prot_batch, lig_batch)

            true_stop = lig_batch.stop_tokens
            true_atom = lig_batch.atom_labels
            true_pos  = lig_batch.pos_labels
            true_edges= lig_batch.edge_labels

            loss, _ = loss_function(
                outputs[0], outputs[1], outputs[2], outputs[3],
                true_stop, true_atom, true_pos, true_edges,
                outputs[4], outputs[5]
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    print(f"Evaluation avg loss: {avg_loss:.4f}")


if __name__ == '__main__':
    from graph_constructor import build_protein_graph, build_ligand_graph
    # Example usage
    dataset = []  # populate your (prot_graph, lig_graph) tuples
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeNovo3D( prot_feat_dim=21,
                      lig_feat_dim=32,  # adjust to your max
                      hidden_dim=128,
                      latent_dim=64,
                      max_lig_nodes=50,
                      edge_feature_dim=3 ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(model, loader, optimizer, device)
    evaluate(model, loader, device)
