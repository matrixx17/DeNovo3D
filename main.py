import os
import torch
from torch_geometric.data import Batch, DataLoader
from torch.optim import Adam
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from process_dataset import build_example_dataset
from mvp_model import DeNovo3D_MVP
from custom_loss import mvp_generation_loss
from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LR,
    DEFAULT_MAX_LIG_NODES,
)


def collate_fn(data_list):
    batch_protein = Batch.from_data_list([data[0] for data in data_list])
    batch_ligand = Batch.from_data_list([data[1] for data in data_list])
    return batch_protein, batch_ligand


def build_labels_from_ligand_batch(lig_batch, max_nodes):
    # Build dense labels: pad/trim to max_nodes
    B = lig_batch.num_graphs
    device = lig_batch.x.device
    # Compute per-graph slices
    ptr = lig_batch.__slices__['x'] if hasattr(lig_batch, '__slices__') else None
    if ptr is None:
        # Fallback: infer using batch vector
        counts = torch.bincount(lig_batch.batch)
        ptr_list = [0]
        acc = 0
        for c in counts.tolist():
            acc += c
            ptr_list.append(acc)
        ptr = {'x': ptr_list}

    atom_labels = torch.full((B, max_nodes), 0, dtype=torch.long, device=device)
    node_mask = torch.zeros((B, max_nodes), dtype=torch.bool, device=device)
    pos_labels = torch.zeros((B, max_nodes, 3), dtype=torch.float, device=device)
    bond_labels = torch.zeros((B, max_nodes, max_nodes), dtype=torch.long, device=device)

    start_indices = ptr['x'] if isinstance(ptr, dict) else ptr
    for b in range(B):
        start = start_indices[b]
        end = start_indices[b + 1]
        n = end - start
        n_clip = min(n, max_nodes)
        atom_class = lig_batch.atom_class[start:start + n_clip]
        atom_labels[b, :n_clip] = atom_class
        node_mask[b, :n_clip] = True
        pos_labels[b, :n_clip] = lig_batch.pos[start:start + n_clip]
        # Build dense adjacency label of bond types using edge_index/edge_attr
        mask0 = (lig_batch.batch[lig_batch.edge_index[0]] == b)
        mask1 = (lig_batch.batch[lig_batch.edge_index[1]] == b)
        emask = mask0 & mask1
        ei = lig_batch.edge_index[:, emask]
        eattr = lig_batch.edge_attr[emask]
        for k in range(ei.size(1)):
            i = ei[0, k].item() - start
            j = ei[1, k].item() - start
            if i < max_nodes and j < max_nodes and 0 <= i < n_clip and 0 <= j < n_clip:
                # edge_attr: [distance, type_code(2 for ligand), bond_order]
                bond_order = eattr[k, 2].item()
                # map to classes: 0 no-bond, 1 single, 2 double, 3 triple, 4 aromatic (~1.5)
                if abs(bond_order - 1.5) < 1e-3:
                    cls = 4
                elif abs(bond_order - 1.0) < 1e-3:
                    cls = 1
                elif abs(bond_order - 2.0) < 1e-3:
                    cls = 2
                elif abs(bond_order - 3.0) < 1e-3:
                    cls = 3
                else:
                    cls = 0
                bond_labels[b, i, j] = cls
    return atom_labels, pos_labels, bond_labels, node_mask


def train_epoch(model, loader, optimizer, device, max_nodes):
    model.train()
    total = 0.0
    for prot_batch, lig_batch in loader:
        prot_batch = prot_batch.to(device)
        lig_batch = lig_batch.to(device)
        optimizer.zero_grad()
        exist_logits, atom_logits, pos_pred, bond_logits = model(prot_batch)
        true_atom, true_pos, true_bonds, node_mask = build_labels_from_ligand_batch(lig_batch, max_nodes)
        loss, _ = mvp_generation_loss(exist_logits, atom_logits, pos_pred, bond_logits,
                                      true_atom, true_pos, true_bonds, node_mask)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def eval_epoch(model, loader, device, max_nodes):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for prot_batch, lig_batch in loader:
            prot_batch = prot_batch.to(device)
            lig_batch = lig_batch.to(device)
            exist_logits, atom_logits, pos_pred, bond_logits = model(prot_batch)
            true_atom, true_pos, true_bonds, node_mask = build_labels_from_ligand_batch(lig_batch, max_nodes)
            loss, _ = mvp_generation_loss(exist_logits, atom_logits, pos_pred, bond_logits,
                                          true_atom, true_pos, true_bonds, node_mask)
            total += loss.item()
    return total / max(1, len(loader))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load dataset if exists; otherwise, build example
    if os.path.exists('dataset.pt'):
        dataset = torch.load('dataset.pt')
    else:
        dataset = build_example_dataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    loader_kwargs = dict(batch_size=DEFAULT_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    train_loader = DataLoader(list(train_dataset), **loader_kwargs)
    test_loader = DataLoader(list(test_dataset), batch_size=DEFAULT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Infer protein feature dim from first sample
    prot_feat_dim = dataset[0][0].x.size(1)
    max_nodes = DEFAULT_MAX_LIG_NODES

    model = DeNovo3D_MVP(prot_feat_dim=prot_feat_dim, max_lig_nodes=max_nodes).to(device)
    optimizer = Adam(model.parameters(), lr=DEFAULT_LR)

    train_losses, test_losses = [], []
    for epoch in range(1, DEFAULT_EPOCHS + 1):
        tr = train_epoch(model, train_loader, optimizer, device, max_nodes)
        te = eval_epoch(model, test_loader, device, max_nodes)
        train_losses.append(tr)
        test_losses.append(te)
        print(f'Epoch {epoch:03d} | train {tr:.4f} | test {te:.4f}')

    torch.save(model.state_dict(), 'mvp_model_state.pth')

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss'); plt.tight_layout()
    plt.savefig('loss_curve.png')


if __name__ == '__main__':
    main()
