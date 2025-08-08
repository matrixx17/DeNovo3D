from torch_geometric.data import Data, DataLoader, Batch
from torch.optim import Adam


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_protein, batch_ligand in train_loader:
        optimizer.zero_grad()
        generated_graph = model(batch_protein)
        # Ensure combined_loss can handle batched input
        loss = combined_loss_fn(generated_graph, batch_ligand)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Total training loss: {total_loss}")


def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            protein_graph, ligand_graph = batch.protein_graph, batch.ligand_graph
            generated_graph = model(protein_graph)
            loss = combined_loss(generated_graph, ligand_graph)
            total_loss += loss.item()
    print(f"Average test loss: {total_loss / len(test_loader.dataset)}")
