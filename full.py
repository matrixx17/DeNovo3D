import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch.optim import Adam
from torch.utils.data import random_split
import matplotlib.pyplot as plt


class Encoder(nn.Module):

    def __init__(self, num_node_features, out_dim):
        super(Encoder, self).__init__()

        self.conv1 = GATConv(num_node_features, 128)
        self.bn1 = nn.BatchNorm1d(128)

        # Dropout layer added for regularization
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = GATConv(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)  # Another dropout layer

        self.conv3 = GATConv(128, out_dim)

        # Global pooling to aggregate node features into a graph-level representation
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout1(x)  # Applying dropout

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout2(x)  # Applying another dropout

        x = self.conv3(x, edge_index)

        # Aggregate node features to get a single vector for the entire graph
        x = self.pool(x, batch)

        return x


class Generator(nn.Module):
    def __init__(self, encoded_protein_dim, node_feature_dim, max_nodes):
        super(Generator, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.max_nodes = max_nodes

        # Predicting node type from encoded protein representation
        self.node_type_predictor = nn.Linear(
            encoded_protein_dim, node_feature_dim)

        # Predicting 3D positions for atoms
        self.position_predictor = nn.Sequential(
            nn.Linear(encoded_protein_dim + node_feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 3))

        # Predicting edge types between atoms; using softmax for bond type classification
        self.edge_predictor = nn.Sequential(
            nn.Linear(node_feature_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 4), nn.Softmax(dim=-1))

    def forward(self, encoded_protein):
        device = encoded_protein.device
        generated_nodes = []
        generated_positions = []
        edge_indices = []

        for i in range(self.max_nodes):
            node_feature = self.node_type_predictor(
                encoded_protein).unsqueeze(0)
            position = self.position_predictor(
                torch.cat((encoded_protein, node_feature), dim=-1))

            generated_nodes.append(node_feature)
            generated_positions.append(position)

            if i > 0:
                # Use previously generated nodes to predict connections
                edge_data = torch.cat((torch.cat(generated_nodes, dim=0).repeat(
                    i, 1), node_feature.repeat(i, 1)), dim=1)
                bond_types = self.edge_predictor(edge_data)
                for j in range(i):
                    bond_type_idx = torch.argmax(bond_types[j]).item()
                    if bond_type_idx > 0:  # Only considering bonds with a type
                        edge_indices.append((j, i))

        generated_nodes = torch.cat(generated_nodes, dim=0)
        generated_positions = torch.cat(generated_positions, dim=0)
        edge_index = torch.tensor(
            edge_indices, dtype=torch.long, device=device).t().contiguous()
        # Placeholder for edge attributes
        edge_attr = torch.tensor(
            [1] * edge_index.size(1), dtype=torch.long, device=device)

        # Construct and return the PyG Data object representing the generated graph
        generated_graph = Data(x=generated_nodes, pos=generated_positions,
                               edge_index=edge_index, edge_attr=edge_attr)
        return generated_graph


class DeNovo3D(nn.Module):
    def __init__(self, num_node_features, encoded_protein_dim, node_feature_dim, max_nodes, out_dim):
        super(DeNovo3D, self).__init__()
        self.encoder = Encoder(num_node_features, out_dim)
        self.generator = Generator(
            encoded_protein_dim, node_feature_dim, max_nodes)

    def forward(self, protein_graph):
        encoded_protein = self.encoder(
            protein_graph.x, protein_graph.edge_index, protein_graph.batch)
        generated_graph = self.generator(encoded_protein)
        return generated_graph


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


def collate_fn(data_list):
    """Custom collate function to handle tuples of (protein, ligand)."""
    batch_protein = Batch.from_data_list([data[0] for data in data_list])
    batch_ligand = Batch.from_data_list([data[1] for data in data_list])
    return batch_protein, batch_ligand


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    losses = []
    for batch_protein, batch_ligand in train_loader:
        batch_protein = batch_protein.to(device)
        batch_ligand = batch_ligand.to(device)
        optimizer.zero_grad()
        generated_graph = model(batch_protein)
        # Ensure combined_loss can handle batched input
        loss = combined_loss_fn(generated_graph, batch_ligand)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        losses.append(loss.item())
    print(f"Total training loss: {total_loss}")
    return losses


def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    losses = []
    with torch.no_grad():
        for batch_protein, batch_ligand in test_loader:
            batch_protein = batch_protein.to(device)
            batch_ligand = batch_ligand.to(device)
            generated_graph = model(batch_protein)
            loss = combined_loss_fn(generated_graph, batch_ligand)
            total_loss += loss.item()
            losses.append(loss.item())
    print(f"Average test loss: {total_loss / len(test_loader.dataset)}")
    return losses


# Define baseline model parameters
num_node_features = 5
encoded_protein_dim = 128
node_feature_dim = 13
max_nodes = 150
out_dim = 128  # Output dimensionality of the encoder
learning_rate = 0.01  # Learning rate for the optimizer
epochs = 50


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = torch.load('dataset.pt')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(list(train_dataset), batch_size=32,
                              shuffle=True, follow_batch=['x'], collate_fn=collate_fn)
    test_loader = DataLoader(list(test_dataset), batch_size=32,
                             shuffle=False, follow_batch=['x'], collate_fn=collate_fn)

    model = DeNovo3D(num_node_features, encoded_protein_dim,
                     node_feature_dim, max_nodes, out_dim)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(model, train_loader, optimizer)
        test_loss = test(model, test_loader)
        train_losses.extend(train_loss)
        test_losses.extend(test_loss)

    # Save trained model
    model_path = "path/to/save/model_state.pth"
    optimizer_path = "path/to/save/optimizer_state.pth"

    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    # Plotting the training and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
