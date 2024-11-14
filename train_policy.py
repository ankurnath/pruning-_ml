import torch
from torch_geometric.loader import DataLoader
import os

# Define the folder where the graph data is saved
train_folder = 'train_folder'

# Load all graph data files into a list
graph_files = [f"{train_folder}/graph_{i}.pt" for i in range(len(os.listdir(train_folder)))]

# Load the data from the files
dataset = [torch.load(file) for file in graph_files]

# Create a DataLoader with a batch size of 2
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # Second GCN layer
        x = self.conv2(x, edge_index)
        return x





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.optim import Adam
import torch.nn.functional as F

# Initialize the model, optimizer, and loss function
model = GCN(in_channels=1, out_channels=1)  # assuming 100 features for each node and 2 output classes
model.to(device)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 10  # number of epochs for training
for epoch in range(epochs):
    model.train()
    for data in dataloader:
        # Move data to the appropriate device (GPU or CPU)
        data = data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        out = model(data)

        # print(out.shape)
        # print(data.y.shape)
        # break


    #     # Compute loss (using data.y as the ground truth labels)
        loss = criterion(out, data.y)
        # break

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.10f}")

