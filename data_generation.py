from utils import *
import networkx as nx
import numpy as np
from collections import defaultdict
import torch
from torch_geometric.utils import from_networkx

N = 100
budget = 5
core_size = 10
num_train = 10
samples_per_graph = 10 
idx = 0 

# Create train_folder if it doesn't exist
train_folder = 'train_folder'
os.makedirs(train_folder, exist_ok=True)



for _ in range(num_train//samples_per_graph):


    # Generate Barabasi-Albert graph
    graph = nx.barabasi_albert_graph(n=N, m=4)

    for _ in range(samples_per_graph):

        # Random core size
        random_core_size = np.random.randint(1, core_size + 1)

        # Randomly select nodes
        nodes = np.random.choice(graph.nodes, size=random_core_size, replace=False)

        # Set of covered nodes
        covered = set()
        for node in nodes:
            covered.add(node)
            for neighbor in graph.neighbors(node):
                covered.add(neighbor)

        # Calculate gains
        gains = defaultdict(int)
        for node in graph.nodes:
            if node not in covered:
                gains[node] += 1
            for neighbor in graph.neighbors(node):
                gains[neighbor] += 1

        # Convert graph to PyTorch geometric data
        data = from_networkx(graph)

        # Initialize x (features) for the graph
        x = torch.zeros(size=(N,1))
        x[nodes] = 1  # Mark selected nodes
        data.x = x

        # Find node with maximum gain
        max_node = max(gains, key=gains.get)

        # Initialize y (labels) for the graph
        data.y = torch.zeros(size=(N,1))
        data.y[max_node] = 1  # Mark the node with maximum gain

        # Save the data to train_folder
        filename = f"{train_folder}/graph_{idx}.pt"
        torch.save(data, filename)
        idx +=1

        print(f"Graph {idx} saved to {filename}")


# print(data.y)