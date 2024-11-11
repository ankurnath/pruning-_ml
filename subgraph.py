import networkx as nx
import torch
from torch_geometric.utils import from_networkx,subgraph,k_hop_subgraph
from torch_geometric.data import Data

# Step 1: Create a simple graph using networkx
G = nx.Graph()

# Adding nodes
G.add_nodes_from([0, 1, 2, 3])

# Adding edges
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0),(1,3)])

# Step 2: Convert to PyTorch Geometric graph
data = from_networkx(G)

# Step 3: Verify the data
print(data)

# extracted_graph=subgraph(subset=[0,1],edge_index=data.edge_index)

extracted_graph=k_hop_subgraph(node_idx=[0],num_hops=1,edge_index=data.edge_index)

print(extracted_graph)