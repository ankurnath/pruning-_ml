from utils import *

class PolicyValueGCN(nn.Module):
    def __init__(self, in_channels = 1,out_channels=1):
        super(PolicyValueGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)
        self.policy_head = nn.Linear(out_channels,1)
        self.value_head = nn.Linear(out_channels, 1)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))

        # Second GCN layer
        x = self.conv2(x, edge_index)

        policy_logits = self.policy_head(x)
        # policy = F.softmax(policy_logits, dim=-1)

        # Initialize an empty tensor to store the softmax results
        policy= torch.zeros_like(x)
        
        for graph_id in range(data.num_graphs):
            # Mask for nodes belonging to the current graph
            mask = data.batch == graph_id
            
            # Apply softmax over the node features for this graph
            policy[mask] = F.softmax(policy_logits[mask], dim=0)  # Apply softmax along the feature dimension for each graph

        # Step 2: Global pooling (mean pooling in this case)
        x = global_mean_pool(x, data.batch)

        value = self.value_head(x)

        return policy,F.sigmoid(value) 
    


# ### Sanity check

# from torch_geometric.data import Data, Batch

# # Initialize an empty list to store graph data objects
# data_list = []

# # Create multiple random graph data objects (example: 10 nodes, with edges generated randomly)
# for _ in range(2):
#     # Generate a random graph using Erdos-Renyi model (10 nodes, p=0.1 probability for edges)
#     graph = nx.erdos_renyi_graph(n=10, p=0.1)
    
#     # Create node features: All ones (shape: num_nodes x num_node_features)
#     node_features = torch.ones((graph.number_of_nodes(), 1))  # 10 nodes, 1 feature per node
    
#     # Convert NetworkX graph to PyG Data object
#     data = from_networkx(graph)
    
#     # Add node features to the Data object
#     data.x = node_features
    
#     # Append the data object to the list
#     data_list.append(data)


# # print()

# # Batch the list of Data objects into a single Batch object
# data = Batch.from_data_list(data_list)
# # data.nu
# # print(data.batch)
# # print(data.x)



# model = PolicyValueGCN()

# policy,values = model(data=data)

# print(policy)
# print(values)



