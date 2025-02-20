import torch
from utils import *
from torch_geometric.utils.convert import  from_networkx
import pandas as pd

from greedy_maxcover import greedy as maxcover_heuristic
from greedy_maxcut import greedy as maxcut_heuristic
from imm import imm


class GCN(torch.nn.Module):
    def __init__(self, num_features,hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x



class GNNpruner:


    def __init__(self,model = None) -> None:
        if model:
            self.model = model
        else:
            self.model = GCN(num_features=1,hidden_channels=16)

    def test(self, test_graph, threshold=0.5):
        # Convert NetworkX graph to PyTorch Geometric data
        test_data = from_networkx(test_graph)
        test_data.x = torch.rand(size=(test_graph.number_of_nodes(), 1))  # Random node features

        # Device setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        test_data = test_data.to(device)

        # Model evaluation
        self.model.eval()
        with torch.no_grad():  # Disable gradient computation for inference
            out = self.model(test_data.x, test_data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1]  # Probability of class 1

        # Apply threshold
        pred = (probs >= threshold).cpu().numpy()
        indices = np.where(pred == 1)[0]


        # Map indices back to node labels
        reverse_mapping = dict(zip(range(test_graph.number_of_nodes()), test_graph.nodes()))
        pruned_universe = [reverse_mapping[node] for node in indices]

        return pruned_universe

    # def test(self,test_graph,threshold=0.5):

    #     # start = time.time()
    #     test_data = from_networkx(test_graph)
    #     test_data.x = torch.rand(size=(test_graph.number_of_nodes(),1))
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     self.model = self.model.to(device)
    #     test_data = test_data.to(device)
    #     out = self.model(test_data.x, test_data.edge_index)
    #     pred = out.argmax(dim=1).cpu().numpy()  # Use the class with highest probability.
    #     indices = np.where(pred == 1)[0]
    #     reverse_mapping = dict(zip(range(test_graph.number_of_nodes()),test_graph.nodes()))
    #     pruned_universe = [reverse_mapping[node] for node in indices]
    #     # end = time.time()
    #     return pruned_universe



    def train(self,train_graph,budget,heuristic,save_folder):
        data= from_networkx(train_graph)

        _,solution,_=heuristic(graph=train_graph,budget=budget,ground_set=None)



        mapping = dict(zip(train_graph.nodes(), range(train_graph.number_of_nodes())))
        train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
        # class_weights = torch.tensor([1-len(train_mask)/train_graph.number_of_nodes() , len(train_mask)/train_graph.number_of_nodes()], dtype=torch.float)
        y=torch.zeros(train_graph.number_of_nodes(),dtype=torch.long)

        for node in solution:
            y[mapping[node]]=1


        data.y = y
        data.x = torch.rand(size=(train_graph.number_of_nodes(),1))

        num_classes = int(data.y.max().item() + 1)  # Determine the number of classes.
        class_counts = torch.bincount(y)  # Count occurrences of each class
        total_samples = class_counts.sum()
        class_weights = total_samples / (num_classes * class_counts)
        

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        class_weights = class_weights.to(torch.float).to(device=device)
        self.model = self.model.to(device=device)
        data.to(device=device)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


        self.model.train()


        best_loss = float('inf')  # Initialize the best training loss to infinity

        os.makedirs(save_folder,exist_ok=True)

        save_file_path = os.path.join(save_folder,'best_model.pth')

        for epoch in tqdm(range(1000)):

            out = self.model(data.x, data.edge_index)  # Perform a single forward pass.

            # mask=torch.cat([train_mask,torch.randint(graph.number_of_nodes())],axis=0)
            # mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
            # print('Mask size',train_mask.shape)
            # print('Mask size',mask.shape)

            # print(torch.sum(data.y[mask]))
            # loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
            loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

            # Save the best model if training loss improves
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), save_file_path)  # Save the model's state dictionary



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budget", type= int , default= 100, help="Budget" )
    parser.add_argument( "--problem", type= str , default= "MaxCover", help="Problem" )
    args = parser.parse_args()


    dataset = args.dataset
    budget = args.budget
    problem = args.problem

    if problem == 'MaxCover':
        heuristic = maxcover_heuristic
    elif problem == 'MaxCut':
        heuristic = maxcut_heuristic
    elif problem == 'IM':
        heuristic = imm
    else:
        raise ValueError('Unknown Problem')


    # file_path = f'data/snap_dataset/{dataset}.txt'
    train_graph =  load_graph(f'../snap_dataset/train/{dataset}')


    pruner = GNNpruner()

    save_folder =  f'pretrained/{problem}/GNNpruner/{dataset}'

    pruner.train(train_graph=train_graph,
                 budget=budget,
                 heuristic=heuristic,
                 save_folder =save_folder)
    
   
    # load_model_path = os.path.join(save_folder,'best_model.pth')
    # pruner.model.load_state_dict(torch.load(load_model_path))
    # test_graph = load_graph(f'../snap_dataset/test/{dataset}')
    # pruned_universe = pruner.test(test_graph=test_graph)

    # Pg = len(pruned_universe)/test_graph.number_of_nodes()
    # objective_unpruned, solution_unpruned, queries_unpruned = heuristic(test_graph,budget)
    # objective_pruned,solution_pruned, queries_pruned = heuristic(graph=test_graph,
    #                                                              budget=budget,
    #                                                              ground_set=pruned_universe)
    # ratio = objective_pruned/objective_unpruned

    # print('Performance of GCNPruner')
    # print('Size Constraint,k:',budget)
    # print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
    # print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    # print('Pg(%):', round(1-Pg,4)*100)
    # print('Ratio:',round(ratio,4)*100)
    # print('C',round((1-Pg)*ratio,4)*100)





    






