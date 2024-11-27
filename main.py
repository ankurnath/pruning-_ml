import torch

# from game import MaxCover
from updated_game import MaxCover
from model import PolicyValueGCN
from train import Trainer
from utils import *
from greedy import greedy
from gnnpruner_train import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 10,
    'numIters': 10,                                # Total number of training iterations
    'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 1,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 10,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}

# game = Connect2Game()
# board_size = game.get_board_size()
# action_size = game.get_action_size()

# model = Connect2Model(board_size, action_size, device)

model = PolicyValueGCN()

dataset = 'Facebook'
graph = nx.read_edgelist(f'data/snap_dataset/{dataset}.txt',create_using=nx.Graph(), nodetype=int) 
budget = 100
depth = 10

# pruner = GNNpruner()

# save_folder =  f'pretrained/Maxcover/GNNpruner/{dataset}'

# pruner.train(train_graph=graph,budget=budget,heuristic=greedy,save_folder =save_folder)


# load_model_path = os.path.join(save_folder,'best_model.pth')
# pruner.model.load_state_dict(torch.load(load_model_path))
# game  = MaxCover(graph=graph,heuristic=greedy,budget=budget,depth=depth,GNNpruner=pruner)
game  = MaxCover(graph=graph,heuristic=greedy,budget=budget,depth=depth,GNNpruner=None)
# trainer = Trainer(model, args) 

trainer = Trainer(model=model,game=game,args= args)
trainer.learn()