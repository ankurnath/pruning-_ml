import torch

from game import MaxCover
from model import PolicyValueGCN
from train import Trainer
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 10,
    'numIters': 10,                                # Total number of training iterations
    'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 20,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 10,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}

# game = Connect2Game()
# board_size = game.get_board_size()
# action_size = game.get_action_size()

# model = Connect2Model(board_size, action_size, device)

model = PolicyValueGCN()


graph = nx.read_edgelist('data/snap_dataset/Facebook.txt',create_using=nx.Graph(), nodetype=int) 
budget =100
depth = 150
game  = MaxCover(graph=graph,budget=budget,depth=depth)

# trainer = Trainer(model, args) 

trainer = Trainer(model=model,game=game,args= args)
trainer.learn()