import torch

from game import MaxCover
from model import PolicyValueGCN
from train import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 10,
    'numIters': 1,                                # Total number of training iterations
    'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 20,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 1,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}

# game = Connect2Game()
# board_size = game.get_board_size()
# action_size = game.get_action_size()

# model = Connect2Model(board_size, action_size, device)

model = PolicyValueGCN()

# trainer = Trainer(game, model, args)

trainer = Trainer(model, args)
trainer.learn()