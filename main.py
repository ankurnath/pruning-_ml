import torch

from updated_game import MaxCover,MaxCut,IM
from model import PolicyValueGCN
from train import Trainer
from utils import *
from greedy_maxcover import greedy as maxcover_heuristic
from greedy_maxcut import greedy as maxcut_heuristic
from imm import imm
from gnnpruner_train import *

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--problem",type=str,default='MaxCover')
    parser.add_argument("--budget",type=int,default=100)
    parser.add_argument("--depth",type=int,default=100)
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Get the number of CUDA devices
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print("CUDA Device {}: {}".format(i, device_name))

    if torch.cuda.is_available():
        if args.device is None:
            device = 'cuda:0' 
        else:
            device=f'cuda:{args.device}'

    else:
        device='cpu'


    dataset = args.dataset
    budget = args.budget
    depth = args.depth
    problem = args.problem

    print(f'Training for the problem {problem} Dataset {dataset} Budget {budget}')

    save_folder = f'pretrained/{problem}/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'best.pth')

    mcts_args = {
        'batch_size': 10,
        'numIters': 10,                                # Total number of training iterations
        'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
        'numEps': 1,                                  # Number of full games (episodes) to run during each iteration
        'numItersForTrainExamplesHistory': 20,
        'epochs': 10,                                    # Number of epochs of training per iteration
        'checkpoint_path': save_file_path                # location to save latest set of weights
    }



    model = PolicyValueGCN()

    
    
    graph = load_graph(f'../snap_dataset/train/{dataset}')
    

    # pruner = GNNpruner()

    # save_folder =  f'pretrained/Maxcover/GNNpruner/{dataset}'

    # pruner.train(train_graph=graph,budget=budget,heuristic=greedy,save_folder =save_folder)


    # load_model_path = os.path.join(save_folder,'best_model.pth')
    # pruner.model.load_state_dict(torch.load(load_model_path))
    # game  = MaxCover(graph=graph,heuristic=greedy,budget=budget,depth=depth,GNNpruner=pruner)
    
    if problem =='MaxCover':
    
        game  = MaxCover(graph=graph,heuristic=maxcover_heuristic,budget=budget,depth=depth,GNNpruner=None,train=True)
    elif problem =='MaxCut':
        game  = MaxCut(graph=graph,heuristic=maxcut_heuristic,
                       budget=budget,depth=depth,
                       GNNpruner=None,train=True)

    elif problem == 'IM':
        game = IM(graph=graph,heuristic=imm,
                       budget=budget,depth=depth,
                       GNNpruner=None,train=True)

    else:
        raise ValueError('Unknown Problem')
     

    trainer = Trainer(model=model,game=game,args= mcts_args)
    trainer.learn()