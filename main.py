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

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    parser = ArgumentParser()

    parser.add_argument( "--dataset", type=str, default='Twitter', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--problem",type=str,default='MaxCut')
    parser.add_argument("--budget",type=int,default=100)
    parser.add_argument("--depth",type=int,default=50)
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    parser.add_argument("--pre_prune", type=bool,default=False, help="Whether to use GNNpruner to pre prune")
    parser.add_argument("--guide_with_expert", type=bool,default=False, help="Guide with expert")
    args = parser.parse_args()

    
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
    pre_prune = args.pre_prune
    guide_with_expert = args.guide_with_expert

    print(f'Training for the problem {problem} Dataset {dataset} Budget {budget}')

    save_folder = f'pretrained/{problem}/{dataset}'
    os.makedirs(save_folder,exist_ok=True)


    if pre_prune:
        # GNN pruning + then training
        save_file_path = os.path.join(save_folder,'best_model_gnnpruner.pth')
    else: 
        save_file_path = os.path.join(save_folder,'best.pth')

    mcts_args = {
        'batch_size': 10,
        'numIters': 10,                                # Total number of training iterations
        'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
        'numEps': 1,
                                                                            # Number of full games (episodes) to run during each iteration
        # 'numItersForTrainExamplesHistory': 20,
        'epochs': 1,                                    # Number of epochs of training per iteration
        'checkpoint_path': save_file_path                # location to save latest set of weights
    }



    model = PolicyValueGCN()
    train_graph = load_graph(f'../snap_dataset/train/{dataset}')
    
    if problem =='MaxCover':
        heuristic = maxcover_heuristic
        problem = MaxCover
    elif problem =='MaxCut':
        heuristic = maxcut_heuristic
        problem = MaxCut
    elif problem == 'IM':
        heuristic = imm
        problem = IM

    else:
        raise ValueError('Unknown Problem')
    


    if pre_prune:

        pruner = GNNpruner()
        save_folder =  f'pretrained/{args.problem}/GNNpruner/{dataset}'
        pruner.train(train_graph=train_graph,budget=budget,heuristic=heuristic,save_folder =save_folder)
        load_model_path = os.path.join(save_folder,'best_model.pth')
        pruner.model.load_state_dict(torch.load(load_model_path,weights_only=False))

    else:
        pruner = None

    game = problem(graph=train_graph,
                  heuristic=heuristic,
                  budget=budget,
                  depth=depth,
                  GNNpruner=pruner,
                  train=True)
   
     
    trainer = Trainer(model=model,game=game,args= mcts_args)
    trainer.learn()