from utils import *

from model import PolicyValueGCN

from updated_game import MaxCover,MaxCut,IM

from greedy_maxcover import greedy as maxcover_heuristic
from greedy_maxcut import greedy as maxcut_heuristic
from imm import imm

from mcts_progressive_widening import MCTS_PROGRESSIVE

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--problem",type=str,default='MaxCover')
    parser.add_argument("--budget",type=int,default=100)
    parser.add_argument("--depth",type=int,default=150)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolicyValueGCN()
    dataset = args.dataset
    budget = args.budget
    depth = args.depth
    problem = args.problem

    print(f'Solving {problem} for {dataset} Budget {budget} Depth {depth}')

    save_folder = f'pretrained/{problem}/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'latest.pth')

    model.load_state_dict(torch.load(save_file_path,weights_only=False))
    

    args = {
        'batch_size': 10,
        'numIters': 1,                                # Total number of training iterations
        'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
        'numEps': 1,                                  # Number of full games (episodes) to run during each iteration
        'numItersForTrainExamplesHistory': 20,
        'epochs': 10,                                    # Number of epochs of training per iteration
        'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
    }


    test_graph = load_graph(f'../data/test/{dataset}')

    if problem == 'MaxCover':
        heuristic = maxcover_heuristic
        env = MaxCover

    elif problem == 'MaxCut':
        heuristic = maxcut_heuristic
        env = MaxCut

    elif problem == 'IM':
        heuristic = imm
        env = IM

    else:
        raise ValueError('Unknown Problem')

    game  = env(graph=test_graph,heuristic=heuristic,
                         budget=budget,depth=depth,
                         GNNpruner=None,
                         train=False)



    start = time.time()
   
    mcts = MCTS_PROGRESSIVE(game=game,
                            model=model,
                            k=0.5,
                            args=args
                            )

    






    root=mcts.run(model=model,state=game.get_init_state())


    node = root
    pruned_universe  = []
    for i in range(depth):

        
        # if node.expanded():
        if len(node.children)>0:
            # print(np.sum(node.state))

            max_visit_count = 0
            next_node = None
            best_action = None
            for action in node.children:
                child = node.children[action]
                if child.visit_count>= max_visit_count:
                    max_visit_count = child.visit_count
                    
                    next_node = child
                    best_action = action 
            pruned_universe.append(best_action)
            node = next_node


    
    end= time.time()

    time_to_prune = end-start

    print('time elapsed to pruned',time_to_prune)


    print([test_graph.degree(node) for node in pruned_universe])
    Pg = len(pruned_universe)/test_graph.number_of_nodes()
    start = time.time()
    objective_unpruned, solution_unpruned, queries_unpruned = heuristic(test_graph,budget)
    end = time.time()
    time_unpruned = round(end-start,4)
    print('Elapsed time (unpruned):',round(time_unpruned,4))

    start = time.time()
    objective_pruned,solution_pruned, queries_pruned = heuristic(graph=test_graph,
                                                                 budget=budget,
                                                                 ground_set=pruned_universe)
    end = time.time()
    time_pruned = round(end-start,4)
    print('Elapsed time (pruned):',time_pruned)
    
    
    ratio = objective_pruned/objective_unpruned


    print('Performance of MCTSPruner')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
    print('Pg(%):', round(Pg,4)*100)
    print('Ratio:',round(ratio,4)*100)
    # print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


    save_folder = f'{problem}/data/{dataset}'
    os.makedirs(save_folder,exist_ok=True)
    save_file_path = os.path.join(save_folder,'MCTSPruner')

    

    df ={     'Dataset':dataset,
              'Budget':budget,
              'Objective Value(Unpruned)':objective_unpruned,
              'Objective Value(Pruned)':objective_pruned ,
              'Ground Set': test_graph.number_of_nodes(),
              'Ground set(Pruned)':len(pruned_universe), 
            #   'Queries(Unpruned)': queries_unpruned,
              'Time(Unpruned)':time_unpruned,
              'Time(Pruned)': time_pruned,
            #   'Queries(Pruned)': queries_pruned, 
              'Pruned Ground set(%)': round(Pg,4)*100,
              'Ratio(%)':round(ratio,4)*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
              'TimeRatio': time_pruned/time_unpruned,
              'TimeToPrune':time_to_prune,
              


              }

   
    df = pd.DataFrame(df,index=[0])
    save_to_pickle(df,save_file_path)
   

