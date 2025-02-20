from utils import *

from model import PolicyValueGCN

from updated_game import MaxCover,MaxCut,IM

from greedy_maxcover import greedy as maxcover_heuristic
# from greedy_original_maxcover import greedy_maxcover
from greedy_maxcut import greedy as maxcut_heuristic
from imm import imm
from gnnpruner_train import *

from mcts_progressive_widening import MCTS_PROGRESSIVE

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument( "--dataset", type=str, default='Twitter', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument("--problem",type=str,default='MaxCut')
    parser.add_argument("--budget",type=int,default=100)
    parser.add_argument("--depth",type=int,default=150)
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    parser.add_argument("--gnnpruner", type=bool,default=True, help="Whether to use GNNpruner to pre-prune")
    parser.add_argument("--train_dist", type=str,default="ER_200", help="cuda device")
    
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        # print("CUDA Device {}: {}".format(i, device_name))

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
    pre_prune = args.gnnpruner

    train_dist = args.train_dist

    if train_dist is None:
        train_dist = dataset
    else:
        root_folder = f'generelization/{train_dist}/{problem}/data/{dataset}'
        os.makedirs(root_folder,exist_ok=True)

    print(f'Solving {problem} for {dataset} Budget {budget} Depth {depth} with train_dist {train_dist}')

    # save_folder = f'pretrained/{problem}/{dataset}'
    save_folder = f'pretrained/{problem}/{train_dist}'
    os.makedirs(save_folder,exist_ok=True)
    model = PolicyValueGCN()
    save_file_path = os.path.join(save_folder,'best.pth')
    model.load_state_dict(torch.load(save_file_path,weights_only=True))
    

    args = {
        'batch_size': 10,
        'numIters': 1,                                # Total number of training iterations
        'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
        'numEps': 1,                                  # Number of full games (episodes) to run during each iteration
        'numItersForTrainExamplesHistory': 20,
        'epochs': 1,                                    # Number of epochs of training per iteration
        'checkpoint_path': 'best.pth'                 # location to save latest set of weights
    }


    test_graph = load_graph(f'../snap_dataset/test/{dataset}')

    if problem == 'MaxCover':
        heuristic = maxcover_heuristic
        # heuristic = greedy_maxcover
        env = MaxCover

    elif problem == 'MaxCut':
        heuristic = maxcut_heuristic
        env = MaxCut

    elif problem == 'IM':
        heuristic = imm
        env = IM

    else:
        raise ValueError('Unknown Problem')
    
    START = time.time()


    #### GNN only

    pruner = GNNpruner()
    # save_folder =  f'pretrained/{problem}/GNNpruner/{dataset}'
    save_folder =  f'pretrained/{problem}/GNNpruner/{train_dist}'
    load_model_path = os.path.join(save_folder,'best_model.pth')
    pruner.model.load_state_dict(torch.load(load_model_path,weights_only=False))
    pruner.model.to(device)
    pruned_universe_gnn = pruner.test(test_graph)
    Pg_gnn = len(pruned_universe_gnn)/test_graph.number_of_nodes()
    start = time.time()
    objective_unpruned, solution_unpruned, queries_unpruned = heuristic(test_graph,budget)
    end = time.time()

    time_unpruned = round(end-start,4)
    start = time.time()
    objective_pruned,solution_pruned, queries_pruned = heuristic(graph=test_graph,
                                                                budget=budget,
                                                                ground_set=pruned_universe_gnn)
    end = time.time()

    time_pruned = round(end-start,4)
    ratio_gnn = objective_pruned/objective_unpruned


    

    print('Performance of GCNPruner')
    print('Size Constraint,k:',budget)
    print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
    print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe_gnn))

    print('Pg(%):', round(1-Pg_gnn,4)*100)
    print('Ratio:',round(ratio_gnn,4)*100)
    print('C',round((1-Pg_gnn)*ratio_gnn,4)*100)
    # print('Pg(%):', Pg_gnn*100)
    # print('Ratio:',ratio_gnn*100)
    # print('C',Pg_gnn*ratio_gnn)

    

    

    

    df_gnn ={   
                'Dataset':dataset,
                'Budget':budget,
                'Objective Value(Unpruned)':objective_unpruned,
                'Objective Value(Pruned)':objective_pruned ,
                'Ground Set': test_graph.number_of_nodes(),
                'Ground set(Pruned)':len(pruned_universe_gnn), 
            #   'Queries(Unpruned)': queries_unpruned,
                'Time(Unpruned)':time_unpruned,
                'Time(Pruned)': time_pruned,
                'Pruned Ground set(%)': Pg_gnn*100,
            #   'Queries(Pruned)': queries_pruned, 
            #   'Pruned Ground set(%)': round(Pg,4)*100,
                'Ratio(%)':ratio_gnn*100, 
            #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
                'TimeRatio': time_pruned/time_unpruned,
            #   'TimeToPrune':time_to_prune,
                'Speedup':time_unpruned/time_pruned

            }
    df_gnn = pd.DataFrame(df_gnn,index=[0])

    

    if train_dist == dataset:
        save_file_path = os.path.join(f'{problem}/data/{dataset}','GNNPruner')
        save_to_pickle(df_gnn,save_file_path)
    else:
        # save_file_path = os.path.join(f'{problem}/data/{dataset}','GNNPruner')
        save_file_path = os.path.join(root_folder,'GNNPruner')
        save_to_pickle(df_gnn,save_file_path)

    print('*'*10)
# else:
    
#     k = 1000
#     data = from_networkx(test_graph)
#     data.x = torch.ones((test_graph.number_of_nodes(),1))

    
#     data = Batch.from_data_list([data])
#     actions_prob,value = model(data)
#     actions_prob = actions_prob.reshape(actions_prob.shape[0],)

#     top_k_actions = torch.topk(actions_prob,k=k).indices.numpy()

    
#     # print('TOP-K actions')
#     # print([test_graph.degree(node) for node in top_k_actions[:100]])
#     # print(test_graph.subgraph(top_k_actions).number_of_nodes())
#     # print('High degree nodes')
#     # print(sorted([test_graph.degree(node) for node in test_graph.nodes()])[::-1][:100])
#     subgraph = make_subgraph(test_graph,top_k_actions)
#     relabel_subgraph,_,reverse_transformation=relabel_graph(subgraph)


# if pre_prune: 
#     game  = env(graph=test_graph,
#                 heuristic=heuristic,
#                 budget=budget,
#                 depth=depth,
#                 GNNpruner=pruner,
#                 train=False)
# else:

subgraph = make_subgraph(test_graph,pruned_universe_gnn)
relabel_subgraph,forward_transformation,reverse_transformation=relabel_graph(subgraph)

relabeled_pruned_universe_gnn = [forward_transformation[node] for node in pruned_universe_gnn]

game  = env(graph=relabel_subgraph,
            heuristic=heuristic,
            pruned_universe = None,
            # pruned_universe = relabeled_pruned_universe_gnn,
            budget = budget,
            depth = depth,
            # GNNpruner =None,
            train =False)


print('Action mask length in MCTS:',len(game.action_mask))  
end = time.time()

print('Time elpased to create the game',round(end-start,4))

time_to_create_game = end-start

start = time.time()

mcts = MCTS_PROGRESSIVE(game=game,
                        model=model,
                        k=0.1,
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
    else:
        break



end= time.time()

time_to_prune = end-start

# print('time elapsed to pruned',time_to_prune)


# # print([test_graph.degree(node) for node in pruned_universe])
# print(pruned_universe)

# if pre_prune:
#     pruned_universe = [game.reverse_mapping[node] for node in pruned_universe]
#     # print(sorted(pruned_universe))
# else:
pruned_universe = [reverse_transformation[node] for node in pruned_universe]

# for node in pruned_universe:
#     if node not in pruned_universe_gnn:
#         raise ValueError('Node not in GNN pruned universe')
# pruned_universe = top_k_actions
# print([test_graph.degree(node) for node in pruned_universe])

END = time.time()

total_time_to_prune = END-START

print("Total time to prune",total_time_to_prune)

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

print('*'*10)
print('Performance of MCTSPruner')
print('Size Constraint,k:',budget)
print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
print('Pg(%):', round(1-Pg,4)*100)
print('Ratio:',round(ratio,4)*100)
print('C',round((1-Pg)*ratio,4)*100)

# print('Queries:',round(queries_pruned/queries_unpruned,4)*100)


save_folder = f'{problem}/data/{dataset}'
os.makedirs(save_folder,exist_ok=True)

# if pre_prune:
#     save_file_path = os.path.join(save_folder,'MCTSPruner+GNNPruner')

# else:
#     save_file_path = os.path.join(save_folder,'MCTSPruner')



df ={       'Dataset':dataset,
            'Budget':budget,
            'Objective Value(Unpruned)':objective_unpruned,
            'Objective Value(Pruned)':objective_pruned ,
            'Ground Set': test_graph.number_of_nodes(),
            'Ground set(Pruned)':len(pruned_universe), 
        #   'Queries(Unpruned)': queries_unpruned,
            'Time(Unpruned)':time_unpruned,
            'Time(Pruned)': time_pruned,
            'Pruned Ground set(%)': Pg*100,
        #   'Queries(Pruned)': queries_pruned, 
        #   'Pruned Ground set(%)': round(Pg,4)*100,
            'Ratio(%)':ratio*100, 
        #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
            'TimeRatio': time_pruned/time_unpruned,
            'TimeToPrune':time_to_prune,
            'TimeToCreateGame':time_to_create_game,
            'TotalTimeToPrune':total_time_to_prune,
            'Speedup':time_unpruned/time_pruned
            
            


            }


df = pd.DataFrame(df,index=[0])


if train_dist == dataset:
    # save_file_path = os.path.join(f'{problem}/data/{dataset}','GNNPruner')
    save_file_path = os.path.join(save_folder,'MCTSPruner+GNNPruner')
    save_to_pickle(df,save_file_path)
else:
    # save_file_path = os.path.join(f'{problem}/data/{dataset}','GNNPruner')
    save_file_path = os.path.join(root_folder,'MCTSPruner+GNNPruner')
    save_to_pickle(df,save_file_path)

print('*'*10)



#### Guided MCTS
game  = env(graph=relabel_subgraph,
            heuristic=heuristic,
            # pruned_universe = None,
            pruned_universe = relabeled_pruned_universe_gnn,
            budget = budget,
            depth = depth,
            # GNNpruner =None,
            train =False)


print('Action mask length in MCTS:',len(game.action_mask))  
end = time.time()

print('Time elpased to create the game',round(end-start,4))

time_to_create_game = end-start

start = time.time()

mcts = MCTS_PROGRESSIVE(game=game,
                        model=model,
                        k=0.1,
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
    else:
        break



end= time.time()

time_to_prune = end-start

# print('time elapsed to pruned',time_to_prune)


# # print([test_graph.degree(node) for node in pruned_universe])
# print(pruned_universe)

# if pre_prune:
#     pruned_universe = [game.reverse_mapping[node] for node in pruned_universe]
#     # print(sorted(pruned_universe))
# else:
pruned_universe = [reverse_transformation[node] for node in pruned_universe]

# for node in pruned_universe:
#     if node not in pruned_universe_gnn:
#         raise ValueError('Node not in GNN pruned universe')
# pruned_universe = top_k_actions
# print([test_graph.degree(node) for node in pruned_universe])

END = time.time()

total_time_to_prune = END-START

print("Total time to prune",total_time_to_prune)

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

print('*'*10)
print('Performance of MCTSPruner+GNNPruner+GuidedMCTS')
print('Size Constraint,k:',budget)
print('Size of Ground Set,|U|:',test_graph.number_of_nodes())
print('Size of Pruned Ground Set, |Upruned|:', len(pruned_universe))
print('Pg(%):', round(1-Pg,4)*100)
print('Ratio:',round(ratio,4)*100)
print('C',round((1-Pg)*ratio,4)*100)
# print('Queries:',round(queries_pruned/queries_unpruned,4)*100)
# print('*'*10)

save_folder = f'{problem}/data/{dataset}'
os.makedirs(save_folder,exist_ok=True)

# if pre_prune:
#     save_file_path = os.path.join(save_folder,'MCTSPruner+GNNPruner')

# else:
#     save_file_path = os.path.join(save_folder,'MCTSPruner')



df ={     'Dataset':dataset,
            'Budget':budget,
            'Objective Value(Unpruned)':objective_unpruned,
            'Objective Value(Pruned)':objective_pruned ,
            'Ground Set': test_graph.number_of_nodes(),
            'Ground set(Pruned)':len(pruned_universe), 
        #   'Queries(Unpruned)': queries_unpruned,
            'Time(Unpruned)':time_unpruned,
            'Time(Pruned)': time_pruned,
            'Pruned Ground set(%)': Pg*100,
        #   'Queries(Pruned)': queries_pruned, 
        #   'Pruned Ground set(%)': round(Pg,4)*100,
            'Ratio(%)':ratio*100, 
        #   'Queries(%)': round(queries_pruned/queries_unpruned,4)*100,
            'TimeRatio': time_pruned/time_unpruned,
            'TimeToPrune':time_to_prune,
            'TimeToCreateGame':time_to_create_game,
            'TotalTimeToPrune':total_time_to_prune,
            'Speedup':time_unpruned/time_pruned
            
            


            }


df = pd.DataFrame(df,index=[0])
# save_file_path = os.path.join(save_folder,'MCTSPruner+GNNPruner+GuidedMCTS')

if train_dist == dataset:
    # save_file_path = os.path.join(f'{problem}/data/{dataset}','GNNPruner')
    save_file_path = os.path.join(save_folder,'MCTSPruner+GNNPruner+GuidedMCTS')
    save_to_pickle(df,save_file_path)
else:
    # save_file_path = os.path.join(f'{problem}/data/{dataset}','GNNPruner')
    save_file_path = os.path.join(root_folder,'MCTSPruner+GNNPruner+GuidedMCTS')
    save_to_pickle(df,save_file_path)

print('*'*10)
# save_to_pickle(df,save_file_path)



# Multi-Budget analysis
# budgets = [1,10,25,50,75,100]
# save_folder = f'{problem}_multibudget/data/{dataset}'
# os.makedirs(save_folder,exist_ok=True)

# ratios = []
# for budget in budgets:
#     print('Solving for budget:',budget)
    
#     objective_unpruned, solution_unpruned, queries_unpruned = heuristic(test_graph,budget)
    
#     objective_pruned,solution_pruned, queries_pruned = heuristic(graph=test_graph,
#                                                                 budget=budget,
#                                                                 ground_set=pruned_universe)
    

    
    
#     ratio = objective_pruned/objective_unpruned

#     ratios.append(ratio)


#     print('Performance of MCTSPruner')
#     print('Ratio:',round(ratio,4)*100)


# save_file_path = os.path.join(save_folder,f'MCTSPruner')


# df = {     'Dataset':[dataset]*len(budgets),
#     'budegt':budgets,
#     'Ratio':ratios
#     }

# df = pd.DataFrame(df)
# print(df)

# save_to_pickle(df,save_file_path)

        


        
        
   

