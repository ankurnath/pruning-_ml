from utils import *

from model import PolicyValueGCN


from game import MaxCover
from mcts_maxcover import MCTS
from greedy import *


model = PolicyValueGCN()

model.load_state_dict(torch.load('latest.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

graph = nx.erdos_renyi_graph(n=1000,p=0.05)
budget = 5
game = MaxCover(graph=graph,budget=budget)




args = {
    'batch_size': 10,
    'numIters': 10,                                # Total number of training iterations
    'num_simulations': 10,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 20,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 10,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}
mcts=MCTS(game=game,model=model,args=args)

# state = game.get_init_state()

# # cnt = 0
# actions = []
# while state is not None:
#     print('Actions:',actions)
#     node=mcts.run(model=model,state=state)

#     if node.expanded():
#         # print(np.sum(node.state))

#         max_visit_count = 0
#         next_node = None
#         best_action = None
#         for action in node.children:
#             child = node.children[action]
#             if child.visit_count>= max_visit_count:
#                 max_visit_count = child.visit_count
                
#                 next_node = child
#                 best_action = action 
#         actions.append(best_action)
#         node = next_node
#         state = next_node.state

#     else:
#         break




root=mcts.run(model=model,state=game.get_init_state())


node = root
actions  = []
for i in range(budget):

    
    if node.expanded():
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
        actions.append(best_action)
        node = next_node


print(node)
print(actions)
print(calculate_obj(graph=graph,solution=actions))
print(calculate_obj(graph=graph,solution=[0,1,2,3,4]))
print(greedy(graph=graph,budget=budget))


