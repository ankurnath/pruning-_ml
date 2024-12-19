from utils import *
from game import MaxCover
from model import PolicyValueGCN
from mcts_maxcover import MCTS

from greedy_maxcover import *


args = {
                            # Total number of training iterations
    'num_simulations': 1000,                         # Total number of MCTS simulations to run when deciding on a move to play
          # location to save latest set of weights
}

# graph = nx.barabasi_albert_graph(n=1000,m=4)

# graph = nx.barabasi_albert_graph(n=1000,m=4)
graph = nx.erdos_renyi_graph(n=100,p=0.1)

budget = 5
model = PolicyValueGCN()
game = MaxCover(graph=graph,budget=budget)



mcts = MCTS(game=game,model=model,args=args)

root=mcts.run(model=model,state=game.get_init_state())

# print(root.visit_count)

node = root
actions  = []
for i in range(budget):

    # print(i)
    if node.expanded():

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







