import networkx as nx
import numpy as np
from greedy import greedy



# Implementing for MaxCover (Upper Confidence Bounds)

class Node:

    def __init__(self, state, budget, size, parent=None):
        self.state = state
        self.parent = parent
        self.visit_count = 0
        self.total_value = 0
        self.budget = budget
        self.size = size
        self.children = []

    def expand(self):
        n = self.state.shape[0]
        for i in range(n):
            if self.state[i] == 0:
                new_state = self.state.copy()
                new_state[i] = 1
                new_node = Node(state=new_state, budget=self.budget, size=self.size+1, parent=self)
                self.children.append(new_node)

    def is_terminal(self):
        return self.size == self.budget 

def obj_val(graph, sol):
    # Calculate the objective value: number of covered nodes
    covered = np.zeros(graph.number_of_nodes())
    for node in range(len(sol)):
        if sol[node] == 1:
            covered[node] = 1
            for neighbor in graph.neighbors(node):
                covered[neighbor] = 1
    return np.sum(covered)

n = 100
budget = 5
num_iter = 10000
# graph = nx.erdos_renyi_graph(n=n, p=0.5)
graph = nx.barabasi_albert_graph(n=n,m=4)



root_node = Node(state=np.zeros(shape=(n,)), size=0, budget=budget)

for iter in range(1000):
    node = root_node

    # Selection phase (UCB selection)
    while not node.is_terminal():
        if len(node.children) == 0:
            # Expand node if it hasn't been expanded yet
            node.expand()

        # Select the best child based on UCB
        best_uct = -float('inf')
        best_child = None
        for child in node.children:
            if child.visit_count == 0:
                uct = float('inf')  # Ensure unexplored nodes are chosen first
            else:
                uct = child.total_value / child.visit_count + 2 * np.sqrt(np.log(iter + 1) / child.visit_count)
            if uct > best_uct:
                best_uct = uct
                best_child = child
        node = best_child

        if node.visit_count == 0:
            break

    # Rollout (simulation phase)
    if node.visit_count == 0:  # If we hit an unexplored node, rollout from here
        zero_indices = np.where(node.state == 0)[0]
        random_zero_indices = np.random.choice(zero_indices, size=budget-node.size, replace=False)
        sol = node.state.copy()
        for idx in random_zero_indices:
            sol[idx] = 1
        reward = obj_val(graph=graph, sol=sol)
    else:
        # reward = node.total_value / node.visit_count
        reward = obj_val(graph=graph, sol=node.state)

    # Backpropagation phase
    while node is not None:
        node.visit_count += 1
        node.total_value += reward
        node = node.parent


node = root_node

# Selection phase (UCB selection)
while not node.is_terminal() and node:
    # print(node.state)
    print(node.size)
    print(np.sum(node.state))
    

    # Select the best child based on UCB
    best_value = -float('inf')
    best_child = None
    for child in node.children:
       
        if  child.visit_count>0 and child.total_value/child.visit_count > best_value:
            best_value = child.total_value/child.visit_count
            best_child = child
    node = best_child

    
# print(node.state)

print(greedy(graph=graph,budget=budget))
print(node.state.sum())
print(obj_val(graph=graph,sol=node.state))

print(root_node.total_value/root_node.visit_count)





