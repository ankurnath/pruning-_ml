import numpy as np
from greedy import greedy
# from utils import make_subgraph 
from utils import *




class Game:
    def __init__(self,graph,heuristic,budget,depth,GNNpruner = None,train=True):



        self.budget = budget
        self.heuristic = heuristic
        self.depth = depth
        


        if GNNpruner:
            self.action_mask = GNNpruner.test(test_graph=graph)
            # print('Action mask length:',len(self.action_mask),self.action_mask)

            subgraph = make_subgraph(graph=graph,nodes=self.action_mask)

            relabeled_subgraph,forward_mapping,reverse_mapping = relabel_graph(graph=subgraph)
            # print(relabeled_subgraph.number_of_nodes())
            self.graph = relabeled_subgraph
            self.foward_mapping = forward_mapping
            self.action_mask =[ forward_mapping[action] for action in self.action_mask]
            self.reverse_mapping = reverse_mapping
            # print(self.reverse_mapping)         
        else:
            self.graph = graph
            if train:
                _,self.action_mask,_ = heuristic(graph=graph,budget=budget)
            else:
                self.action_mask = [node for node in graph.nodes()]
            
        _action_mask = set(self.action_mask)
        self.action_demask = [node for node in self.graph.nodes() if node 
                                not in set(_action_mask)]

    def get_init_state(self):

        state = np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.float32) * -1

        state [self.action_mask] = 1
        return state
    

    def get_action_size(self):

        return self.graph.number_of_nodes()
    

    def get_next_state(self, state, action):
       

        new_state = state.copy()
        new_state[action] = 0

        return new_state
    
    def has_legal_moves(self, state):

        # valid_moves = (state == 1)
        return  np.sum(state == 1)>0 and self.depth> np.sum(state==0)
        # return self.depth-(self.graph.number_of_nodes()-np.sum(state)-len(self.action_mask)) > 0

        
    
    def get_valid_moves(self, state):

        return state == 1
    

    def get_reward_for_player(self, state):

        if self.has_legal_moves(state):

            return None
        
        nodes = np.where(state == 0)[0]
        subgraph = make_subgraph(graph=self.graph,nodes=nodes)
        reward,_,_ = self.heuristic(graph=subgraph,budget=self.budget)
        return reward/self.max_reward


        
    
class MaxCover(Game):
    def __init__(self, graph, heuristic, budget, depth, GNNpruner,train):
        # Properly call the parent class's initializer using `super()`
        super().__init__(graph, heuristic, budget, depth, GNNpruner,train)
        
        # Correctly access the `max_reward` attribute from the `graph` object
        self.max_reward = self.graph.number_of_nodes()

        

    



    
    # def has_legal_moves(self, board):
    #     for index in range(self.columns):
    #         if board[index] == 0:
    #             return True
    #     return False
    

    

