import numpy as np
from greedy import greedy
# from utils import make_subgraph 
from utils import *




class Game:


    def __init__(self,graph,heuristic,budget,depth,GNNpruner = None):



        self.budget = budget
        self.heuristic = heuristic
        self.depth = depth
        


        if GNNpruner:
            self.graph,self.action_mask = GNNpruner.test(graph=graph,budget=budget)

            
        else:
            self.graph = graph
            self.action_mask = []

    def get_init_state(self):

        state = np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.float32)

        state [self.action_mask] = -1
        return state
    

    def get_action_size(self):

        return self.graph.number_of_nodes()
    

    def get_next_state(self, state, action):
       

        new_state = state.copy()
        new_state[action] = 0

        return new_state
    
    def has_legal_moves(self, state):

        return self.depth-(self.graph.number_of_nodes()-np.sum(state)-len(self.action_mask)) > 0

        
    
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
    def __init__(self, graph, heuristic, budget, depth, use_gnn_mask=False):
        # Properly call the parent class's initializer using `super()`
        super().__init__(graph, heuristic, budget, depth, use_gnn_mask)
        
        # Correctly access the `max_reward` attribute from the `graph` object
        self.max_reward = self.graph.max_reward

        

    



    
    # def has_legal_moves(self, board):
    #     for index in range(self.columns):
    #         if board[index] == 0:
    #             return True
    #     return False
    

    

