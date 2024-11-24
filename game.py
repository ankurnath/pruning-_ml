import numpy as np
from greedy import greedy
from utils import make_subgraph


class MaxCover:


    def __init__(self,graph,budget,depth):
        self.graph = graph
        self.budget = budget
        self.max_reward = graph.number_of_nodes()
        self.depth = depth
        # pass

        # self.graph = graph
        # pass

    # def get_init_state(self):

    #     return np.ones(shape=(self.graph.num_nodes()))

    def get_init_state(self):
        return np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.float32)
    

    def get_action_size(self):

        return self.graph.number_of_nodes()
    

    def get_next_state(self, state, action):
        # b = np.copy(board)
        # b[action] = player

        new_state = state.copy()
        new_state[action] = 0

        # Return the new game, but
        # change the perspective of the game with negative
        # return (b, -player)

        return new_state
    
    def has_legal_moves(self, state):
        return self.depth-(self.graph.number_of_nodes()-np.sum(state)) >0

        # return self.budget-(self.graph.number_of_nodes()-np.sum(state)) >0 
    
    def get_valid_moves(self, state):

        return state == 1
    

    def get_reward_for_player(self, state):

        if self.has_legal_moves(state):

            return None
        
        # reward,_,_ = greedy(graph=self.graph,budget=self.budget)

        # nodes = list(np.where(state == 0)[0])
        nodes = list(np.where(state == 0)[0])

        # print(nodes)
        # print(len(nodes))
        subgraph = make_subgraph(graph=self.graph,nodes=nodes)
        reward,_,_ = greedy(graph=subgraph,budget=self.budget)
        return reward/self.max_reward


        # covered_elements=set()
        # for node in range(self.graph.number_of_nodes()):
        #     if state[node] == 0:
        #         covered_elements.add(node)
        #         for neighbour in self.graph.neighbors(node):
        #             covered_elements.add(neighbour)
        
        # return len(covered_elements)/self.max_reward






        return reward
    


        

    



    
    # def has_legal_moves(self, board):
    #     for index in range(self.columns):
    #         if board[index] == 0:
    #             return True
    #     return False
    

    

