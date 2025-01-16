import numpy as np
# from greedy_maxcover import greedy
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
            # self.graph = graph
            if train:
                _,self.action_mask,_ = heuristic(graph=graph,budget=budget)
                print('Action mask length:',len(self.action_mask))
                # print([graph.degree(node) for node in self.action_mask])
                # *****
                subgraph = make_subgraph(graph=graph,nodes=self.action_mask)
                # subgraph = graph.subgraph(self.action_mask).copy()
                # print('Subgraph:Node',subgraph.number_of_nodes())
                # print('Subgraph:Edges',subgraph.number_of_edges())
                relabeled_subgraph,forward_mapping,reverse_mapping = relabel_graph(graph=subgraph)
                self.graph = relabeled_subgraph
                self.action_mask =[ forward_mapping[action] for action in self.action_mask]
                self.reverse_mapping = reverse_mapping
                
                # *****
            else:
                self.graph = graph
                self.action_mask = [node for node in graph.nodes()]
        if train:
            _action_mask = set(self.action_mask)
            self.action_demask = [node for node in self.graph.nodes() if node 
                                    not in set(_action_mask)]
        else:
            self.action_demask = []

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
        # print('Nodes:',nodes)
        # print(len(nodes))
        # subgraph = make_subgraph(graph=self.graph,nodes=nodes)
        # reward,_,_ = self.heuristic(graph=subgraph,budget=self.budget)
        reward,_,_ = self.heuristic(graph=self.graph,
                                    budget=self.budget,
                                    ground_set=nodes)
        # print('Reward:',reward/self.max_reward)
        return reward/self.max_reward


        
    
class MaxCover(Game):
    def __init__(self, graph, heuristic, budget, depth, GNNpruner,train):
        # Properly call the parent class's initializer using `super()`
        super().__init__(graph, heuristic, budget, depth, GNNpruner,train)
        
        # Correctly access the `max_reward` attribute from the `graph` object
        self.max_reward = self.graph.number_of_nodes()


class MaxCut(Game):
    def __init__(self, graph, heuristic, budget, depth, GNNpruner,train):
        # Properly call the parent class's initializer using `super()`

        # Correctly access the `max_reward` attribute from the `graph` object
        
        super().__init__(graph, heuristic, budget, depth, GNNpruner,train)
        self.max_reward = graph.number_of_edges()
        
        




class IM(Game):
    def __init__(self, graph, heuristic, budget, depth, GNNpruner,train):


        self.budget = budget
        self.heuristic = heuristic
        self.depth = depth
        self.train = train
        self.graph = graph


        if GNNpruner:
            self.action_mask = GNNpruner.test(test_graph=graph)
            subgraph = make_subgraph(graph=graph,nodes=self.action_mask)
            relabeled_subgraph,forward_mapping,reverse_mapping = relabel_graph(graph=subgraph)
            self.graph = relabeled_subgraph
            self.action_mask =[ forward_mapping[action] for action in self.action_mask]
            self.reverse_mapping = reverse_mapping


            _,_,rr = heuristic(graph=relabeled_subgraph,budget=budget)
            self.rr = rr

            rr_degree = defaultdict(int)
            node_rr_set = defaultdict(list)
            
            for j,rr in enumerate(rr):
                for rr_node in rr:
                    rr_degree[rr_node]+=1
                    node_rr_set[rr_node].append(j)
            
            self.rr_degree= rr_degree
            self.node_rr_set = node_rr_set

        elif self.train:
        
            coverage,self.action_mask,rr = heuristic(graph=graph,budget=budget)
            subgraph = make_subgraph(graph=graph,nodes=self.action_mask)
            relabeled_subgraph,forward_mapping,reverse_mapping = relabel_graph(graph=subgraph)
            self.graph = relabeled_subgraph
            self.action_mask =[ forward_mapping[action] for action in self.action_mask]
            self.reverse_mapping = reverse_mapping


            print(np.max([graph.degree(node) for node in graph.nodes()]))
            print('Coverage:',coverage)
            print([self.graph.degree(node) for node in self.action_mask])

            self.rr = rr

            rr_degree = defaultdict(int)
            node_rr_set = defaultdict(list)
            
            for j,rr in enumerate(rr):
                for rr_node in rr:
                    rr_degree[rr_node]+=1
                    node_rr_set[rr_node].append(j)
            
            self.rr_degree= rr_degree
            self.node_rr_set = node_rr_set

        else:
            self.action_mask = [node for node in graph.nodes()]
        self.max_reward = graph.number_of_nodes()
            
        _action_mask = set(self.action_mask)
        if self.train:

            self.action_demask = [node for node in self.graph.nodes() if node 
                                    not in set(_action_mask)]
        else:
            self.action_demask = []


        
    def get_reward_for_player(self, state):

        if self.has_legal_moves(state) or not self.train:
            return None
        
        # print('Using this reward')
        nodes = np.where(state == 0)[0]
        gains = {node:self.rr_degree[node] for node in nodes}
        covered_rr_set = set()
        matched_count = 0
        for i in range(self.budget):
            max_point = max(gains,key=gains.get)
            
            matched_count +=gains[max_point]
            for index in self.node_rr_set[max_point]:
                if index not in covered_rr_set:
                    covered_rr_set.add(index)
                    for rr_node in self.rr[index]:
                        if rr_node in gains:
                            gains[rr_node]-=1
        return   matched_count / len(self.rr)/self.max_reward
            
    



    

    

    

