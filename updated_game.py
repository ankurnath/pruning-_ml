import numpy as np
# from greedy_maxcover import greedy
# from utils import make_subgraph 
from utils import *




class Game:
    def __init__(self,
                 graph,
                 heuristic,
                 budget,
                 depth,
                 pruned_universe=None,
                 pre_prune=False,
                #  GNNpruner = None,
                 train=False,
                 greedy_rollout_func=None
                 ):



        self.budget = budget
        self.heuristic = heuristic
        self.depth = depth
        self.greedy_rollout_func = greedy_rollout_func
        self.train = train
        

        if train:
            print('Training')
            ####### Rollout

            self.graph = graph
            self.action_mask = [node for node in graph.nodes()]

            ######


            ##### Warm start 
            # _,self.action_mask,_ = heuristic(graph=graph,budget=budget)
            # print('Action mask length:',len(self.action_mask))
            
            # # *****
            # subgraph = make_subgraph(graph=graph,nodes=self.action_mask)
            # relabeled_subgraph,forward_mapping,reverse_mapping = relabel_graph(graph=subgraph)
            # self.graph = relabeled_subgraph
            # self.action_mask =[ forward_mapping[action] for action in self.action_mask]
            # self.reverse_mapping = reverse_mapping
            ##### 
        elif pruned_universe is not None:
            print('Pruned universe from GNN is given')
            self.graph = graph
            # self.action_mask = [node for node in graph.nodes()]
            self.action_mask = pruned_universe

        else:
            self.graph = graph
            self.action_mask = [node for node in graph.nodes()]
        
        if train or pruned_universe is not None:

            all_nodes = np.arange(self.graph.number_of_nodes())  # All node indices
            self.action_demask = np.setdiff1d(all_nodes, self.action_mask, assume_unique=True)


            # action_demask = np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.int32)
            # action_demask[self.action_mask] = 0
            # self.action_demask = np.where(action_demask == 1)[0]
            # _action_mask = set(self.action_mask)
            # self.action_demask = [node for node in self.graph.nodes() if node 
            #                             not in set(_action_mask)]
        else:
            self.action_demask = []

    def get_init_state(self):
        state = np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.float32)
        # state = np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.float32) * -1
        # state = np.ones(shape=(self.graph.number_of_nodes(),1),dtype=np.float32) * -1

        # state [self.action_mask] = 1
        return state
    

    def get_action_size(self):

        return self.graph.number_of_nodes()
    

    def get_next_state(self, state, action):
       

        new_state = state.copy()
        new_state[action] = 0

        return new_state
    
    def has_legal_moves(self, state):

        # valid_moves = (state == 1)
        # return  np.sum(state == 1)>0 and self.depth> np.sum(state==0)
        return  np.sum(state == 1)>0 and self.depth> np.sum(state==0)
        # return self.depth-(self.graph.number_of_nodes()-np.sum(state)-len(self.action_mask)) > 0

        
    
    def get_valid_moves(self, state):

        valid_moves = (state == 1)
        valid_moves[self.action_demask] = 0

        return valid_moves
    

    # def get_reward_for_player(self, state, threshold =0.0000001,best_action=None):
    def get_reward_for_player(self, state):

        # if self.has_legal_moves(state):


        #     # if self.greedy_rollout_func and self.train:
        #     #    return self.greedy_rollout_func(graph=self.graph, 
        #     #                               depth=self.depth, 
        #     #                               nodes=np.where(state == 0)[0])
                
        #     # else:

        #     return None

        
        
        nodes = np.where(state == 0)[0]
        # print('Nodes:',nodes)
        # print(len(nodes))
        # subgraph = make_subgraph(graph=self.graph,nodes=nodes)
        # reward,_,_ = self.heuristic(graph=subgraph,budget=self.budget)

        
        

        reward,_,_ = self.heuristic(graph=self.graph,
                                    budget=self.budget,
                                    ground_set=nodes)
        

        # nodes = list(nodes)
        # nodes.append(best_action)
        # # print('Nodes:',nodes)


        # new_reward,_,_ = self.heuristic(graph=self.graph,
        #                             budget=self.budget,
        #                             ground_set=nodes)
        
        # delta = new_reward - reward

        # # print('Delta:',delta)

        # if delta >= reward * threshold:

        #     self.depth = self.depth + 50 
        #     return None


        # print('Reward:',reward)
        return reward/self.max_reward


        
    
class MaxCover(Game):
    def __init__(self, 
                 graph, 
                 heuristic, 
                 budget, 
                 depth, 
                #  GNNpruner,
                 train,
                 pruned_universe,
                 greedy_rollout_func = None
                 ):
        # Properly call the parent class's initializer using `super()`
        super().__init__(graph=graph, 
                         heuristic= heuristic, 
                         budget=budget, 
                         depth=depth, 
                        #  GNNpruner=GNNpruner,
                         train=train,
                         pruned_universe=pruned_universe,
                         greedy_rollout_func=greedy_rollout_func
                         )
        
        # Correctly access the `max_reward` attribute from the `graph` object
        self.max_reward = self.graph.number_of_nodes()


class MaxCut(Game):
    def __init__(self, 
                 graph, 
                 heuristic, 
                 budget, 
                 depth, 
                #  GNNpruner,
                 train,
                 pruned_universe,
                 greedy_rollout_func = None
                 ):
        # Properly call the parent class's initializer using `super()`

        # Correctly access the `max_reward` attribute from the `graph` object
        
        super().__init__(graph=graph, 
                         heuristic=heuristic, 
                         budget=budget, 
                         depth=depth, 
                        #  GNNpruner = GNNpruner,
                         pruned_universe= pruned_universe,
                         train=train,
                         greedy_rollout_func=greedy_rollout_func,
                        #  greedy_rollout=greedy_rollout
                         )
        self.max_reward = graph.number_of_edges()
        
        




class IM(Game):
    def __init__(self, 
                 graph, 
                 heuristic, 
                 budget, 
                 depth, 
                #  GNNpruner,
                 train,
                 pruned_universe):


        self.budget = budget
        self.heuristic = heuristic
        self.depth = depth
        self.train = train
        # self.graph = graph


        # if GNNpruner:
        #     self.action_mask = GNNpruner.test(test_graph=graph)
        #     subgraph = make_subgraph(graph=graph,nodes=self.action_mask)
        #     relabeled_subgraph,forward_mapping,reverse_mapping = relabel_graph(graph=subgraph)
        #     self.graph = relabeled_subgraph
        #     self.action_mask =[ forward_mapping[action] for action in self.action_mask]
        #     self.reverse_mapping = reverse_mapping


        #     _,_,rr = heuristic(graph=relabeled_subgraph,budget=budget)
        #     self.rr = rr

        #     rr_degree = defaultdict(int)
        #     node_rr_set = defaultdict(list)
            
        #     for j,rr in enumerate(rr):
        #         for rr_node in rr:
        #             rr_degree[rr_node]+=1
        #             node_rr_set[rr_node].append(j)
            
        #     self.rr_degree= rr_degree
        #     self.node_rr_set = node_rr_set

        if self.train:
        
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

        elif pruned_universe is not None:
            print('Pruned universe from GNN is given')
            self.graph = graph
            # self.action_mask = [node for node in graph.nodes()]
            self.action_mask = pruned_universe


        else:
            self.graph = graph
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
            
    



    

    

    

