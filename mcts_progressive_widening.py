##### Training for MaxCover

import networkx as nx
import numpy as np
from greedy_maxcover import greedy
from torch_geometric.utils import from_networkx
import torch
from utils import *
import heapq


# Implementing for MaxCover (Upper Confidence Bounds)
def ucb_score(parent,child):
    prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
    return  child.value() + prior_score 
    





class Node:

    def __init__(self, prior):
        
        self.visit_count = 0
        
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self,k):
        return len(self.children) > np.floor((self.visit_count)**k)
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action
    

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score >= best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


    def expand(self, state = None,action_probs = None):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """

        if state is not None:
            self.state = state


        if action_probs is not None:
            # never been expanded before
            action_probs = action_probs.reshape(action_probs.shape[0],)
            self.action_probs = action_probs.reshape(action_probs.shape[0],)
            self.actions = np.argsort(-action_probs)
        
        best_action = self.actions[len(self.children)]
        best_action_prob = self.action_probs[len(self.children)]
        # action_probs = action_probs.reshape(action_probs.shape[0],)


        # best_action = None
        # best_action_prob = 0

        # for i in range(action_probs.shape[0]):

        #     if action_probs[i] > best_action_prob and i not in self.children:
        #         best_action = i
        #         best_action_prob = action_probs[i]

        # # if best

        # partition_index = np.argpartition(-action_probs,len(self.children)+1)
        # # print(partition_index)
        # print('Children',self.children.keys())
        # print(action_probs[partition_index[:len(self.children)+1]])
        # print('Options to explore',partition_index[:len(self.children)+1])
        # action = partition_index[len(self.children)]

        
        if best_action in self.children:
            raise ValueError('Already expanded')
        self.children[best_action] = Node(prior=best_action_prob)



    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())






class MCTS_PROGRESSIVE:

    def __init__(self, game, model,k, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = game
        self.model = model.to(self.device)
        self.args = args
        self.k = k

    def run(self, model, state):

        # print('Running MCTS')
        # print('Ground set size',state.sum())
        # print('State',state[50])

        # root = Node(0, to_play)
        root = Node(prior=0)

        data = from_networkx(self.game.graph)
        data.x = torch.from_numpy(state)
        data = Batch.from_data_list([data])
        data = data.to(self.device)

        # print(data.x.dtype)

        # EXPAND root
        # action_probs, value = model(state)
        action_probs, value = model(data)
        action_probs = action_probs.cpu().detach().numpy()
        
        valid_moves = self.game.get_valid_moves(state)
        # action_probs = action_probs * state  # mask invalid moves
        action_probs = action_probs * valid_moves  # mask invalid moves

        # print('Action probs',action_probs[50])
        # print('Max Action child',np.argmax(action_probs))
        # action_probs*=state
        action_probs /= np.sum(action_probs)
        # print(action_probs)

        # pr
        # root.expand(state,action_probs)
        root.expand(action_probs=action_probs)
        # print(root.children)
        # print(root.expanded(self.k))

        # print(self.args['num_simulations'])

        # for _ in tqdm(range(self.args['num_simulations'])):
        for simulation in range(self.args['num_simulations']):


        # for _ in range(1):
            node = root
            search_path = [node]


            actions =[]
            while len(node.children) > 0:
            # while node.expanded(self.k):

                ############

                if not node.expanded(self.k):
                    node.expand()


                ############
                action, node = node.select_child()
                actions.append(action)

                search_path.append(node)


            # parent = search_path[-2]
            # state = parent.state


            # next_state = self.game.get_next_state(state,action=action)
            state[actions]  = 0
            next_state = state

            if len(actions) == self.game.depth:
                value = self.game.get_reward_for_player(next_state)
                # print('Value',value)
            # value = self.game.get_reward_for_player(next_state)
            # else:
            #     value = None

            # # print('Value',value)
            # if value is None:
            if len(actions) < self.game.depth or value is None:
                # EXPAND

                ####
                # data = from_networkx(self.game.graph)
                data.x = torch.from_numpy(next_state)
                data = Batch.from_data_list([data])
                data = data.to(self.device)

                ####
                # action_probs, value = model(next_state)
                action_probs, value = model(data)
                action_probs = action_probs.cpu().detach().numpy()
                value = value.item()
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                # action_probs*=state
                action_probs /= np.sum(action_probs)
                # node.expand(next_state, parent.to_play * -1, action_probs)
                # node.expand(next_state,action_probs)
                node.expand(action_probs=action_probs)
                # print('Number of children of the last node of the search path',len(node.children))

            self.backpropagate(search_path, value)
            state[actions] = 1  # reset state
        return root

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value 
            node.visit_count += 1






