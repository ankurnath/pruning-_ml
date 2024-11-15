##### Training for MaxCover

import networkx as nx
import numpy as np
from greedy import greedy
from torch_geometric.utils import from_networkx
import torch
from utils import *



# Implementing for MaxCover (Upper Confidence Bounds)
def ucb_score(parent,child):

    prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)


    # if child.visit_count > 0:
    #     # The value of the child is from the perspective of the opposing player
    #     value_score = child.value()
    # else:
    #     value_score = 0

    #     ## if visit count is zero

    return  child.value() + prior_score 





class Node:

    def __init__(self, prior):
        
        self.visit_count = 0
        
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0
    
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
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


    def expand(self, state,action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.state = state
        # for a, prob in enumerate(action_probs):

        # print(action_probs.shape)

        action_probs = action_probs.reshape(action_probs.shape[0],)

        for action in range(action_probs.shape[0]):

            # print(prob)
            # print(action_probs[action])
            # print
            if action_probs[action]!= 0:
                self.children[action] = Node(prior=action_probs[action])

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())






class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, model, state):

        # root = Node(0, to_play)
        root = Node(prior=0)

        data = from_networkx(self.game.graph)
        data.x = torch.from_numpy(state)
        data = Batch.from_data_list([data])

        # print(data.x.dtype)

        # EXPAND root
        # action_probs, value = model(state)
        action_probs, value = model(data)
        action_probs = action_probs.detach().numpy()
        
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        # print(action_probs)

        # pr
        root.expand(state,action_probs)
        # print(len(root.children))

        # print(self.args['num_simulations'])

        for _ in range(self.args['num_simulations']):

        # for _ in range(200):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            # print(search_path)

            # print('Length of the search path',len(search_path))

            parent = search_path[-2]
            state = parent.state

            # print(state)
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state = self.game.get_next_state(state,action=action)
            # print(next_state)
            # Get the board from the perspective of the other player
            # next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state)

            # print('Value',value)
            if value is None:
                
                # EXPAND

                ####
                data = from_networkx(self.game.graph)
                data.x = torch.from_numpy(next_state)
                data = Batch.from_data_list([data])

                ####
                # action_probs, value = model(next_state)
                action_probs, value = model(data)
                action_probs = action_probs.detach().numpy()
                value = value.item()
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                # node.expand(next_state, parent.to_play * -1, action_probs)
                node.expand(next_state,action_probs)

            self.backpropagate(search_path, value)

        return root

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value 
            node.visit_count += 1








