import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random



# MCTS Node
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

# MCTS with a neural network
class MCTS:
    def __init__(self, policy_value_net, c_puct=1.0):
        self.policy_value_net = policy_value_net
        self.c_puct = c_puct
        self.root = None

    def search(self, state):
        node = self.root

        # Selection
        while node.is_fully_expanded():
            action, node = self.select(node)
        
        # Expansion
        if not node.is_fully_expanded():
            self.expand(node)

        # Simulation (using the neural network for rollout)
        policy, value = self.policy_value_net(torch.tensor(state, dtype=torch.float32))
        
        # Backpropagation
        self.backpropagate(node, value.item())

    def select(self, node):
        best_value = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            uct_value = child.total_value / (1 + child.visit_count) + \
                        self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)

            if uct_value > best_value:
                best_value = uct_value
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, node):
        # Generate valid actions for the current state
        actions = self.get_valid_actions(node.state)
        
        # Use neural network to predict policy (action probabilities)
        policy, _ = self.policy_value_net(torch.tensor(node.state, dtype=torch.float32))
        for action in actions:
            if action not in node.children:
                new_state = self.next_state(node.state, action)
                child_node = MCTSNode(new_state, parent=node)
                child_node.prior_prob = policy[action].item()
                node.children[action] = child_node

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def get_valid_actions(self, state):
        # Placeholder for valid actions in a given state (problem-specific)
        return [0, 1, 2, 3]  # Example: 4 possible actions

    def next_state(self, state, action):
        # Placeholder for the next state based on the current state and action (problem-specific)
        return state + action  # Simplified example for state transition
