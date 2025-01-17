from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt


def greedy_maxcover(graph, budget, ground_set=None):
    number_of_queries = 0
    obj_val = 0

    # If ground_set is not provided, initialize it with all nodes in the graph
    if ground_set is None:
        ground_set = list(graph.nodes())

    solution = []
    uncovered = set()

    for i in range(budget):
        gains = {}

        # Calculate gains for each node in the ground_set
        for node in ground_set:
            if node not in solution:
                number_of_queries += 1
                gains[node] = 0

                # Gain for covering the current node
                if node not in uncovered:
                    gains[node] += 1
                
                # Gain for covering the neighbors of the node
                for neighbor in graph.neighbors(node):
                    if neighbor not in uncovered:
                        gains[node] += 1

        # Find the node with the maximum gain
        if gains:
            selected_element = max(gains, key=gains.get)

            # If no gain can be achieved, break the loop
            if gains[selected_element] == 0:
                break
            else:
                obj_val += gains[selected_element]
                solution.append(selected_element)

                # Mark the selected node and its neighbors as covered
                uncovered.add(selected_element)
                for neighbor in graph.neighbors(selected_element):
                    uncovered.add(neighbor)

    return obj_val, solution, number_of_queries
