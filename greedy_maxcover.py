from argparse import ArgumentParser
# from utils import *
# import pandas as pd
from collections import defaultdict
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_obj(graph, solution):

    covered_elements=set()
    for node in solution:
        covered_elements.add(node)
        for neighbour in graph.neighbors(node):
            covered_elements.add(neighbour)
    
    return len(covered_elements)


# def calculate_obj(graph,solution):


#     covered_elements=set()
#     for node in range(graph.number_of_nodes()):
#         if solution[node] == 0:
#             covered_elements.add(node)
#             for neighbour in graph.neighbors(node):
#                 covered_elements.add(neighbour)

#     return len(covered_elements)



def get_gains(graph,ground_set):
    if ground_set is None:

        gains={node:graph.degree(node)+1 for node in graph.nodes()}
    else:
        # print('A ground set has been given')
        gains={node:graph.degree(node)+1 for node in ground_set}
        # print('Size of ground set',len(gains))
    return gains

    
def gain_adjustment(graph,gains,selected_element,uncovered):

    
            

    if uncovered[selected_element]:
        gains[selected_element]-=1
        uncovered[selected_element]=False
        for neighbor in graph.neighbors(selected_element):
            if neighbor in gains and gains[neighbor]>0:
                gains[neighbor]-=1

    for neighbor in graph.neighbors(selected_element):
        if uncovered[neighbor]:
            uncovered[neighbor]=False
            
            if neighbor in gains:
                gains[neighbor]-=1
            for neighbor_of_neighbor in graph.neighbors(neighbor):
                if neighbor_of_neighbor  in gains:
                    gains[neighbor_of_neighbor ]-=1


    assert gains[selected_element] == 0





def greedy(graph,budget,ground_set=None):
    
    number_of_queries = 0

    obj_val = 0 

    gains = get_gains(graph,ground_set)
    


    solution=[]
    uncovered=defaultdict(lambda: True)

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        obj_val += gains[selected_element]

        if gains[selected_element]==0:
            # print('All elements are already covered')
            break
        solution.append(selected_element)
        # print(gains[selected_element])
        gain_adjustment(graph,gains,selected_element,uncovered)

    # print('Number of queries:',number_of_queries)


    return obj_val,solution,number_of_queries



def maxcover_greedy_rollout(graph,depth,nodes):

    # print('Using greedy rollout')
    print('Nodes',nodes)


   
    uncovered = defaultdict(lambda: True)
    gains = get_gains(graph=graph,ground_set=None)
    solution=[]
    for node in nodes:
        gain_adjustment(graph=graph,
                        gains=gains,
                        selected_element=node,
                        uncovered=uncovered)
        solution.append(node)
        
    
    for i in range(depth-len(nodes)):
        selected_element=max(gains, key=gains.get)

        if gains[selected_element] == 0:
            # print('All elements are already covered')
            break
        gain_adjustment(graph=graph,
                        gains=gains,
                        selected_element=selected_element,
                        uncovered=uncovered)
        solution.append(selected_element)

    return calculate_obj(graph,solution)
        
    


    # solution=[]
    # uncovered=defaultdict(lambda: True)

    # for i in range(budget):
    #     number_of_queries += (len(gains)-i)

    #     selected_element=max(gains, key=gains.get)

    #     obj_val += gains[selected_element]

    #     if gains[selected_element]==0:
    #         # print('All elements are already covered')
    #         break
    #     solution.append(selected_element)
    #     # print(gains[selected_element])
    #     gain_adjustment(graph,gains,selected_element,uncovered)

    # # print('Number of queries:',number_of_queries)


    # return obj_val,solution,number_of_queries


    
    

        









