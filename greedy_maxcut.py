from utils import *

def get_gains(graph,ground_set):
    if ground_set is None:

        gains={node:graph.degree(node) for node in graph.nodes()}
    else:
        # print('A ground set has been given')
        gains={node:graph.degree(node) for node in ground_set}
        # print('Size of the ground set = ',len(gains))

    
    return gains


def gain_adjustment(graph,gains,selected_element,spins):

    gains[selected_element]=-gains[selected_element]

    for neighbor in graph.neighbors(selected_element):

        if neighbor in gains:
            gains[neighbor]+=(2*spins[neighbor]-1)*(2-4*spins[selected_element])

    spins[selected_element]=1-spins[selected_element]

def greedy(graph,budget,ground_set=None):
    
    number_of_queries = 0

    gains = get_gains(graph,ground_set)
    
    solution=[]
    
    spins={node:1 for node in graph.nodes()}
    obj_val = 0

    for i in range(budget):
        number_of_queries += (len(gains)-i)

        selected_element=max(gains, key=gains.get)

        if gains[selected_element]<=0:
            # print('All edges are already covered')
            break
        solution.append(selected_element)

        obj_val += gains[selected_element]
        
        gain_adjustment(graph,gains,selected_element,spins)
    # print('Objective value =', obj_val)
    # print('Number of queries =',number_of_queries)

    return obj_val,solution,number_of_queries


def maxcut_greedy_rollout(graph,depth,nodes):


   
    spins={node:1 for node in graph.nodes()}
    gains = get_gains(graph=graph,ground_set=None)
    solution=[]
    obj_val = 0
    for node in nodes:
        obj_val += gains[node]
        gain_adjustment(graph=graph,
                        gains=gains,
                        selected_element=node,
                        spins=spins)
        solution.append(node)
        
    
    for i in range(depth-len(nodes)):
        selected_element=max(gains, key=gains.get)
        obj_val += gains[selected_element]
        if gains[selected_element] <= 0:
            # print('All elements are already covered')
            break
        gain_adjustment(graph=graph,
                        gains=gains,
                        selected_element=selected_element,
                        spins=spins)
        solution.append(node)

    return obj_val



