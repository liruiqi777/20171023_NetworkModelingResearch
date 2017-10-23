#!/usr/bin/env python
''' 
# ==============================================================================
Name: Son N. & Mingo S.
Date: 5/20/2017

CSCI 3210 - Computation Game Theory
External Influence Cascade Model

Description: Simulate linear threshold model with external events that affect
    the threshold. Results are written to 4 files, corresponding to the 4 
    different experimented topologies.

Running instruction (on the command line): 
Option 1: python3 catastrophe_game.py <num_nodes> <initial_prob_of_adoption>

Option 2: python3 catastrophe_game.py <config_file>
    This option will use the config file as input

Both options will prompt the user to enter another input file 
if the file does not exist, or allow them to enter 'q' to 
quit the program.

Output: 4 texts files corresponding to the 4 topologies we experimented. The
    format of the input file has 5 columns: Shock value, Num of Adopters, 
    Proportion of Switched Nodes, Number of Iterations for Eq. and Mean Weight
    (the meanings of these variables/columns are explained in the write-up).

NETWORKX REFERENCES:
https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html

# ==============================================================================
'''

# ==============================================================================
# LIBRARIES
# ==============================================================================
import bisect                       # For CDF functionality
# import matplotlib.pyplot as plt     # Drawing
import networkx as nx               # Constructing and visualizing graph
import numpy as np                  # Numerical methods
import os                           # File reading and checking
import re                           # Output formatting
import sys                          # Command line argument parsing
import timeit                       # Timing
import yaml                         # YAML parsing
from datetime import datetime       # Capture current time

# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================
MAX_ITERATION = 200
MAX_EQUILIBRIUM_ITERATION = 100
MAX_DEVIATION = 0.3
SHOCK_MEAN = 0
SHOCK_SD = 0.3
BARABASI_EDGE_FACTOR = 5
GRAPH_TOPOLOGY_NAME = ["random", "barabasi_albert", "watts_strogatz", "star"]

# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# These are parameters provided by the user.
num_nodes = 60
prob_of_initial = 0.3

# This is the list of actions over time.
action_history = []
shock_history = []
mean_weight_history = []
percent_change_history = []
iteration_history = []

# ==============================================================================
# FUNCTIONS
# ==============================================================================

#dont worry about it for now
# compute the mean weight of the adopters at equilibrium
def compute_mean_weight(equilibrium, G):

    total_weight_of_adopters = 0

    # go throug each node in the equilibrium, and sum up the total weighted 
    # influenced on that node from its incoming neighbors
    for node, action in enumerate(equilibrium):
        if (action == 1):
            incoming_neighbors = G.predecessors(node)
            for neighbor in incoming_neighbors:

                # only add if the neighbor is playing 1
                if (equilibrium[neighbor] == 1):
                    # My attempt to make the code work -- Sally[2]
                    total_weight_of_adopters += G[neighbor][node]["weight"]
                    #total_weight_of_adopters += G.edge[neighbor][node]["weight"]

    return float(total_weight_of_adopters) / len(G.nodes())

# calculate the proportion change between the two states
def calculate_proportion_change(prev_state, curr_state):
    total_change = 0
    for i, val in enumerate(prev_state):
        total_change += abs(val - curr_state[i])
    
    return float(total_change) / len(prev_state)

# perform the shock (more detailed documentation in the writeup)
def shock_effect(thresholds):

    global shock_history

    num_nodes = len(thresholds)

    new_thresholds = list(thresholds)

    # generate shock value
    # shock should be chosen from uniform distribution? --nd
    shock_value = np.random.normal(SHOCK_MEAN, SHOCK_SD, 1)[0]

    shock_history.append(shock_value)

    # each node's reaction to the shock
    sd = np.random.uniform(0, MAX_DEVIATION, num_nodes)

    # assign new threshold by drawing from the normal distribution
    for i, t in enumerate(thresholds):
        new_thresholds[i] += np.random.normal(shock_value, sd[i], 1)[0]
        
    #check that new_threshold is within the boundary
    return new_thresholds

# find the equilibrium of network G given initial state
def find_equilibrium(init_state, G, thresholds):

    global action_history, mean_weight_history, iteration_history

    final_state = list(init_state)
    new_state = list(init_state)
    num_iterations = 0

    # if no equilibrium is found after MAX_ITERATION, then assume that the
    # current state is the equilibrium.
    while (num_iterations < MAX_EQUILIBRIUM_ITERATION):

        state_change = 0

        for node in G.nodes():
            incoming_neighbors = G.predecessors(node) # in-coming neighbors
            
            # Edited by Sally[4] to get around the dict_keyiterator error below
            # and calculate the totla number of neighbors
            num_neighbors = 0
            for i in enumerate(incoming_neighbors):
                num_neighbors += 1
            # print("{} ".format(incoming_neighbors))   --- For debugging. Returns <dict_keyiterator object at 0x10543fc78>. why iterator not the list itself
            # num_neighbors = len(incoming_neighbors)
            sum_action = 0

            # the node has no incoming neighbors
            if not incoming_neighbors:
                continue

            # count the number of neighbors who play 1
            for neighbor in incoming_neighbors:                 
                sum_action += final_state[neighbor] * \
                                G.edge[neighbor][node]['weight']

            # switch to 1 since the added weights are more
            # than the threshold of the node            
            if (sum_action >= thresholds[node]):
                new_state[node] = 1
                if (final_state[node] == 0):
                    state_change += 1

            # switch to 0 since less than threshold
            else:
                new_state[node] = 0
                if (final_state[node] == 1):
                    state_change += 1

        final_state = list(new_state)
        num_iterations += 1

        # if no node switches, resulting in no state changes, then we are at 
        # equilibrium        
        if (state_change == 0):
            break

    action_history.append(final_state)
    mean_weight_history.append(compute_mean_weight(final_state, G))
    iteration_history.append(num_iterations)

    return final_state

# write out the records to files
def save_records(graph_index):
    global action_history, shock_history, mean_weight_history, \
        percent_change_history, iteration_history

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    fo = open("{}_output_{}.txt".format(current_time,
        GRAPH_TOPOLOGY_NAME[graph_index]), "w")
    fo.write("Shock value\tNum of Adopters\tProportion of Switched Nodes" + 
        "\tNum Iterations for Eq\tMean Weight\n")

    # write the records to the output file
    for i, item in enumerate(action_history):
        fo.write("{}\t{}\t{}\t{}\t{}\n".format(shock_history[i], item.count(1),
            percent_change_history[i], iteration_history[i], mean_weight_history[i]))

# reset the data structures that are used to store the measurements
def reset_data():
    global action_history, mean_weight_history, percent_change_history, \
            iteration_history

    action_history = []    
    mean_weight_history = []
    percent_change_history = []
    iteration_history = []

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    
    global action_history, shock_history, mean_weight_history, \
            percent_change_history

    argv = sys.argv
    argc = len(argv)
    
    #just give the num_nodes value--nd
    num_nodes = int(60)
    prob_of_initial = float(0.3)

    # If the user has specified parameters using command line variables, parse
    # them
    if (argc == 3):
        num_nodes, prob_of_initial = list(map(float, argv[1:]))
        num_nodes = int(num_nodes)
        #testing--nd
        print("imhere")
        
    # If the user has provided a YAML file, read it.
    elif (argc == 2):
        
        file_name = argv[1]
        try:
            with open(file_name, "r") as yml_file:
                try:
                    config_file = yaml.load(yml_file)                
                    num_nodes = int(config_file["num_nodes"]) 
                    #test--nd
                    print("imhere")
                    
                    prob_of_initial = config_file["prob_of_initial"]                  
                except KeyError as e:
                    raise KeyError(("Expected variable {} in config file {}, " 
                        + "but it wasn't found.").format(e, config_file))
        except Exception as error:
            print("Reading error: {}".format(error))
            quit()
    else:
        print("Error: Invalid number of arguments. Correct ordering is:")
        print("catastrophe_game.py <config.file>")
        print("or")
        print("catastrophe_game.py <num_nodes> <prob_of_initial>")
        quit()
    
    print("[GAME SETTINGS]")
    print("Number of nodes: {}".format(num_nodes))
    print("Probability of initial adopters: {}".format(prob_of_initial))

    # generate the random networks
    erdos_renyi_graph = nx.erdos_renyi_graph(num_nodes,
        BARABASI_EDGE_FACTOR / num_nodes).to_directed()

    barabasi_albert_graph = nx.barabasi_albert_graph(num_nodes, 
        BARABASI_EDGE_FACTOR).to_directed()

    watts_strogatz_graph = nx.watts_strogatz_graph(num_nodes, 
        BARABASI_EDGE_FACTOR, 0).to_directed()

    star_graph = nx.star_graph(num_nodes - 1).to_directed()

    graphs = [erdos_renyi_graph, barabasi_albert_graph, watts_strogatz_graph, 
        star_graph]

    # generate the weights
    for graph_index, graph in enumerate(graphs):

        for node in range(num_nodes):

            in_degree = graph.in_degree(node)

            if not in_degree:
                continue

            total_weight = np.random.uniform(0,1,1)
            edge_weights = np.random.uniform(0,1,in_degree)
            edge_weights_sum = sum(edge_weights)
#            print("Total weight: {}".format(total_weight))
#            print("Edge_weights: {}".format(edge_weights))
#            print("Edge_weights_sum: {}".format(edge_weights_sum))

#            for weights in edge_weights:
#                weights = weights/edge_weights_sum*total_weight

            edge_weights = edge_weights/edge_weights_sum*total_weight;
            
#            print("Normalized edge_weights: {}".format(edge_weights))
#            print("Normaliezd edge_weights_sum: {}".format(sum(edge_weights)))

 #           print("\n\n\n\n\n")


            # create random weights
            # dirichlet: random numbers with a given sum
            # edge_weights = np.random.dirichlet(np.ones(in_degree))
            # edge_weights[-1] = np.random.uniform(0, edge_weights[-1])

            # assign weight to each incoming edge
            for i, neighbor in enumerate(graph.predecessors(node)):
                graph[node][neighbor]["weight"] = edge_weights[i] 
                
        print("Number of edges for {}: {}".format(GRAPH_TOPOLOGY_NAME[graph_index],
            len(graph.edges())))

    print("[SUMMARY]")

    # Create an initial state by randomly assigning actions to each player.
    init_state = list(np.random.binomial(1, prob_of_initial, num_nodes))
    print("Number of initial adopters for both graphs: {}".format(init_state.count(1)))

    # my attempt to make the code work -- Sally[1]
    # thresholds is a [list] of doubles taken from a uniform distribution [0,1) 
    thresholds = np.random.uniform(0, 1, num_nodes)  

    # should we fix shock to both graphs, as well as its effect
    thresholds_array = []
    new_thresholds = list(thresholds)
    shock_history.append(0)

    # pre-generate a shock value list and its effect on thresholds for all the
    # networks
    for i in range(MAX_ITERATION):
        new_thresholds = shock_effect(new_thresholds)
        thresholds_array.append(new_thresholds)

    # run the experiment on all of the networks
    for i, graph in enumerate(graphs):

        curr_time = 0

        reset_data()

        prev_state = []    
        curr_state = list(init_state)

        # init data
        action_history.append(curr_state)
        mean_weight_history.append(compute_mean_weight(curr_state, graph))
        percent_change_history.append(0)    # no change at initial state

        while (curr_time < MAX_ITERATION):

            prev_state = list(curr_state)            
            curr_state = find_equilibrium(curr_state, graph, thresholds)

            percent_change_history.append(calculate_proportion_change(prev_state,
                                                            curr_state))

            # update time
            curr_time += 1

            # introduce the shock, by setting the threshold to the pre-computed
            # values -- this makes sure that all networks are experiencing the
            # same shock
            if (curr_time < MAX_ITERATION):
                thresholds = thresholds_array[curr_time]

        print("Number of final adopters for {} graph: {}".format(\
            GRAPH_TOPOLOGY_NAME[i], curr_state.count(1)))

        iteration_history.append(0)

        save_records(i)
        
        #print the percentage of adopters--nd
        print("Percentage of final adopters for {} graph: {}".format(\
              GRAPH_TOPOLOGY_NAME[i], (curr_state.count(1)-num_nodes*0.1)\
              /(num_nodes*0.9))
        ) 

import time
# main
if __name__ == "__main__":
    #try to time the experiment--nd
    
   
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
