#By Sally
#1. Apply shocks to only a subset of nodes
#2. Can control iterations by pressing enter
#3. Recording more info

#!/usr/bin/env python

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
import random
from datetime import datetime       # Capture current time


# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================
MAX_DEVIATION = 0.3
SHOCK_MEAN = 0
SHOCK_SD = 0.01
BARABASI_EDGE_FACTOR = 5
GRAPH_TOPOLOGY_NAME = ["random", "barabasi_albert", "watts_strogatz", "star"]


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# These are parameters provided by the user.
num_nodes = 0
prob_of_initial = 0
graphs = []

initial_thresholds = []
initial_state = []
edge_info = []
agent_state = []
agent_thresholds = []

#action_history = []
#shock_history = []
#mean_weight_history = []
#percent_change_history = []
#iteration_history = []










## ==============================================================================
## FUNCTIONS
## ==============================================================================
#
##dont worry about it for now
## compute the mean weight of the adopters at equilibrium
#def compute_mean_weight(equilibrium, G):
#
#    total_weight_of_adopters = 0
#
#    # go throug each node in the equilibrium, and sum up the total weighted 
#    # influenced on that node from its incoming neighbors
#    for node, action in enumerate(equilibrium):
#        if (action == 1):
#            incoming_neighbors = G.predecessors(node)
#            for neighbor in incoming_neighbors:
#
#                # only add if the neighbor is playing 1
#                if (equilibrium[neighbor] == 1):
#                    # My attempt to make the code work -- Sally[2]
#                    total_weight_of_adopters += G[neighbor][node]["weight"]
#                    # total_weight_of_adopters += G.edge[neighbor][node]["weight"]
#
#    return float(total_weight_of_adopters) / len(G.nodes())
#
## calculate the proportion change between the two states
#def calculate_proportion_change(prev_state, curr_state):
#    total_change = 0
#    for i, val in enumerate(prev_state):
#        total_change += abs(val - curr_state[i])
#    
#    return float(total_change) / len(prev_state)
#
## perform the shock (more detailed documentation in the writeup)
#def shock_effect(thresholds):
#
#    global shock_history
#
#    num_nodes = len(thresholds)
#
#    new_thresholds = list(thresholds)
#
#    # generate shock value
#    # shock should be chosen from uniform distribution? --nd
#    shock_value = np.random.normal(SHOCK_MEAN, SHOCK_SD, 1)[0]
#
#    shock_history.append(shock_value)
#
#    # each node's reaction to the shock
#    sd = np.random.uniform(0, MAX_DEVIATION, num_nodes)
#
#    # assign new threshold by drawing from the normal distribution
#    for i, t in enumerate(thresholds):
#        effect = np.random.normal(shock_value, sd[i], 1)[0]
#        new_thresholds[i] = new_thresholds[i] + (1/2)*effect*(1-new_thresholds[i])
#    
##    print("Standard Deviation generated for each agent during this shock:")
##    print(sd)
#        
#    #check that new_threshold is within the boundary
#    return new_thresholds
#

#
## write out the records to files
#def save_records(graph_index):
#    global action_history, shock_history, mean_weight_history, \
#        percent_change_history, iteration_history
#
#    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#
#    fo = open("{}_output_{}.txt".format(current_time,
#        GRAPH_TOPOLOGY_NAME[graph_index]), "w")
#    fo.write("Shock value\tNum of Adopters\tProportion of Switched Nodes" + 
#        "\tNum Iterations for Eq\tMean Weight\n")
#
#    # write the records to the output file
#    for i, item in enumerate(action_history):
#        fo.write("{}\t{}\t{}\t{}\t{}\n".format(shock_history[i], item.count(1),
#            percent_change_history[i], iteration_history[i], mean_weight_history[i]))
#



def find_equilibrium(graph_index, state_record):

    global num_nodes, graphs, edge_info, agent_state

    new_state = agent_state * 1

    while True:
        for node in range(num_nodes):
            influence_from_neighbor = 0
            for neighbor in range(num_nodes):
                influence_from_neighbor = 0
                for neighbor_index, neighbor in enumerate(graphs[graph_index].predecessors(node)):
                    influence_from_neighbor = influence_from_neighbor + edge_info[graph_index][neighbor][node] * agent_state[neighbor]
            if (influence_from_neighbor >= agent_thresholds[node]): new_state[node] = 1
            else: new_state[node] = 0
        if np.array_equal(new_state, agent_state): break
        else:
            for ind in range(num_nodes):
                state_record.write("{} ".format(new_state[ind]))
            state_record.write("\n")
            agent_state = new_state


## find the equilibrium of network G given initial state
#def find_equilibrium(init_state, G, thresholds):
#
#    global action_history, mean_weight_history, iteration_history
#
#    final_state = list(init_state)
#    new_state = list(init_state)
#    num_iterations = 0
#
#    # if no equilibrium is found after MAX_ITERATION, then assume that the
#    # current state is the equilibrium.
#    while (num_iterations < MAX_EQUILIBRIUM_ITERATION):
#
#        state_change = 0
#
#        for node in G.nodes():
#            incoming_neighbors = G.predecessors(node) # in-coming neighbors
#
#            # Edited by Sally[4] to get around the dict_keyiterator error below
#            # and calculate the totla number of neighbors
#            num_neighbors = 0
#            for i in enumerate(incoming_neighbors):
#                num_neighbors += 1
#            # print("{} ".format(incoming_neighbors))   --- For debugging. Returns <dict_keyiterator object at 0x10543fc78>. why iterator not the list itself
#            # num_neighbors = len(incoming_neighbors)
#            sum_action = 0
#
#            # the node has no incoming neighbors
#            if not incoming_neighbors:
#                continue
#
#            # count the number of neighbors who play 1
#            for neighbor in incoming_neighbors:
#                sum_action += final_state[neighbor] * \
#                                G.edge[neighbor][node]['weight']
#
#            # switch to 1 since the added weights are more
#            # than the threshold of the node
#            if (sum_action >= thresholds[node]):
#                new_state[node] = 1
#                if (final_state[node] == 0):
#                    state_change += 1
#
#            # switch to 0 since less than threshold
#            else:
#                new_state[node] = 0
#                if (final_state[node] == 1):
#                    state_change += 1
#
#        final_state = list(new_state)
#        num_iterations += 1
#
#        # if no node switches, resulting in no state changes, then we are at
#        # equilibrium
#        if (state_change == 0):
#            break
#
#    action_history.append(final_state)
#    mean_weight_history.append(compute_mean_weight(final_state, G))
#    iteration_history.append(num_iterations)
#
#    return final_state




def simulate_next_shock(graph_index, state_record, threshold_record):
    global num_nodes, edge_info, graphs, agent_state, agent_thresholds
    shock_value = 






def main():

    global num_nodes, prob_of_initial, edge_info, initial_state, initial_thresholds, graphs, agent_state, agent_thresholds

    argv = sys.argv
    argc = len(argv)

    # processing command line input
    if (argc == 3):
        num_nodes, prob_of_initial = list(map(float, argv[1:]))
        num_nodes = int(num_nodes)

        if (prob_of_initial < 0 or prob_of_initial > 1):
            print("Invalid initial adoption probability. Please make sure the value is between 0 and 1.")
            sys.exit()

    else:
        print("Error: Invalild command line inputs.")
        print("Correct command line format: new_model_20171115.py <num_nodes> <probability of initial adoption>")
        sys.exit()

    print("There are {} agents in the game and the probability of initial adoption is {}."
        .format(num_nodes, prob_of_initial))

    # generate random graphs
    random.seed(None)
    star_graph = nx.star_graph(num_nodes - 1).to_directed()
    barabasi_albert_graph = nx.barabasi_albert_graph(num_nodes, BARABASI_EDGE_FACTOR).to_directed()

    graphs = [star_graph, barabasi_albert_graph]
    graph_name = ["star_graph", "barabasi_albert_graph"]


    # generate the weights of edges for each graph
    for graph_index, graph in enumerate(graphs):

        edge_in_graph = [[0 for x in range(num_nodes)] for y in range(num_nodes)]

        for node in range(num_nodes):

            in_degree = graph.in_degree(node)

            if not in_degree:
                continue

            # calculate the total weight of influence received from neighbors
            total_weight = np.random.uniform(0, 1, 1)
            edge_weights = np.random.uniform(0, 1, in_degree)
            edge_weights_sum = sum(edge_weights)
            # normalizing weights
            edge_weights = edge_weights/edge_weights_sum*total_weight

            print("Normalized edge weights are: {}".format(edge_weights))

            # storing the weights
            # print(list(enumerate(graph.predecessors(node))))
            for neighbor_index, neighbor in enumerate(graph.predecessors(node)):
                edge_in_graph[neighbor][node] = edge_weights[neighbor_index]

        edge_info.append(edge_in_graph)

    # print("{}".format(edge_info))

    # generating an initial state for all agents
    # format: array of 0 and 1 indicating the intial adoption decision
    # 1 for adopters and 0 otherwise
    initial_state = np.random.binomial(1, prob_of_initial, num_nodes)
    print(initial_state)
    print("The number of initial adopters for all graphs are {}".format(sum(initial_state)))

    # generate initial agent thresholds
    # format: array of real numbers between 0 and 1
    # indicating the magnitude of threshold
    initial_thresholds = np.random.uniform(0, 1, num_nodes)
    for i in range(0, len(initial_thresholds)):
        if (initial_state[i] == 1):
            initial_thresholds[i] = 0
    print("{}".format(initial_thresholds))


    print("{}".format(edge_info))

    # record essential information
    # name files using current time
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    for graph_index, graph in enumerate(graphs):

        agent_state = initial_state * 1
        agent_thresholds = initial_thresholds * 1

        # open two files to record information
        state_record = open("{}_change_of_state_{}".format(graph_name[graph_index], current_time), "w")
        state_record.write("Number of initial adopters: {}\n\n".format(sum(initial_state)))
        threshold_record = open("{}_change_of_threshold_{}".format(graph_name[graph_index], current_time), "w")
        threshold_record.write("Number of initial adopters: {}\n\n".format(sum(initial_state)))

        # record initial states and intial thresholds
        for ind in range(num_nodes):
            state_record.write("{} ".format(initial_state[ind]))
            threshold_record.write("{} ".format(initial_thresholds[ind]))
        state_record.write("\n")
        threshold_record.write("\n")

        print("Current graph is: {}".format(graph_name[graph_index]))
        print("Press enter for next round of shock simulation." +
              "Press t then enter to terminate simulation for this graph and start simulation for the next.")
        user_keypress = input("")
        find_equilibrium(graph_index, state_record)


        # wait for user input
        while(user_keypress == "" or user_keypress == "t"):
            if (user_keypress == "t"):
                print("End of simulation for {}.\n\n\n\n".format(graph_name[graph_index]))
                break
            print("Next iteration: ")
            simulate_next_shock(graph_index, state_record, threshold_record)
            user_keypress = input("")



main()
