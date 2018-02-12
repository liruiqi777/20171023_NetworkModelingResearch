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
SHOCK_MEAN = 0
SHOCK_SD = 0.01
BARABASI_EDGE_FACTOR = 5
SHOCK_PROB = 0.2
GRAPH_TOPOLOGY_NAME = ["random", "barabasi_albert", "watts_strogatz", "star"]
INITIAL_ADOPTER_GENERATOR = ["greedy", "degree", "influence"]
INITIAL_ADOPTER_THRESHOLD = 0.001


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# These are parameters provided by the user.
num_nodes = 0
prob_of_initial = 0
graphs = []

initial_thresholds = []
initial_states = []
edge_info = []
agent_state = []
agent_thresholds = []
num_initial_adopter = 0


def initial_adopter_selection_by_degree(graph_index):
    global edge_info, num_initial_adopter

    print(num_initial_adopter)

    node_degree = [0] * num_nodes

    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if (edge_info[graph_index][node][neighbor] != 0):
                node_degree[node] = node_degree[node] + 1

    node_degree_copy = node_degree * 1

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (node_degree[i] < node_degree[j]):
                temp = node_degree[i]
                node_degree[i] = node_degree[j]
                node_degree[j] = temp
    lowest_degree = node_degree[num_initial_adopter-1]

    initial_adopter_by_degree = [0] * num_nodes
    for node in range(num_nodes):
        if (node_degree_copy[node] >= lowest_degree):
            initial_adopter_by_degree[node] = 1
        if (sum(initial_adopter_by_degree) == num_initial_adopter): break

    print(node_degree)
    print(lowest_degree)

    return initial_adopter_by_degree





def initial_adopter_selection_by_influence(graph_index):
    global edge_info, num_initial_adopter

    print(num_initial_adopter)

    node_influence = []
    for node in range(num_nodes):
        node_influence.append(sum(edge_info[graph_index][node]))
    node_influence_copy = node_influence*1
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (node_influence[i] < node_influence[j]):
                temp = node_influence[i]
                node_influence[i] =node_influence[j]
                node_influence[j] = temp
    lowest_influence = node_influence[num_initial_adopter-1]

    print(node_influence)
    print(lowest_influence)

    initial_adopter_by_influence = [0] * num_nodes
    for node in range(num_nodes):
        if (node_influence_copy[node] >= lowest_influence):
            initial_adopter_by_influence[node] = 1
        if (sum(initial_adopter_by_influence) == num_initial_adopter): break

    return initial_adopter_by_influence




def run_til_eq(state, node_to_try, dynamic_threshold):

    global edge_info, num_nodes, edge_info

    state_copy = state * 1
    state_copy[node_to_try] = 1
    dynamic_threshold_copy = dynamic_threshold * 1
    dynamic_threshold[node_to_try] = 0

    new_state = state_copy * 1

    while 1:
        for node in range(num_nodes):
            if (state_copy[node] == 1): continue
            influence_from_neighbor = 0
            for neighbor in range(num_nodes):
                influence_from_neighbor = influence_from_neighbor + state[neighbor] * edge_info[graph_index][neighbor][node]
            if(influence_from_neighbor >= dynamic_threshold[node]): new_state[node] = 1

            if(sum(new_state) == sum(state)): break

    return sum(new_state)-sum(state)





def initial_adopter_selection_greedy(graph_index):
    global initial_thresholds, num_nodes, edge_info

    dynamic_threshold = initial_thresholds * 1
    num_converted = [-1] * num_nodes
    greedy_optimal = [0] * num_nodes

    state = [0] * num_nodes

    while (sum(greedy_optimal) != num_initial_adopter) && (sum(state) != num_nodes):
        for node in range(num_nodes):
            if (state[node] != 1):
                num_converted[node] = run_til_eq(state, node, dynamic_threshold)
        index = num_converted.index(max(num_converted))
        dynamic_threshold[index] = 0
        num_converted = [-1] * num_nodes
        greedy_optimal[index] = 1
        state[index] = 1

        new_state = state * 1

        while 1:
            for node in range(num_nodes):
                if (state[node] == 1): continue
                influence_from_neighbor = 0
                for neighbor in range(num_nodes):
                    influence_from_neighbor = influence_from_neighbor + state[neighbor] * edge_info[graph_index][neighbor][node]
                if(influence_from_neighbor >= dynamic_threshold[node]): new_state[node] = 1

            if(sum(new_state) == sum(state)):
                state = new_state * 1
                break

            state = new_state * 1


    return greedy_optimal





def find_initial_adopter(graph_index):
    global initial_states
    initial_states = []
    initial_states.append(initial_adopter_selection_greedy(graph_index))
    initial_states.append(initial_adopter_selection_by_degree(graph_index))
    initial_states.append(initial_adopter_selection_by_influence(graph_index))



def find_equilibrium(graph_index, state_record, round_num):

    global num_nodes, graphs, edge_info, agent_state

    new_state = agent_state * 1
    iteration = 0
    max_iter = 2**num_nodes

    while iteration < max_iter:
        iteration = iteration + 1
        for node in range(num_nodes):
            influence_from_neighbor = 0
            for neighbor in range(num_nodes):
                influence_from_neighbor = influence_from_neighbor + edge_info[graph_index][neighbor][node] * agent_state[neighbor]
            if (influence_from_neighbor >= agent_thresholds[node]): new_state[node] = 1
            else: new_state[node] = 0
        if np.array_equal(new_state, agent_state): break
        else:
            agent_state = new_state

    state_record.write("Round {}: ".format(round_num))
    for ind in range(num_nodes):
        state_record.write("{} ".format(agent_state[ind]))
    state_record.write("\n")



def simulate_next_shock(graph_index, state_record, threshold_record, shock_record, round_num):
    global num_nodes, edge_info, graphs, agent_state, agent_thresholds
    shock_value = np.random.uniform(-1, 1, 1)
    shocked_agent = np.random.binomial(1, SHOCK_PROB, num_nodes)
    shock_record.write("Round {}: Shock value is: {}\n".format(round_num, shock_value[0]))
    shock_record.write("Shocked agents are: {}\n".format(shocked_agent))
    print("Shock value is {}".format(shock_value[0]))
    print("Shocked agents are: {}".format(shocked_agent))

    agent_thresholds = agent_thresholds + shock_value * (agent_thresholds - agent_thresholds * agent_thresholds) * shocked_agent

    find_equilibrium(graph_index, state_record, round_num)

    threshold_record.write("Round {}: ".format(round_num))
    for ind in range(num_nodes):
        threshold_record.write("{:0.3f} ".format(agent_thresholds[ind]))
    threshold_record.write("\n")

    print("Number of adopters at the new equilibrium is {}".format(sum(agent_state)))



def main():

    global num_nodes, num_initial_adopter, edge_info, initial_state, initial_thresholds, graphs, agent_state, agent_thresholds

    argv = sys.argv
    argc = len(argv)

    # processing command line input
    if (argc == 3):
        num_nodes, num_initial_adopter = list(map(int, argv[1:]))

        if (num_initial_adopter < 1):
            print("Invalid initial adoption probability. Please make sure the value is between 0 and 1.")
            sys.exit()

    else:
        print("Error: Invalild command line inputs.")
        print("Correct command line format: new_model_20171115.py <num_nodes> <probability of initial adoption>")
        sys.exit()

    print("There are {} agents in the game and {} of them are initial adopters."
        .format(num_nodes, num_initial_adopter))

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

#            print("Normalized edge weights are: {}".format(edge_weights))

            # storing the weights
            # print(list(enumerate(graph.predecessors(node))))
            for neighbor_index, neighbor in enumerate(graph.predecessors(node)):
                edge_in_graph[neighbor][node] = edge_weights[neighbor_index]

        edge_info.append(edge_in_graph)

    # print("{}".format(edge_info))

    # generate initial agent thresholds
    # format: array of real numbers between 0 and 1
    # indicating the magnitude of threshold
    initial_thresholds = np.random.uniform(0, 1, num_nodes)



    # print("{}".format(edge_info))

    # record essential information
    # name files using current time
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    for graph_index, graph in enumerate(graphs):
        # generating initial state for all agents
        # format: array of 0 and 1 indicating the initial adoption decision
        # 1 for adopters and 0 otherwise
        find_initial_adopter(graph_index)
        print("Using greedy algorithm, the initial states are {}\n".format(initial_states[0]))
        print("Using the degree approach, the initial states are {}\n".format(initial_states[1]))
        print("Using the total influence approach, the initial states are {}\n\n".format(initial_states[2]))




#    for graph_index, graph in enumerate(graphs):
#
#
#        # generating an initial state for all agents
#        # format: array of 0 and 1 indicating the intial adoption decision
#        # 1 for adopters and 0 otherwise
#        # initial_state = np.random.binomial(1, prob_of_initial, num_nodes)
#        find_initial_adopter(graph_index)
#        print(initial_state[0])
#        print("The number of initial adopters for {} are {}\n".format(graph_name[graph_index], sum(initial_state[0])))
#
#        for neighbor in range(num_nodes):
#            print("\n{}'s impact: ".format(neighbor))
#            for node in range(num_nodes):
#                print("{:0.3f} ".format(edge_info[graph_index][neighbor][node]))
#
#        print("\n\n")
#        print(initial_thresholds)
#
#
#
#
#        for i in range(0, len(initial_thresholds)):
#            if (initial_state[i] == 1):
#                initial_thresholds[i] = 0
#
##        round_num = 0
##
##
##        agent_state = initial_state * 1
##        agent_thresholds = initial_thresholds * 1
##
##        # open two files to record information
##        state_record = open("{}_{}_state_hist".format(graph_name[graph_index], current_time), "w")
##        state_record.write("Number of initial adopters: {}\n".format(sum(initial_state)))
##        state_record.write("Initial adoption decision: \n")
##        threshold_record = open("{}_{}_threshold_hist".format(graph_name[graph_index], current_time), "w")
##        threshold_record.write("Number of initial adopters: {}\n\n".format(sum(initial_state)))
##        shock_record = open("{}_{}_shock_hist".format(graph_name[graph_index], current_time), "w")
##        edge_info_record = open("{}_{}_edge_info".format(graph_name[graph_index], current_time), "w")
##
##        for neighbor in range(num_nodes):
##            edge_info_record.write("\n{}'s impact: ".format(neighbor))
##            for node in range(num_nodes):
##                edge_info_record.write("{:0.3f} ".format(edge_info[graph_index][neighbor][node]))
##
##        # record initial states and intial thresholds
##        threshold_record.write("Round {}: ".format(round_num))
##        state_record.write("Round {}: ".format(round_num))
##        for ind in range(num_nodes):
##            state_record.write("{} ".format(initial_state[ind]))
##            threshold_record.write("{:0.3f} ".format(initial_thresholds[ind]))
##
##        state_record.write("\n\n")
##        threshold_record.write("\n")
##
##        print("\nCurrent graph is: {}".format(graph_name[graph_index]))
##        print("Press enter for next round of shock simulation." +
##              "Press t then enter to terminate simulation for this graph and start simulation for the next.")
##        user_keypress = input("")
##        find_equilibrium(graph_index, state_record, 0)
##
###        for i in range(0, len(initial_thresholds)):
###            if (initial_state[i] == 1):
###                agent_thresholds[i] = INITIAL_ADOPTER_THRESHOLD
##
##        # wait for user input
##        while(user_keypress == "" or user_keypress == "t"):
##            round_num = round_num + 1
##            if (user_keypress == "t"):
##                print("End of simulation for {}.\n\n\n\n".format(graph_name[graph_index]))
##                break
##            print("Next iteration: ")
##            simulate_next_shock(graph_index, state_record, threshold_record, shock_record, round_num)
##            user_keypress = input("")
##
##        state_record.close()
##        threshold_record.close()
##        edge_info_record.close()
##        shock_record.close()



main()
