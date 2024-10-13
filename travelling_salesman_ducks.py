# Travelling Salesman

import itertools
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def path_cost(graph, path):

    """
    Calculate the total cost of a given path. 
    
    """
    cost = 0

    # iterate through path adding costs as you go 
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i+1]]
    
    # add a cost to return to the starting point 
    cost += graph[path[-1]][path[0]]

    return cost

def traveling_salesman(graph, vertices):
    """
    Finds all possible paths to complete the trip then finds 
    the best path by finding the one with the shortest length. 

    """
    n = len(vertices) # number of verticies to visit 
    min_cost = float('inf') # initialize minimum cost to inifity 
    best_path = None 

    # generate all paths that the verticies can form 
    for path in itertools.permutations(vertices):

        # calcuate the cost of the path 
        cost = path_cost(graph, path)

        # checks if this is the best path and replaces best path if so 
        if cost < min_cost:
            min_cost = cost
            best_path = path

    return best_path, min_cost

def generate_random_tsp_graph(num_vertices, edge_probability=0.5):
    """
    Generates a random graph using NetworkX (as was recommended in class.) 
    Then creates an adj matrix from that graph. 
    
    """
    # create a random graph 
    G = nx.gnp_random_graph(num_vertices, edge_probability)
    
    # assign weights to the edges 
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.randint(1, 100)
    
    # covert to adjacency matrix 
    adj_matrix = nx.to_numpy_array(G, weight='weight', dtype=int)
    return adj_matrix

# Measure execution time of the TSP solver
def measure_tsp_execution_time(graph):
    """
    Measures time taken to execute the above functions to find the 
    most optimal path for the traveling salesman to take 
    
    """

    # create a list of verticies 
    vertices = list(range(len(graph)))

    # time and execute the function 
    start_time = time.time()
    best_path, min_cost = traveling_salesman(graph, vertices)
    end_time = time.time()

    return best_path, min_cost, end_time - start_time

# Generate graphs of varying sizes and test the TSP solver
def test_tsp_solver_performance():
    """
    This function tests the performance of the traveling salesman solver for randomly 
    generated graphs of different sizes. 
    
    It records the execution time and plots them. 
    """
     
    sizes = [4, 5, 6, 7, 8, 9, 10] # test sizes 
    execution_times = []

    for size in sizes:
        # generate a random graph with the current number of verticies 
        graph = generate_random_tsp_graph(size, edge_probability=0.5)
        # measure how long it takes to find the best path 
        best_path, min_cost, exec_time = measure_tsp_execution_time(graph)
        # store the execution times 
        execution_times.append(exec_time)
        # print the size of graph, the best path found and the execution time 
        print(f"Size: {size}, Best path: {best_path}, Minimum cost: {min_cost}, Execution time: {exec_time:.4f} seconds")

    # plot the execution times 
    plt.plot(sizes, execution_times, marker='o')
    plt.title("Traveling Salesman Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (seconds)")
    plt.show()

# run performance test 
if __name__ == "__main__":
    test_tsp_solver_performance()
