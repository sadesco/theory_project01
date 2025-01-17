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

    return end_time - start_time

# Generate graphs of varying sizes and test the TSP solver
def test_tsp():
    """
    This function tests the performance of the Hamiltonian cycle solver for randomly
    generated graphs of different sizes.

    It records the execution time and plots to show an O(n!) complexity.

    """
    sizes = [4, 5, 6, 7, 8, 9, 10] # sizes that represent the number of verticies
    avg_times = []
    all_times = {}
    trials = 15 # number of trials per graph size

    for size in sizes:
        exec_times = []
        for _ in range(trials):
            # gerate a random graph with the current number of verticies
            graph = generate_random_tsp_graph(size, edge_probability=0.5)
            exec_time = measure_tsp_execution_time(graph)
            exec_times.append(exec_time)

        # store all execution times for this given size
        all_times[size] = exec_times

        # take the average
        avg_time = np.mean(exec_times)
        avg_times.append(avg_time)

        # print the size of the graph and how long it took to execute
        print(f"Size: {size}, Execution time: {avg_time:.4f} seconds")

    # plot
    plt.figure(figsize=(10, 6))

    # scatter plot
    for size in sizes:
        jittered_size = [size + np.random.uniform(-0.1, 0.1) for _ in all_times[size]]
        plt.scatter(jittered_size, all_times[size], color='blue', alpha=0.6, label='Execution times' if size == sizes[0] else "")

    # plot average execution times
    plt.plot(sizes, avg_times, marker='o', color='red', label='Average execution time')

    plt.title("Traveling Salesman Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()


def test():
    """
    Tests the Traveling Salesman solver with known graphs and minimum costs 

    """
    # define the custom graphs and their known minimum costs
    test_cases = [
        (np.array([[0, 10, 15, 20],
                   [10, 0, 35, 25],
                   [15, 35, 0, 30],
                   [20, 25, 30, 0]]), 80),  # First test case

        (np.array([[0, 29, 20, 21],
                   [29, 0, 15, 17],
                   [20, 15, 0, 28],
                   [21, 17, 28, 0]]), 73)   # Second test case
    ]

    # Iterate over each test case and check the result
    for i, (graph, known_min_cost) in enumerate(test_cases, 1):
        print(f"Test case {i}:")

        # List of vertices (0, 1, 2, ..., n-1)
        vertices = list(range(len(graph)))

        # Solve TSP for the current graph
        best_path, min_cost = traveling_salesman(graph, vertices)

        print(f"Calculated min_cost: {min_cost}")
        print(f"Expected min_cost: {known_min_cost}")

        # Check if the calculated cost matches the known minimum cost
        if min_cost == known_min_cost:
            print(f"Test case {i} passed!\n")
        else:
            print(f"Test case {i} failed!\n")




# run performance test 
if __name__ == "__main__":
    test()
    test_tsp()
