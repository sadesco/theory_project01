# Hamiltonian Cycle Solver 

import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def check(graph, path):
    """
    A Hamiltonian cycle starts and ends on the same vertex and visits every edge once. 

    This function takes a graph represented by an adjancency matrix and a proposed 
    path and it checks if said path is a Hamiltonian cycle. 

    """
    for i in range(len(path) - 1):
        # if there is no edge between two verticies, false is immediatley retuned 
        if graph[path[i]][path[i+1]] == 0:
            return False
        
    # check if the last vertex is connected to the first (condition for cycle)
    if graph[path[-1]][path[0]] == 0:
        return False
    
    # if all of these passed then return true 
    return True

def hamiltonian_cycles(graph, vertices):
    """
    Finds all Hamiltonian cycles in a graph by generating all possible paths
    of the verticies and checks if they form a Hamiltonian cycle. 

    """
    # Convert adjacency matrix back to a NetworkX graph to check connectivity
    G = nx.from_numpy_array(graph)
    
    # Check if the graph is connected
    if not nx.is_connected(G):
        return []  # Return no cycles if the graph is disconnected
    
    n = len(vertices) # number of verticies to loop through 
    cycles = [] # number of successful Hamiltonian cycles 
    
    # recursive function to generate all possible permutations of the verticies 
    def generate_path(path, index):

        # if there is a full path 
        if index == n:
            # check if the generated path is a hamiltonian cycle 
            if check(graph, path):
                # if it is valid - add this path to the list of cycles 
                cycles.append(path[:]) 
        
        # if there isn't a full path  
        else:
            # attempt at placing every remaining vertex at the current index 
            for i in range(index, n):
                path[index], path[i] = path[i], path[index]
                # recursively call to generate new paths 
                generate_path(path, index + 1)
                path[index], path[i] = path[i], path[index]
   
    # start generating paths
    generate_path(vertices, 0)

    # return the succesful paths 
    return cycles

def random_graph(num_vertices, edge_probability=0.5):
    """
    Generates a random graph using NetworkX (as was recommended in class).
    This function then takes that graph and converts it to an adj. matrix. 

    """
    # generate a random graph using G(n, p)
    # n is the number of verticies 
    # p is the probability of an edge exisiting between any pair of verticies 
    G = nx.gnp_random_graph(num_vertices, edge_probability)

    # convert to adjacency matrix 
    adj_matrix = nx.to_numpy_array(G, dtype=int)

    return adj_matrix

def measure_time(graph):
    """
    Measures the time taken by the above functions to find all possible
    Hamilton cycles in a given graph. 
     
    """
    # creates list of verticies 
    vertices = list(range(len(graph))) 

    # time and execute the function 
    start_time = time.time()
    cycles = hamiltonian_cycles(graph, vertices)
    end_time = time.time()

    # return time taken 
    return end_time - start_time

# OPTIMIZATION, DO SEVERAL TESTS FOR EACH SIZE OF GRAPH AND TAKE AN AVERAGE OF EXECUTION TIMES
def test_solver_performance():
    """
    This function tests the performance of the Hamiltonian cycle solver for randomly 
    generated graphs of different sizes.
    
    It records the execution time and plots to show an O(n!) complexity. 

    """
    sizes = [4, 5, 6, 7, 8, 9, 10] # sizes that represent the number of verticies 
    avg_times = [] 
    trials = 5 # number of trials per graph size 

    for size in sizes:
        exec_times = []
        for _ in range(trials): 
            # generate a random graph with the current number of verticies 
            graph = random_graph(size, edge_probability=0.5)
            exec_time = measure_time(graph)
            exec_times.append(exec_time)
        
        # take the average
        avg_time = np.mean(exec_times)
        avg_times.append(avg_time)

        # print the size of the graph and how long it took to execute 
        print(f"Size: {size}, Execution time: {avg_time:.4f} seconds")

    # plot 
    plt.plot(sizes, avg_times, marker='o')
    plt.title("Hamiltonian Cycle Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Average execution time (seconds)")
    plt.show()

def test():
    """
    Test the Hamiltonian cycle solver with known test cases: one that should have a Hamiltonian cycle
    and one that should not.
    """
    # Example 1: A complete graph (which should have a Hamiltonian cycle)
    complete_graph = nx.complete_graph(5)
    complete_matrix = nx.to_numpy_array(complete_graph, dtype=int)
    complete_cycles = measure_time(complete_matrix)
    if complete_cycles:
        print("Test 1 Passed: Hamiltonian cycle found in complete graph.")
    else:
        print("Test 1 Failed: No Hamiltonian cycle found in complete graph.")

    # Example 2: A disconnected graph (which should not have a Hamiltonian cycle)
    disconnected_graph = nx.Graph()
    disconnected_graph.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components
    disconnected_matrix = nx.to_numpy_array(disconnected_graph, dtype=int)
    disconnected_cycles = measure_time(disconnected_matrix)
    if not disconnected_cycles:
        print("Test 2 Passed: No Hamiltonian cycle found in disconnected graph.")
    else:
        print("Test 2 Failed: Hamiltonian cycle found in disconnected graph.")

# run the whole program 
if __name__ == "__main__":
    test() 
    test_solver_performance()
