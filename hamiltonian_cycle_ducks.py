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
def test_performance():
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
            # generate a random graph with the current number of verticies 
            graph = random_graph(size, edge_probability=0.5)
            exec_time = measure_time(graph)
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

    # scatter plot, use jittered to scatter them so you can see all points
    for size in sizes:
        jittered_size = [size + np.random.uniform(-0.1, 0.1) for _ in all_times[size]]
        plt.scatter(jittered_size, all_times[size], color='blue', alpha=0.6, label='Execution times' if size == sizes[0] else "")
    
    # plot average execution times 
    plt.plot(sizes, avg_times, marker='o', color='red', label='Average execution time')

    plt.title("Hamiltonian Cycle Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

def test():
    """ 
     Tests for success of the program 
     
    """
    # Test graph with a Hamiltonian cycle
    # Graph: 0-1-2-3-0
    graph_with_cycle = np.array([[0, 1, 0, 1],
                                  [1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [1, 0, 1, 0]])
    vertices_with_cycle = list(range(len(graph_with_cycle)))
    cycles = hamiltonian_cycles(graph_with_cycle, vertices_with_cycle)

    print("Testing graph with a Hamiltonian cycle:")
    if cycles:
        print(f"Found Hamiltonian cycles: {cycles}")
    else:
        print("No Hamiltonian cycle found, but there should be one!")

    # Test graph without a Hamiltonian cycle
    # Graph: 0-1, 2 disconnected
    graph_without_cycle = np.array([[0, 1, 0],
                                     [1, 0, 0],
                                     [0, 0, 0]])
    vertices_without_cycle = list(range(len(graph_without_cycle)))
    cycles = hamiltonian_cycles(graph_without_cycle, vertices_without_cycle)

    print("Testing graph without a Hamiltonian cycle:")
    if cycles:
        print("Found Hamiltonian cycle, but there shouldn't be one!")
    else:
        print("Correctly identified no Hamiltonian cycle.")

# run the whole program 
if __name__ == "__main__":
    test() 
    test_performance()
