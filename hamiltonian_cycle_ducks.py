# Hamiltonian Cycle Solver 

import time
import random
import networkx as nx
import matplotlib.pyplot as plt

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
def test_solver_performance():
    """
    This function tests the performance of the Hamiltonian cycle solver for randomly 
    generated graphs of different sizes.
    
    It records the execution time and plots to show an O(n!) complexity. 

    """
    sizes = [4, 5, 6, 7, 8, 9, 10] # sizes that represent the number of verticies 
    execution_times = [] # list to store the execution time for each graph sizer

    for size in sizes:
        # generate a random graph with the current number of verticies 
        graph = random_graph(size, edge_probability=0.5)
        # measure how long it takes to solve the Hamiltonian cycle problem 
        exec_time = measure_time(graph)
        # store the execution time 
        execution_times.append(exec_time)
        # print the size of the graph and how long it took to execute 
        print(f"Size: {size}, Execution time: {exec_time:.4f} seconds")

    # plot execution times 
    plt.plot(sizes, execution_times, marker='o')
    plt.title("Hamiltonian Cycle Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (seconds)")
    plt.show()

# run the whole program 
if __name__ == "__main__":
    test_solver_performance()
