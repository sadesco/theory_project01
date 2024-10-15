import time
import random
import networkx as nx
import matplotlib.pyplot as plt

def is_hamiltonian_path(graph, path):
    """
    This function takes a graph represented by an adjancency matrix and a proposed 
    path and it checks if said path is a Hamiltonian path. 

    """
    for i in range(len(path) - 1):
        #if no edge between two vertices return false
        if graph[path[i]][path[i+1]] == 0:
            return False
    return True

def hamiltonian_paths(graph, vertices):
    """
    Finds all hamiltonian paths in a graph by generating all possible 
    paths. 
    
    """
    n = len(vertices)  
    paths = []  

    # generate all possible variations of the vertices
    def generate(path, index):
        if index == n:
            if is_hamiltonian_path(graph, path):
                paths.append(path[:])
        else:
            for i in range(index, n):
                path[index], path[i] = path[i], path[index]
                generate(path, index + 1)
                path[index], path[i] = path[i], path[index]

    generate(vertices, 0)
    return paths


#generate random graphs to test
def generate_random(num_vertices, edge_probability=0.5):
    graph = nx.gnp_random_graph(num_vertices, edge_probability)
    matrix = nx.to_numpy_array(graph, dtype=int)
    return matrix

def measure_time(graph):
    """
    Measures the time it takes to generate all Hamiltonian paths in a graph
    
    """
    vertices = list(range(len(graph))) 
    start = time.time()
    hamiltonian_paths(graph, vertices)
    end = time.time()

    # return time taken 
    return end - start

def plot_times():
    """
    Plots and tests the performance of the solver
    
    """
    #sizes that represent the number of vertices 
    sizes = [4, 5, 6, 7, 8, 9, 10] 
    times = []
    
    for size in sizes: 
        graph = generate_random(size, edge_probability=0.5)
        execution = measure_time(graph)
        times.append(execution)
        print("Size: {}, Execution time: {:.4f} seconds".format(size, execution))
        
    plt.plot(sizes, times, marker='o')
    plt.title("Hamiltonian Path Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (seconds)")
    plt.show()

def test():
    """
    Test the solver with a graph that has a Hamiltonian path and one that does not.
    """
    # Example 1: A complete graph (which should have a Hamiltonian path)
    complete_graph = nx.complete_graph(5)
    complete_matrix = nx.to_numpy_array(complete_graph, dtype=int)
    _, complete_paths = measure_time(complete_matrix)
    if complete_paths:
        print("Test 1 Passed: Hamiltonian path found in complete graph.")
    else:
        print("Test 1 Failed: No Hamiltonian path found in complete graph.")

    # Example 2: A disconnected graph (which should not have a Hamiltonian path)
    disconnected_graph = nx.Graph()
    disconnected_graph.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components
    disconnected_matrix = nx.to_numpy_array(disconnected_graph, dtype=int)
    _, disconnected_paths = measure_time(disconnected_matrix)
    if not disconnected_paths:
        print("Test 2 Passed: No Hamiltonian path found in disconnected graph.")
    else:
        print("Test 2 Failed: Hamiltonian path found in disconnected graph.")

if __name__ == "__main__":
    test()
    plot_times()



