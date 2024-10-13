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


def random(num_vertices, edge_probability=0.5):
    """
    Generates random graphs to test 

    """
    graph = nx.gnp_random_graph(num_vertices, edge_probability)
    matrix = nx.to_numpy_arry(graph, dtype=int)
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
        graph = random(size, edge_probability=0.5)
        execution = measure_time(graph)
        times.append(execution)
        print("Size: {}, Execution time: {:.4f} seconds".format(size, execution))
        
        plt.plot(sizes, times, marker='o')
        plt.title("Hamiltonian Path Solver Performance")
        plt.xlabel("Number of vertices")
        plt.ylabel("Execution time (seconds)")
        plt.show()
        
if __name__ == "__main__":
    plot_times()



