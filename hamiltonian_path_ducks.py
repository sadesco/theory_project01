import time
import random
import networkx as nx
import matplotlib.pyplot as plt

#Checks if path is an hamiltonian path
def is_hamiltonian_path(graph, path):
    for i in range(len(path) - 1):
        #if no edge between two vertices return false
        if graph[path[i]][path[i+1]] == 0:
            return False
    return True

#Finds all paths in a graph by generating all possible paths
def hamiltonian_paths(graph, vertices):
    n = len(vertices)  
    paths = []  

    #Generate all possible variations of the vertices
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
def random(num_vertices, edge_probability=0.5):
    graph = nx.gnp_random_graph(num_vertices, edge_probability)
    matrix = nx.to_numpy_arry(graph, dtype=int)
    return matrix

#measures time it takes find all possible paths in a graph 
def measure_time(graph):
    vertices = list(range(len(graph))) 
    start = time.time()
    hamiltonian_paths(graph, vertices)
    end = time.time()

    # return time taken 
    return end - start

#plots performance - all times measured 
def plot_times():
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



