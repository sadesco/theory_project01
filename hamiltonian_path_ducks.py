import time
import numpy as np
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
    Plots and tests the performance of the hamiltonian path solver

    """
    sizes = [4, 5, 6, 7, 8, 9, 10] # sizes that represent the number of verticies
    avg_times = []
    all_times = {}
    trials = 15 # number of trials per graph size

    for size in sizes:
        exec_times = []
        for _ in range(trials):
            # generate a random graph with the current number of verticies
            graph = generate_random(size, edge_probability=0.5)
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

    # scatter plot
    for size in sizes:
        jittered_size = [size + np.random.uniform(-0.1, 0.1) for _ in all_times[size]]
        plt.scatter(jittered_size, all_times[size], color='blue', alpha=0.6, label='Execution times' if size == sizes[0] else "")

    # plot average execution times
    plt.plot(sizes, avg_times, marker='o', color='red', label='Average execution time')

    plt.title("Hamiltonian Path Solver Performance")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

def test():
    """
    Tests the Hamiltonian path solver with known graphs.
    It checks a graph with a Hamiltonian path and a graph without one.
    """

    # Test graph with a Hamiltonian path
    # Graph: 0-1-2-3
    graph_with_path = np.array([[0, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0]])
    vertices_with_path = list(range(len(graph_with_path)))
    paths = hamiltonian_paths(graph_with_path, vertices_with_path)

    print("Testing graph with a Hamiltonian path:")
    if paths:
        print(f"Found Hamiltonian paths: {paths}")
    else:
        print("No Hamiltonian path found, but there should be one!")

    # Test graph without a Hamiltonian path
    # Graph: 0-1, 2 disconnected
    graph_without_path = np.array([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, 0]])
    vertices_without_path = list(range(len(graph_without_path)))
    paths = hamiltonian_paths(graph_without_path, vertices_without_path)

    print("Testing graph without a Hamiltonian path:")
    if paths:
        print("Found Hamiltonian path, but there shouldn't be one!")
    else:
        print("Correctly identified no Hamiltonian path.")


if __name__ == "__main__":
    test()
    plot_times()