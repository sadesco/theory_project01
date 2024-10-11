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
def hamilton_paths(graph, vertices):
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

#measures time it takes find all possible paths in a graph 
def measure_time(func, *args):
    start = time.time()
    end = time.time()
    return end - start

#plots all times measured 
def plot_times(graphs):
    sizes = []
    times = []
    for graph in graphs:
        




