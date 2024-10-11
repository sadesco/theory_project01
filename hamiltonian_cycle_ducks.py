import itertools
import random
import time
import matplotlib.pyplot as plt

def is_hamiltonian_cycle(graph, path):
    """Check if a given path forms a Hamiltonian cycle in the graph"""
    n = len(graph)
    for i in range(n):
        if graph[path[i]][path[(i + 1) % n]] == 0:
            return False
    return True

def brute_force_hamiltonian(graph):
    """Brute force solver for Hamiltonian cycle"""
    n = len(graph)
    vertices = list(range(n))
    for perm in itertools.permutations(vertices[1:]):  # Fix first vertex
        cycle = [vertices[0]] + list(perm)
        if is_hamiltonian_cycle(graph, cycle):
            return cycle
    return None

def time_hamiltonian(graph):
    """Time the Hamiltonian cycle solver"""
    start_time = time.time()
    cycle = brute_force_hamiltonian(graph)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1e6  # Time in microseconds
    return cycle, elapsed_time

