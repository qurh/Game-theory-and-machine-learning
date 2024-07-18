# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:58:19 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count

def create_er_graph_with_partner_switching(n, m, c_fraction, w, u, alpha):
    # Initialize strategies
    strategies = ['C'] * int(n * c_fraction) + ['D'] * (n - int(n * c_fraction))
    random.shuffle(strategies)
    
    # Create ER graph
    p = 2 * m / (n * (n - 1))  # Probability of edge creation
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                edges.add((i, j))
    
    # Adjust number of edges to match m
    while len(edges) > m:
        edges.remove(random.choice(list(edges)))
    while len(edges) < m:
        i, j = random.sample(range(n), 2)
        if i != j and (i, j) not in edges and (j, i) not in edges:
            edges.add((min(i, j), max(i, j)))
    
    # Define payoff matrix
    payoff = {
        ('C', 'C'): (1,1),
        ('C', 'D'): (0,1 + u),
        ('D', 'C'): (1 + u,0),
        ('D', 'D'): (u,u)
    }
    
    def calculate_payoff(node):
        return sum(payoff[strategies[node], strategies[neighbor]][0] for neighbor in neighbors[node])
    
    def fermi_function(pi, pj):
       diff = np.clip(alpha * (pi - pj), -700, 700)
       return 1 / (1 + np.exp(diff))
    
    # Create neighbor list
    neighbors = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    
    while True:
        cd_edges = [(i, j) for i in range(n) for j in neighbors[i] if strategies[i] != strategies[j] and i < j]
        if not cd_edges:
            break
        
        # Randomly select one cd_edge
        i, j = random.choice(cd_edges)
        
        if random.random() < w:
            # Strategy updating
            pi = calculate_payoff(i)
            pj = calculate_payoff(j)
            if random.random() < fermi_function(pi, pj):
                strategies[i] = strategies[j]
            else:
                strategies[j] = strategies[i]
        else:
            # Partner switching
            c_node = i if strategies[i] == 'C' else j
            neighbors[i].remove(j)
            neighbors[j].remove(i)
            available_nodes = set(range(n)) - set([c_node]) - neighbors[c_node]
            if available_nodes:
                new_neighbor = random.choice(list(available_nodes))
                neighbors[c_node].add(new_neighbor)
                neighbors[new_neighbor].add(c_node)
    
    return strategies.count('C') / n

def run_single_simulation(args):
    n, m, c_fraction, w, u, alpha = args
    return create_er_graph_with_partner_switching(n, m, c_fraction, w, u, alpha)

def run_simulations(n, m, c_fraction, alpha, variable_param, fixed_param, param_values, num_simulations):
    results = []
    for value in tqdm(param_values, desc=f"Running simulations for {variable_param}"):
        if variable_param == 'u':
            u, w = value, fixed_param
        else:
            w, u = value, fixed_param
        
        args_list = [(n, m, c_fraction, w, u, alpha)] * num_simulations
        
        with Pool(processes=cpu_count()) as pool:
            cooperator_fractions = pool.map(run_single_simulation, args_list)
        
        avg_fraction = np.mean(cooperator_fractions)
        results.append(avg_fraction)
    
    return results

if __name__ == '__main__':
    # Simulation parameters
    n = 1000  # Number of nodes
    m = 5000  # Number of edges
    c_fraction = 0.5  # Initial fraction of cooperators
    alpha = 30  # Intensity of selection
    num_simulations = 1000  # Number of simulations per data point

    # Create parameter arrays
    u_values = np.linspace(0, 0.99, 50)
    w_values = np.linspace(0, 0.99, 30)

    # Run simulations for left panel (varying u)
    w_values_left = [0, 0.05, 0.1, 0.5]
    results_left = {}
    for w in w_values_left:
        print(f"\nSimulating left panel for w = {w}")
        results_left[w] = run_simulations(n, m, c_fraction, alpha, 'u', w, u_values, num_simulations)

    # Run simulations for right panel (varying w)
    u_values_right = [0, 0.01, 0.2, 0.8]
    results_right = {}
    for u in u_values_right:
        print(f"\nSimulating right panel for u = {u}")
        results_right[u] = run_simulations(n, m, c_fraction, alpha, 'w', u, w_values, num_simulations)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel
    markers = ['s', 'o', '^', 'v']
    for (w, result), marker in zip(results_left.items(), markers):
        ax1.plot(u_values, result, marker=marker, linestyle='', label=f'w = {w}')
    ax1.set_xlabel('Cost-to-benefit ratio, u')
    ax1.set_ylabel('Fraction of cooperators')
    ax1.legend()
    ax1.set_title('Cooperators vs Cost-to-benefit ratio')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.01, 1.01)

    # Right panel
    for (u, result), marker in zip(results_right.items(), markers):
        ax2.plot(w_values, result, marker=marker, linestyle='', label=f'u = {u}')
    ax2.set_xlabel('w')
    ax2.set_ylabel('Fraction of cooperators')
    ax2.legend()
    ax2.set_title('Cooperators vs Strategy updating probability')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.01, 1.01)

    plt.tight_layout()
    plt.show()