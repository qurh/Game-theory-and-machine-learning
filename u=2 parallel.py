import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from multiprocessing import Pool, cpu_count

def create_er_graph_with_partner_switching(n, m, c_fraction, w, u, alpha, max_iterations=1000000000000):
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
    
    iteration = 0
    while iteration < max_iterations:
        cd_edges = [(i, j) for i in range(n) for j in neighbors[i] if strategies[i] != strategies[j] and i < j]
        if not cd_edges:
            print("No cd edges")
            return strategies.count('C') / n, True  # Simulation completed successfully
        
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
        
        iteration += 1

    print(f"Max iterations ({max_iterations}) reached without convergence")
    return strategies.count('C') / n, False  # Simulation did not complete

def run_single_simulation(args):
    n, m, c_fraction, w, u, alpha = args
    result, completed = create_er_graph_with_partner_switching(n, m, c_fraction, w, u, alpha)
    return result, completed

def run_simulations(n, m, c_fraction, alpha, w, u, num_simulations):
    args_list = [(n, m, c_fraction, w, u, alpha)] * num_simulations
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_single_simulation, args_list)
    
    completed_results = [r for r, c in results if c]
    completion_rate = len(completed_results) / num_simulations
    
    if completed_results:
        avg_fraction = np.mean(completed_results)
    else:
        avg_fraction = None
    
    return avg_fraction, completion_rate

if __name__ == '__main__':
    # Simulation parameters
    n = 1000  # Number of nodes
    m = 5000  # Number of edges
    c_fraction = 0.5  # Initial fraction of cooperators
    alpha = 30  # Intensity of selection
    num_simulations = 100  # Number of simulations per data point
    u = 0.2  # Fixed cost-to-benefit ratio

    # Create parameter array for w
    w_values = np.linspace(0.3,0.4,1)

    results = []
    completion_rates = []
    for w in tqdm(w_values, desc="Running simulations"):
        avg_fraction, completion_rate = run_simulations(n, m, c_fraction, alpha, w, u, num_simulations)
        results.append(avg_fraction)
        completion_rates.append(completion_rate)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot fraction of cooperators
    ax1.plot(w_values, results, marker='o', linestyle='-')
    ax1.set_xlabel('Strategy updating probability (w)')
    ax1.set_ylabel('Fraction of cooperators')
    ax1.set_title(f'Cooperators vs Strategy updating probability (u = {u})')
    ax1.set_xlim(0.27, 0.31)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)

    # Plot completion rates
    ax2.plot(w_values, completion_rates, marker='s', linestyle='-', color='red')
    ax2.set_xlabel('Strategy updating probability (w)')
    ax2.set_ylabel('Simulation Completion Rate')
    ax2.set_title('Simulation Completion Rate vs Strategy updating probability')
    ax2.set_xlim(0.27, 0.31)
    ax2.set_ylim(-0.01, 1.01)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print average completion rate
    avg_completion_rate = np.mean(completion_rates)
    print(f"Average completion rate across all simulations: {avg_completion_rate:.2%}")