# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:02:25 2024

@author: SAM
"""

import simulation

import numpy
import matplotlib.pyplot as plt
import multiprocessing
import tqdm

def get_parameter_list_for_scatter():
    n = 1000
    m = 5000
    rho = 0.5
    alpha = 30
    type_of_random_graph = "ER"
    max_iter = -1
    number_of_grid = 50
    u_list = [float(round(x, 2)) for x in numpy.linspace(0.77, 0.99, 12)]
    w_list = [float(round(x, 2)) for x in numpy.linspace(0.01, 0.99, number_of_grid)]
    parameter_list = [(n, m, u, w, rho, alpha, type_of_random_graph, max_iter) for u in u_list for w in w_list]
    return parameter_list

def load_experiment_for_scatter(parameter_list):
    u_list = []
    w_list = []
    rho_list = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for (n, m, u, w, rho, alpha, type_of_random_graph, _) in tqdm.tqdm(parameter_list, "Loading Data"):
        game_list = simulation.load(pool, (n, m, u, w, rho, alpha, type_of_random_graph))
        u_list.append(u)
        w_list.append(w)
        rho_list.append(simulation.get_mean_cooperative_ratio(pool, game_list))
    pool.close()
    pool.join()
    return u_list, w_list, rho_list

def plot_scatter(u_list, w_list, rho_list):
    print("Plotting Scatter:")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(u_list, w_list, c=rho_list, cmap='viridis', s=20)
    plt.colorbar(scatter, label='Fraction of cooperators')
    plt.xlabel('Cost-to-benefit ratio, u')
    plt.ylabel('Strategy updating probability, w')
    plt.title('Fraction of Cooperators')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    print("Finish")

def main():
    number_of_trial = 1000
    option = "write"
    parameter_list = get_parameter_list_for_scatter()
    simulation.run_simulation(parameter_list, number_of_trial, option)
    u_list, w_list, rho_list = load_experiment_for_scatter(parameter_list)
    plot_scatter(u_list, w_list, rho_list)

if __name__ == "__main__":
    main()
    