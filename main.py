import simulation

import numpy
import matplotlib.pyplot as plt
import multiprocessing
import tqdm

def get_parameter_list_of_experiment_rho_vs_w_with_fixed_u(u_list):
    n = 1000
    m = 5000
    rho = 0.5
    alpha = 30
    type_of_random_graph = "ER"
    max_iter = -1
    number_of_grid = 51
    parameter_list = [(n, m, u, w, rho, alpha, type_of_random_graph, max_iter) for u in u_list for w in [float(round(x,5)) for x in numpy.linspace(0, 1, number_of_grid)]]
    return parameter_list
def load_experiment_rho_vs_w_with_fixed_u(parameter_list, u_interested):
    w_list = []
    rho_list = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for (n, m, u, w, rho, alpha, type_of_random_graph, _) in parameter_list:
        if u == u_interested:
            game_list = simulation.load(pool, (n, m, u, w, rho, alpha, type_of_random_graph))
            w_list.append(w)
            rho_list.append(simulation.get_mean_cooperative_ratio(pool, game_list))
    pool.close()
    pool.join()
    return w_list, rho_list
def get_parameter_list_of_experiment_rho_vs_u_with_fixed_w(w_list):
    n = 1000
    m = 5000
    rho = 0.5
    alpha = 30
    type_of_random_graph = "ER"
    max_iter = -1
    number_of_grid = 51
    parameter_list = [(n, m, u, w, rho, alpha, type_of_random_graph, max_iter) for w in w_list for u in [float(round(x,5)) for x in numpy.linspace(0, 1, number_of_grid)]]
    return parameter_list
def load_experiment_rho_vs_u_with_fixed_w(parameter_list, w_interested):
    u_list = []
    rho_list = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for (n, m, u, w, rho, alpha, type_of_random_graph, _) in parameter_list:
        if w == w_interested:
            game_list = simulation.load(pool, (n, m, u, w, rho, alpha, type_of_random_graph))
            u_list.append(u)
            rho_list.append(simulation.get_mean_cooperative_ratio(pool, game_list))
    pool.close()
    pool.join()
    return u_list, rho_list
def plot_figure_1(parameter_list_of_experiment_rho_vs_w_with_fixed_u, u_list, parameter_list_of_experiment_rho_vs_u_with_fixed_w, w_list):
    print("Figure 1:")
    print("Loading Data:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    marker_list = ['s', 'o', '^', 'v']
    tmp_u_list = []
    tmp_cooperative_ratio_list = []
    for w, marker in tqdm.tqdm(list(zip(w_list, marker_list)), "Left Panal"):
        tmp_u_list, tmp_cooperative_ratio_list = load_experiment_rho_vs_u_with_fixed_w(parameter_list_of_experiment_rho_vs_u_with_fixed_w, w)
        ax1.plot(tmp_u_list, tmp_cooperative_ratio_list, marker=marker, linestyle='', label=f'w = {w}')
    ax1.set_xlabel('Cost-to-benefit ratio, u')
    ax1.set_ylabel('Fraction of cooperators')
    ax1.legend()
    ax1.set_title('Cooperators vs Cost-to-benefit ratio')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.01, 1.01)
    
    tmp_w_list = []
    tmp_cooperative_ratio_list = []
    for u, marker in tqdm.tqdm(list(zip(u_list, marker_list)), "Right Panal"):
        tmp_w_list, tmp_cooperative_ratio_list = load_experiment_rho_vs_w_with_fixed_u(parameter_list_of_experiment_rho_vs_w_with_fixed_u, u)
        ax2.plot(tmp_w_list, tmp_cooperative_ratio_list, marker=marker, linestyle='', label=f'u = {u}')
    ax2.set_xlabel('Strategy updating probability, w')
    ax2.set_ylabel('Fraction of cooperators')
    ax2.legend()
    ax2.set_title('Cooperators vs Strategy updating probability')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.01, 1.01)

    plt.tight_layout()
    plt.show()
    print("Finish")
def main():
    u_list = [0,0.01,0.2,0.8]
    w_list = [0,0.05,0.1,0.5]
    number_of_trial = 1000
    option = "write"
    parameter_list_of_experiment_rho_vs_w_with_fixed_u = get_parameter_list_of_experiment_rho_vs_w_with_fixed_u(u_list)
    parameter_list_of_experiment_rho_vs_u_with_fixed_w = get_parameter_list_of_experiment_rho_vs_u_with_fixed_w(w_list)
    # simulation.run_simulation(parameter_list_of_experiment_rho_vs_w_with_fixed_u, number_of_trial, option)
    # simulation.run_simulation(parameter_list_of_experiment_rho_vs_u_with_fixed_w, number_of_trial, option)
    plot_figure_1(\
        parameter_list_of_experiment_rho_vs_w_with_fixed_u, u_list,\
        parameter_list_of_experiment_rho_vs_u_with_fixed_w, w_list\
    )
if __name__ == "__main__":
    main()