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



import pickle

def plot_figure_1(parameter_list_of_experiment_rho_vs_w_with_fixed_u, u_list, parameter_list_of_experiment_rho_vs_u_with_fixed_w, w_list):
    print("Figure 1:")
    print("Loading Data:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    marker_list = ['s', 'o', '^', 'v']
    color_list = ['k','r','g','b']
    
    print(f"u_list: {u_list}")
    print(f"w_list: {w_list}")
    
    # Left panel: Cooperators vs Cost-to-benefit ratio
    for w, marker, color in tqdm.tqdm(list(zip(w_list, marker_list, color_list)), "Left Panel"):
        tmp_u_list, tmp_cooperative_ratio_list = load_experiment_rho_vs_u_with_fixed_w(parameter_list_of_experiment_rho_vs_u_with_fixed_w, w)
        if tmp_u_list:
            ax1.plot(tmp_u_list, tmp_cooperative_ratio_list, marker=marker, linestyle='-',color=color, label=f'w = {w} (True)')
            print(f"True data for w={w}: {len(tmp_u_list)} points")
        else:
            print(f"No true data available for w={w}")

    # Right panel: Cooperators vs Strategy updating probability
    for u, marker ,color in tqdm.tqdm(list(zip(u_list, marker_list, color_list)), "Right Panel"):
        tmp_w_list, tmp_cooperative_ratio_list = load_experiment_rho_vs_w_with_fixed_u(parameter_list_of_experiment_rho_vs_w_with_fixed_u, u)
        if tmp_w_list:
            ax2.plot(tmp_w_list, tmp_cooperative_ratio_list, marker=marker, linestyle='-',color=color, label=f'u = {u} (True)')
            print(f"True data for u={u}: {len(tmp_w_list)} points")
        else:
            print(f"No true data available for u={u}")

    # Load and plot GCN results
    try:
        with open('parameter_only_results_rho_vs_u.pkl', 'rb') as f:
            gcn_results_rho_vs_u = pickle.load(f)
        with open('parameter_only_results_rho_vs_w.pkl', 'rb') as f:
            gcn_results_rho_vs_w = pickle.load(f)

        print(f"Total GCN results for rho_vs_u: {len(gcn_results_rho_vs_u)}")
        print(f"Total GCN results for rho_vs_w: {len(gcn_results_rho_vs_w)}")

        # Plot GCN results for left panel
        gcn_w_values = sorted(set([round(result[1], 5) for result in gcn_results_rho_vs_u]))
        i = 0
        for w in gcn_w_values:
            u_values = [round(result[0], 5) for result in gcn_results_rho_vs_u if round(result[1], 5) == w]
            rho_values = [round(result[2], 5) for result in gcn_results_rho_vs_u if round(result[1], 5) == w]
            if u_values:
                ax1.plot(u_values, rho_values, linestyle='--',color=color_list[i], label=f'w = {w} (NN)')
                print(f"GCN data for w={w}: {len(u_values)} points")
                i += 1
            else:
                print(f"No GCN data available for w={w}")
            

        # Plot GCN results for right panel
        gcn_u_values = sorted(set([round(result[0], 5) for result in gcn_results_rho_vs_w]))
        i = 0
        for u in gcn_u_values:
            w_values = [round(result[1], 5) for result in gcn_results_rho_vs_w if round(result[0], 5) == u]
            rho_values = [round(result[2], 5) for result in gcn_results_rho_vs_w if round(result[0], 5) == u]
            if w_values:
                ax2.plot(w_values, rho_values, linestyle='--',color = color_list[i], label=f'u = {u} (NN)')
                print(f"GCN data for u={u}: {len(w_values)} points")
                i += 1
            else:
                print(f"No GCN data available for u={u}")
            
    except FileNotFoundError:
        print("GCN result files not found. Skipping GCN plots.")

    # Set labels and titles
    ax1.set_xlabel('Cost-to-benefit ratio, u')
    ax1.set_ylabel('Fraction of cooperators')
    ax1.set_title('Cooperators vs Cost-to-benefit ratio')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.01, 1.01)
    ax1.legend()

    ax2.set_xlabel('Strategy updating probability, w')
    ax2.set_ylabel('Fraction of cooperators')
    ax2.set_title('Cooperators vs Strategy updating probability')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.01, 1.01)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    print("Finish")

def main():
    u_list = [0,0.01,0.2,0.8]
    w_list = [0,0.05,0.1,0.5]
    parameter_list_of_experiment_rho_vs_w_with_fixed_u = get_parameter_list_of_experiment_rho_vs_w_with_fixed_u(u_list)
    parameter_list_of_experiment_rho_vs_u_with_fixed_w = get_parameter_list_of_experiment_rho_vs_u_with_fixed_w(w_list)
    plot_figure_1(
        parameter_list_of_experiment_rho_vs_w_with_fixed_u, u_list,
        parameter_list_of_experiment_rho_vs_u_with_fixed_w, w_list
    )


if __name__ == "__main__":
    main()

