from partner_switching_model import *
import itertools
import multiprocessing
import numpy
import os
import pickle
import tqdm

def run_simulation(parameter_list: list[int, int, float, float, float, float, str, int], number_of_trial, option) -> None:
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    match option:
        case "write":
            for n, m, u, w, rho, alpha, type_of_random_graph, max_iteration in tqdm.tqdm(parameter_list):
                game_list = get_final_state_list(pool, n, m, u, w, rho, alpha, type_of_random_graph, max_iteration, number_of_trial)
                save(game_list, saving_method="write")
        case "append":
            for n, m, u, w, rho, alpha, type_of_random_graph, max_iteration in tqdm.tqdm(parameter_list):
                game_list = get_final_state_list(pool, n, m, u, w, rho, alpha, type_of_random_graph, max_iteration, number_of_trial)
                save(game_list, saving_method="append")
        case "set":
            for n, m, u, w, rho, alpha, type_of_random_graph, max_iteration in tqdm.tqdm(parameter_list):
                appended_number_of_trial = number_of_trial - get_number_of_trial(n, m, u, w, rho, alpha, type_of_random_graph)
                if appended_number_of_trial <= 0:
                    continue
                game_list = get_final_state_list(pool, n, m, u, w, rho, alpha, type_of_random_graph, max_iteration, appended_number_of_trial)
                save(game_list, saving_method="append")             
    pool.close()
    pool.join()
    return
def get_final_state_list(pool, n: int, m: int, u: float, w: float, rho: float, alpha: float, type_of_random_graph: str, max_iteration: int, number_of_trial: int) -> list[partner_switching_model]:
    """
    run simulations parallelly and return final states
    
    Args:
        N (int): number of nodes
        M (int): number of edges
        U (float): cost to benefit ratio
        W (float): strategy update rate
        Rho (float): initial cooperative ratio
        Alpha (float): Fermi coefficient
        Type_of_random_graph (str): type of random graph
        Max_iteration (int): maximum of iteration
        number_of_trial (int): number of trials
    Returns:
        list[partner_switching_model]: list of games in final states
    """
    # arg_list = [(n, m, u, w, rho, alpha, type_of_random_graph, max_iteration, random.random()) for _ in range(number_of_trial)]
    # game_list = pool.map(\
    #     __get_final_state,\
    #     arg_list\
    # )
    # return game_list
    
    arg_list = [(n, m, u, w, rho, alpha, type_of_random_graph, max_iteration)] * number_of_trial
    game_list = pool.map(\
        __get_final_state,\
        arg_list\
    )
    return game_list
def __get_final_state(arg):
    n, m, u, w, rho, alpha, type_of_random_graph, max_iteration = arg
    return partner_switching_model("new", n, m, u, w, rho, alpha, type_of_random_graph).get_final_state(max_iteration)    
def __get_cooperative_ratio(game:partner_switching_model):
    return game.cooperative_ratio
def get_cooperative_ratio_list(pool, game_list: list[partner_switching_model]):
    cooperative_ratio_list = pool.map(\
        __get_cooperative_ratio,\
        game_list\
        )
    return cooperative_ratio_list
def get_mean_cooperative_ratio(pool, game_list: list[partner_switching_model]):
    cooperative_ratio_list = get_cooperative_ratio_list(pool, game_list)
    if len(cooperative_ratio_list) == 0:
        return None
    return numpy.mean(cooperative_ratio_list)
def get_time_resolved_game(n: int, m: int, u: float, w: float, rho: float, alpha: float, type_of_random_graph: str, max_iteration: int) -> list[partner_switching_model]:
    game_list = itertools.islice(partner_switching_model("new", n, m, u, w, rho, alpha, type_of_random_graph), max_iteration)
    return game_list
def save(game_list : list[partner_switching_model], saving_method=None, file_name=None, folder_name=None) -> None:
    available_modes = ["append", "write"]
    if saving_method is None:
        saving_method = "append"
    elif saving_method not in available_modes:
        raise Exception("unknown mode: available modes are " + ", ".join(available_modes) + ".")
    N = game_list[0]._number_of_node
    M = game_list[0]._number_of_edge
    U = game_list[0]._cost_to_benefit_ratio
    W = game_list[0]._strategy_update_rate
    Rho = game_list[0]._initial_cooperative_ratio
    Alpha = game_list[0]._fermi_coefficient
    Type_of_random_graph = game_list[0]._type_of_random_graph
    if file_name is None:
        file_name = "N_" + str(N)\
                    + "_M_" + str(M)\
                    + "_U_" + str(U)\
                    + "_W_" + str(W)\
                    + "_Rho_" + str(Rho)\
                    + "_Alpha_" + str(Alpha)\
                    + "_RG_" + Type_of_random_graph\
                    + ".pickle"
    if folder_name is None:
        folder_name = "data"
    
    name = folder_name + "/" + file_name
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(name):
        with open(name, "wb") as file:
            pass
        
    field_name = ["neighbor_list", "strategy_list"]
    data_list = [{field_name[0] : g._neighbor_list, field_name[1] : g._strategy_list} for g in game_list]
    match saving_method:
        case "write":
            with open(name, "wb") as file:
                pickle.dump(data_list, file)
        case "append":
            with open(name, "ab") as file:
                pickle.dump(data_list, file)
def load(pool, arg_tuple, folder_name = None, loading_data_method = None) -> list[partner_switching_model]:
    available_methods = ["by_parameter", "by_filename"]
    if loading_data_method is None:
        loading_data_method = "by_parameter"
    elif loading_data_method not in available_methods:
        raise Exception("unknown loading data mode: available method are " + ", ".join(available_methods) + ".")
    if folder_name is None:
        folder_name = "data"
    
    match loading_data_method:
        case "by_parameter":
            n, m, u, w, rho, alpha, type_of_random_graph = arg_tuple
            file_name = "N_" + str(n)\
                    + "_M_" + str(m)\
                    + "_U_" + str(u)\
                    + "_W_" + str(w)\
                    + "_Rho_" + str(rho)\
                    + "_Alpha_" + str(alpha)\
                    + "_RG_" + type_of_random_graph\
                    + ".pickle"
        case "by_filename":
            folder_name, file_name = arg_tuple
    
    name = folder_name + "/" + file_name
    if not os.path.isdir(folder_name):
        raise Exception("folder does not exist")
    if not os.path.exists(name):
        return []
    
    with open(name, "rb") as file:
        data_list = pickle.load(file)
    match loading_data_method:
        case "by_parameter":
            parameter_list = [("load_all", n, m, u, w, rho, alpha, type_of_random_graph, data["neighbor_list"], data["strategy_list"]) for data in data_list]
            game_list =  pool.map(\
                __load_game,\
                parameter_list\
            )
        case "by_filename":
            parameter_list = [("load", data["neighbor_list"], data["strategy_list"]) for data in data_list]
            game_list =  pool.map(\
                __load_game,\
                parameter_list\
            )
    return game_list
def __load_game(parameters):
    return partner_switching_model(*parameters)
def get_number_of_trial(*args, loading_data_method = None):
    available_methods = ["by_parameter", "by_filename"]
    if loading_data_method is None:
        loading_data_method = "by_parameter"
    elif loading_data_method not in available_methods:
        raise Exception("unknown loading data mode: available method are " + ", ".join(available_methods) + ".")
    
    match loading_data_method:
        case "by_parameter":
            n, m, u, w, rho, alpha, type_of_random_graph = args
            folder_name = "data"
            file_name = "N_" + str(n)\
                    + "_M_" + str(m)\
                    + "_U_" + str(u)\
                    + "_W_" + str(w)\
                    + "_Rho_" + str(rho)\
                    + "_Alpha_" + str(alpha)\
                    + "_RG_" + type_of_random_graph\
                    + ".pickle"
        case "by_filename":
            folder_name, file_name = args
    
    name = folder_name + "/" + file_name
    if not os.path.isdir(folder_name):
        raise Exception("folder does not exist")
    if not os.path.exists(name):
        return []
    
    with open(name, "rb") as file:
        data_list = pickle.load(file)
    return len(data_list)
if __name__ == "__main__":
    # you can write unit tests here
    pass