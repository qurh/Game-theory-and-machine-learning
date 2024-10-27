import random
import math
import numpy as np
import typing_extensions

class partner_switching_model(object):
    def __init__(self, initialization_method, *args):
        """
        initialize "partner_switching_model" by
        opt 1 (simulation use) : Generatate a random network by given initialization parameters.
        opt 2 (analysis use)   : Load existed network without initialization and iteration parameters, but some method is limited.
        opt 3 (analysis use)   : Load existed network with initialization and iteration parameters, all methods are available except iteration related ones.

        Args:
            initialization_method (str): there are three options "new" / "load" / "load_all"
            opt 1 : "new", N: int, M: int, U: float, W: float, Rho: float, Alpha: float, Type_of_random_graph: str
            opt 2 : "load", neighbor_list: list[list], strategy_list: list[str]
            opt 3 : "load_all", N: int, M: int, U: float, W: float, Rho: float, Alpha: float, Type_of_random_graph: str, neighbor_list: list[list], strategy_list: list[str] 
        Raises:
            Exception: number of nodes must be greater than zero: N = {N} > 0
            Exception: number of edges must be non-negative: M = {M} >= 0
            Exception: Fermi coefficient must be non-negative: Alpha = {Alpha} >= 0
            Exception: too many edges: M = {M} > N*(N-1)/2 = {N * (N - 1) / 2}
            Exception: initial cooperative ratio out of range: 0 <= Rho = {Rho} <= 1
            Exception: cost to benefit ratio out of range: 0 <= U = {U} <= 1
            Exception: copy rate out of range: 0 <= W = {W} <= 1
        """
        self.initialization_method = initialization_method
        match initialization_method:
            case "new":
                n, m, u, w, rho, alpha, type_of_random_graph = args
                if n is None:
                    return
                if n <= 0:
                    raise Exception(f"number of nodes must be greater than zero: N = {n} > 0")
                if m < 0:
                    raise Exception(f"number of edges must be non-negative: M = {m} >= 0")
                if alpha < 0:
                    raise Exception(f"Fermi coefficient must be non-negative: Alpha = {alpha} >= 0")
                if m > n * (n - 1) / 2:
                    raise Exception(f"too many edges: M = {m} > N*(N-1)/2 = {n * (n - 1) / 2}")
                if not (0 <= rho and rho <= 1):
                    raise Exception(f"initial cooperative ratio out of range: 0 <= Rho = {rho} <= 1")
                if not (0 <= u and u <= 1):
                    raise Exception(f"cost to benefit ratio out of range: 0 <= U = {u} <= 1")
                if not (0 <= w and w <= 1):
                    raise Exception(f"copy rate out of range: 0 <= W = {w} <= 1")
                
                
                self._number_of_node = n
                self._number_of_edge = m
                self._cost_to_benefit_ratio = u
                self._strategy_update_rate = w
                self._initial_cooperative_ratio = rho
                self._type_of_random_graph = type_of_random_graph
                self._fermi_coefficient = alpha
                self._neighbor_list, self._strategy_list = self.__generate_random_network()
                self.__tmp_directed_CD_edge_list = self.directed_CD_edge_list
                return
            case "load":
                self._neighbor_list, self._strategy_list = args
                self._number_of_node = len(self._neighbor_list)
                self._number_of_edge = round(np.sum([len(neighbor) for neighbor in self._neighbor_list])/2)
                self._cost_to_benefit_ratio = None
                self._strategy_update_rate = None
                self._initial_cooperative_ratio = None
                self._type_of_random_graph = None
                self._fermi_coefficient = None
                return
            case "load_all":
                self._number_of_node = args[0]
                self._number_of_edge = args[1]
                self._cost_to_benefit_ratio = args[2]
                self._strategy_update_rate = args[3]
                self._initial_cooperative_ratio = args[4]
                self._fermi_coefficient = args[5]
                self._type_of_random_graph = args[6]
                self._neighbor_list = args[7]
                self._strategy_list = args[8]
                return
    def get_next_iteration(self) -> bool:
        """
        Update the network once.

        Raises:
            Exception: loaded game does not support this method

        Returns:
            is_end (bool) : True  if no CD edge exists
                            False otherwise
        """
        if self.initialization_method != "new":
            raise Exception("loaded game does not support this method")
        
        is_end = (self.__tmp_number_of_CD_edge == 0)
        if is_end is True:
            # if no CD edge initially then early return
            return is_end
        
        idx_of_tmp_CD_edge, cooperative_player, defective_player = self.__get_random_CD_edge()
        match self.__random_behavior():
            case "copy strategy":
                if self.__is_C_copy_D(cooperative_player, defective_player):
                    self._strategy_list[cooperative_player] = self._strategy_list[defective_player]
                    for neighbor in self._neighbor_list[cooperative_player]:
                        if self._strategy_list[neighbor] == "D":
                            self.__tmp_directed_CD_edge_list.remove((cooperative_player, neighbor))
                        else:
                            self.__tmp_directed_CD_edge_list.append((neighbor, cooperative_player))
                else:
                    self._strategy_list[defective_player] = self._strategy_list[cooperative_player]
                    for neighbor in self._neighbor_list[defective_player]:
                        if self._strategy_list[neighbor] == "C":
                            self.__tmp_directed_CD_edge_list.remove((neighbor, defective_player))
                        else:
                            self.__tmp_directed_CD_edge_list.append((defective_player, neighbor))
            case "rewiring":
                # partnar switching
                # choose new neighber first (order matters)
                new_neighor = self.__random_new_neighor(cooperative_player)
                if new_neighor is not None:
                    # drop CD edge
                    self._neighbor_list[cooperative_player].remove(defective_player)
                    self._neighbor_list[defective_player].remove(cooperative_player)

                    # rewire
                    self._neighbor_list[cooperative_player].append(new_neighor)
                    self._neighbor_list[new_neighor].append(cooperative_player)

                    # drop CD edge
                    self.__tmp_directed_CD_edge_list.pop(idx_of_tmp_CD_edge)
                    # rewire
                    if self._strategy_list[new_neighor] == "D":
                        self.__tmp_directed_CD_edge_list.append((cooperative_player, new_neighor))
        is_end = (self.__tmp_number_of_CD_edge == 0)
        return is_end
    def get_final_state(self, Max_iteration, seed = None) -> typing_extensions.Self:
        """
        If only the final state is interesting, you may use this method to access the final state.
        It continuously update the network until no CD edge appear or the given maximum of iteration is reached.
        To not limit the maximum of iteration, set Max_iteration = -1.
        Unlike "get_next_iteration", this method return "self" such that "map", list comprehensioin, etc., are supported. 
        Args:
            Max_iteration (int): maximum of iteration
        Returns:
            partner_switching_model
        """
        if seed is not None:
            random.seed(seed)
        is_end = False
        if Max_iteration == -1:
            while is_end == False:
                is_end = self.get_next_iteration()
        else:
            for _ in range(Max_iteration):
                if is_end is True:
                    break
                is_end = self.get_next_iteration()
        return self
    @property
    def number_of_CC_edge(self) -> int:
        return round(len(self.directed_CC_edge_list)/2)
    @property
    def number_of_CD_edge(self) -> int:
        return len(self.directed_CD_edge_list)
    @property
    def __tmp_number_of_CD_edge(self) -> int:
        if self.initialization_method != "new":
            raise Exception("loaded game does not support this method")
        return len(self.__tmp_directed_CD_edge_list)
    @property
    def number_of_DD_edge(self) -> int:
        return round(len(self.directed_DD_edge_list)/2)
    @property
    def directed_CC_edge_list(self) -> list[tuple]:
        """
        Note that both (cooperator1, cooperator2) and (cooperator2, cooperator1) will appear in the list.
        Returns:
            list[tuple]: [(cooperator1, cooperator2)]
        """
        # directed_CC_edge_list = []
        # for i in range(self._number_of_node):
        #     for j in self._neighbor_list[i]:
        #         if self._strategy_list[i] == "C" and self._strategy_list[j] == "C":
        #             directed_CC_edge_list.append((i, j))
        return [(i,j) for i in range(self._number_of_node) for j in self._neighbor_list[i] if (self._strategy_list[i] == "C" and self._strategy_list[j] == "C")]
    @property
    def directed_DD_edge_list(self) -> list[tuple]:
        """
        Note that both (defector1, defector2) and (defector2, defector1) will appear in the list.
        Returns:
            list[tuple]: [(defector1, defector2)]
        """
        # directed_DD_edge_list = []
        # for i in range(self._number_of_node):
        #     for j in self._neighbor_list[i]:
        #         if self._strategy_list[i] == "D" and self._strategy_list[j] == "D":
        #             directed_DD_edge_list.append((i, j))
        return [(i,j) for i in range(self._number_of_node) for j in self._neighbor_list[i] if (self._strategy_list[i] == "D" and self._strategy_list[j] == "D")]
    @property
    def directed_CD_edge_list(self) -> list[tuple]:
        """
        Returns:
            list[tuple]: [(cooperator, defector)]
        """
        # directed_CD_edge_list = []
        # for i in range(self._number_of_node):
        #     for j in self._neighbor_list[i]:
        #         if self._strategy_list[i] == "C" and self._strategy_list[j] == "D":
        #             directed_CD_edge_list.append((i, j))
        return [(i,j) for i in range(self._number_of_node) for j in self._neighbor_list[i] if (self._strategy_list[i] == "C" and self._strategy_list[j] == "D")]
    @property
    def directed_DC_edge_list(self) -> list[tuple]:
        """
        Returns:
            list[tuple]: [(defector, cooperator)]
        """
        # directed_DC_edge_list = []
        # for i in range(self._number_of_node):
        #     for j in self._neighbor_list[i]:
        #         if self._strategy_list[i] == "D" and self._strategy_list[j] == "C":
        #             directed_DC_edge_list.append({i, j})
        return [(i,j) for i in range(self._number_of_node) for j in self._neighbor_list[i] if (self._strategy_list[i] == "D" and self._strategy_list[j] == "C")]
    @property
    def cooperative_node(self) -> list[int]:
        return [i for i in range(self._number_of_node) if self._strategy_list[i] == "C"]
    @property
    def defective_node(self) -> list[int]:
        return [i for i in range(self._number_of_node) if self._strategy_list[i] == "D"]
    @property
    def number_cooperator(self) -> int:
        return len(self.cooperative_node)
    @property
    def number_defector(self) -> int:
        return len(self.defective_node)
    @property
    def cooperative_ratio(self) -> float:
        """
        Returns:
            float: number of cooperator / number of node
        """
        return self.number_cooperator / self._number_of_node
    @property
    def cooperative_ratio_of_stub(self) -> float:
        """
        Returns:
            float: (2*number of CC edges + 1*number of CD edges) / (2*number of edges)
        """
        if self._number_of_edge == 0:
            return 0
        else:
            return (2 * self.number_of_CC_edge + 1 * self.number_of_CD_edge) / (2 * self._number_of_edge)
    def get_payoff(self, player : int) -> float:
        """
        Args:
            player (int)

        Raises:
            Exception: cost to benefit ratio (U) is not specified

        Returns:
            float: payoff of player
        """
        if self._cost_to_benefit_ratio is None:
            raise Exception("cost to benefit ratio (U) is not specified")
        payoff = 0
        if self._strategy_list[player] == "C":
            for neighbor in self._neighbor_list[player]:
                payoff += 1 * (self._strategy_list[neighbor] == "C") \
                        + 0 * (self._strategy_list[neighbor] == "D")
        else:
            for neighbor in self._neighbor_list[player]:
                payoff += (1+self._cost_to_benefit_ratio) * (self._strategy_list[neighbor] == "C") \
                        + (self._cost_to_benefit_ratio)   * (self._strategy_list[neighbor] == "D")
        return payoff
    def __get_random_CD_edge(self) -> tuple[int, int, int]:
        if self.__tmp_number_of_CD_edge == 0:
            random_index = None
            random_C_node = None
            random_D_node = None
        else:
            random_index = random.choice(range(len(self.__tmp_directed_CD_edge_list)))
            random_C_node, random_D_node = self.__tmp_directed_CD_edge_list[random_index]
        return random_index, random_C_node, random_D_node
    def __random_behavior(self) -> str:
        if (random.random() <= self._strategy_update_rate):
            return "copy strategy"
        else:
            return "rewiring"
        # return (random.random <= self.copy_rate)
    def __is_C_copy_D(self, cooperative_player, defective_player) -> bool:
        # payoff_of_cooperative_player = self.get_payoff(cooperative_player)
        # payoff_of_defective_player = self.get_payoff(defective_player)
        # arg = np.clip(self.fermi_coefficient * (payoff_of_cooperative_player - payoff_of_defective_player), -700, 700)
        # prob = 1/(  1 + math.exp(arg)  )
        # return random.random <= prob
        return random.random() <= 1 / (  1 + math.exp(np.clip(self._fermi_coefficient * (self.get_payoff(cooperative_player) - self.get_payoff(defective_player)), -700, 700))  )
    def __random_new_neighor(self, cooperative_player) -> int:
        if len(self._neighbor_list[cooperative_player])+1 != self._number_of_node:
            node_list = list(range(self._number_of_node))
            for idx in sorted([cooperative_player] + list(self._neighbor_list[cooperative_player]), reverse=True):
                node_list.pop(idx)
            return random.choice(node_list)
        else:
            return None
    def __generate_random_network(self):
        match self._type_of_random_graph:
            case "ER":
                edge_list = [(i,j) for i in range(self._number_of_node) for j in range(i+1, self._number_of_node)]
                edge_list = random.sample(edge_list, k = self._number_of_edge)
                neighbor_list = [list() for _ in range(self._number_of_node)]
                for a, b in edge_list:
                    neighbor_list[a].append(b)
                    neighbor_list[b].append(a)

                strategy_list = ["C"] * int(self._number_of_node * self._initial_cooperative_ratio)\
                              + ["D"] * (self._number_of_node - int(self._number_of_node * self._initial_cooperative_ratio))
            case _:
                raise Exception(f"unknown random graph: {self._type_of_random_graph}")
        return neighbor_list, strategy_list
    def __str__(self):
        match self.initialization_method:
            case "new":
                ret = "partner switching model:\n" \
                        + "initial network parameters:\n" \
                        + f"  number of nodes = {str(self._number_of_node)}\n"\
                        + f"  number of edges = {str(self._number_of_edge)}\n"\
                        + f"  initial cooparative ratio = {str(self._initial_cooperative_ratio)}\n"\
                        + f"  type of random graph =  {self._type_of_random_graph} + \n"\
                        + "iteration parameters: \n"\
                        + f"  cost to benefit ratio = {str(self._cost_to_benefit_ratio)}\n"\
                        + f"  strategy update rate = {str(self._strategy_update_rate)}\n"\
                        + f"  fermi coefficient = {self._fermi_coefficient}\n"\
                        + "current state: \n"\
                        + f"  number of CC edges = {str(self.number_of_CC_edge)}\n"\
                        + f"  number of CD edges = {str(self.number_of_CD_edge)}\n"\
                        + f"  number of DD edges = {str(self.number_of_DD_edge)}\n"\
                        + f"  cooperative ratio of nodes = {str(self.cooperative_ratio)}\n"\
                        + f"  cooperative ratio of stubs = {str(self.cooperative_ratio_of_stub)}\n"
            case "load":
                ret = "partner switching model (loaded):\n" \
                        + "initial network parameters:\n" \
                        + f"  number of nodes = {str(self._number_of_node)}\n"\
                        + f"  number of edges = {str(self._number_of_edge)}\n"\
                        + "current state: \n"\
                        + f"  number of CC edges = {str(self.number_of_CC_edge)}\n"\
                        + f"  number of CD edges = {str(self.number_of_CD_edge)}\n"\
                        + f"  number of DD edges = {str(self.number_of_DD_edge)}\n"\
                        + f"  cooperative ratio of nodes = {str(self.cooperative_ratio)}\n"\
                        + f"  cooperative ratio of stubs = {str(self.cooperative_ratio_of_stub)}\n"
            case "load_all":
                ret = "partner switching model (loaded):\n" \
                        + "initial network parameters:\n" \
                        + f"  number of nodes = {str(self._number_of_node)}\n"\
                        + f"  number of edges = {str(self._number_of_edge)}\n"\
                        + f"  initial cooparative ratio = {str(self._initial_cooperative_ratio)}\n"\
                        + f"  type of random graph = {self._type_of_random_graph} \n"\
                        + "iteration parameters: \n"\
                        + f"  cost to benefit ratio = {str(self._cost_to_benefit_ratio)}\n"\
                        + f"  strategy update rate = {str(self._strategy_update_rate)}\n"\
                        + f"  fermi coefficient = {self._fermi_coefficient}\n"\
                        + "current state: \n"\
                        + f"  number of CC edges = {str(self.number_of_CC_edge)}\n"\
                        + f"  number of CD edges = {str(self.number_of_CD_edge)}\n"\
                        + f"  number of DD edges = {str(self.number_of_DD_edge)}\n"\
                        + f"  cooperative ratio of nodes = {str(self.cooperative_ratio)}\n"\
                        + f"  cooperative ratio of stubs = {str(self.cooperative_ratio_of_stub)}\n"
        return ret
    def __iter__(self):
        while self.__tmp_number_of_CD_edge != 0:
            yield self
            self.get_next_iteration()
        yield self
        return

if __name__ == "__main__":
    # you can write unit tests here
    pass
