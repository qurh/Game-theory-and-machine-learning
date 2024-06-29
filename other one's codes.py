# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:05:57 2024

@author: User
"""

# Coder: Chao-Tsung Wu
# Contact: abt6185890818@gmail.com

import networkx as nx
import math
import random
import numpy as np

def matrix(n1, n2):
    if n1 == "C" and n2 == "C":
        value = 1
    elif n1 == "D" and n2 == "D":
        value = u
    elif n1 == "C" and n2 == "D":
        value = 0
    else:
        value = 1 + u
    return value

def update(n1, n2):
    try:
        F = 1 / (1 + math.exp((payoff[n1] - payoff[n2]) * 30))
    except OverflowError:
        if payoff[n1] - payoff[n2] > 0:
            F = 0
        else:
            F = 1
    return F

def count_payoff(m):
    gneighbor = g.neighbors(m)
    neighbor = list(gneighbor)
    neighbor.sort()
    
    for n in neighbor:
        payoff[m] += matrix(strategy[m], strategy[n])

u = 0
w = 1

Set = ["C", "D"]

g = nx.gnm_random_graph(1000, 5000, directed = False)
gnode = g.nodes()
node = list(gnode)
gedge = g.edges()
edge = list(gedge)

strategy = random.sample([Set[i] for i in range(2) for j in range(int(1000 * 0.5))], 1000)
link = []
for i in range(5000):
    link.append([strategy[edge[i][0]], strategy[edge[i][1]]])
payoff = np.zeros(1000)

while True:
    link_disc = random.choices([i for i, j in enumerate(link) if j == ["C", "D"] or j == ["D", "C"]])[0]
    disc = link[link_disc]
    a = edge[link_disc][disc.index("C")]
    b = edge[link_disc][disc.index("D")]
    
    Type = " ".join(random.choices(["1", "2"], weights = (w, 1 - w), k=1))
    if Type == "1": # Node update
        payoff[a] = 0
        count_payoff(a)
        payoff[b] = 0
        count_payoff(b)
        
        F = update(a, b)
        which = " ".join(random.choices(["a", "b"], weights = (F, 1 - F), k=1))
        if which == "a":
            strategy[a] = " ".join(random.choices([strategy[a], strategy[b]], weights = (1 - F, F), k=1))
        else:
            strategy[b] = " ".join(random.choices([strategy[b], strategy[a]], weights = (F, 1 - F), k=1))
    else: # Edge Update
        g.remove_edge(*edge[link_disc])
        gneighbor = g.neighbors(a)
        neighbor = list(gneighbor)
        neighbor.sort()
        new = random.choices(list(set(node) - (set(neighbor) | set([a]))))[0]
        g.add_edge(a, new)
    
    gedge = g.edges()
    edge = list(gedge)
    link = []
    for i in range(5000):
        link.append([strategy[edge[i][0]], strategy[edge[i][1]]])
    
    if link.count(["C", "D"]) + link.count(["D", "C"]) == 0:
        break
