import pytest
import networkx as nx
from module_control_pkg import (
    all_min_dominating_set,
    greedy_minimum_dominating_set,
    dominating_frequency,
    network_analysis,
    module_controllability,
    plot_network,
    matrix_preprocess
)
import numpy as np

def test_matrix_preprocess():
    matrix = np.array([[1, -2], [3, 4]])
    processed = matrix_preprocess(matrix)
    expected = np.array([[0, 2], [3, 0]])
    assert np.array_equal(processed, expected)

def test_min_dominating_set():
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])
    dom_set, strengths = all_min_dominating_set(G)
    # 预计最小支配集大小为 2
    assert all(len(s) == 2 for s in dom_set)

def test_greedy_min_dominating_set():
    G = nx.path_graph(5)
    dom_set = greedy_minimum_dominating_set(G, 10)
    # 预计最小支配集大小为 2 或 3
    assert len(dom_set[0]) >= 2

def test_network_analysis():
    G = nx.complete_graph(3)
    metrics = network_analysis(G)
    assert metrics["clustering"][0] == 1
    assert metrics["degree"][0] == 1

# 更多测试用例...