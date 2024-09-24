# network_analysis_pkg/dominating_set.py

import itertools
import networkx as nx
import datetime
import random
import os

def all_min_dominating_set(nxG):
    min_dominating_set_result = []
    strength_of_all_min_dominating_set = []
    node_list = list(nxG.nodes())
    node_num = nxG.number_of_nodes()
    min_dominating_size = node_num
    print("start searching....")
    for i in range(1, node_num + 1):
        print(f"searching for size {i} ...")
        print(str(datetime.datetime.now()))
        set_list = itertools.combinations(node_list, i)
        for temp_set in set_list:
            if nx.is_dominating_set(nxG, temp_set):
                min_dominating_size = i
                min_dominating_set_result.append(temp_set)
                temp_total_strength = sum(nxG.degree(dom_node, weight='weight') for dom_node in temp_set)
                strength_of_all_min_dominating_set.append(temp_total_strength)
        if i >= min_dominating_size:
            break
        print(str(datetime.datetime.now()))
    return min_dominating_set_result, strength_of_all_min_dominating_set

def greedy_minimum_dominating_set(nxG, times):
    min_dominating_set = []

    for time in range(times):
        nxG_copy = nxG.copy()
        dominating_set = []

        while nxG_copy.nodes():
            node = random.choice(list(nxG_copy.nodes()))
            dominating_set.append(node)
            remove_list = [node] + list(nxG_copy.neighbors(node))
            nxG_copy.remove_nodes_from(remove_list)

        dominating_set = set(dominating_set)
        if not min_dominating_set:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) == len(dominating_set) and dominating_set not in min_dominating_set:
            min_dominating_set.append(dominating_set)
        elif len(min_dominating_set[0]) > len(dominating_set):
            min_dominating_set = [dominating_set]

        print(f"times: {time + 1} MDSet size: {len(min_dominating_set[0])} MDSet number: {len(min_dominating_set)}  MDSet: {min_dominating_set}")

    return min_dominating_set

def dominating_frequency(all_dom_set, nxG):
    num_dom_set = len(all_dom_set)
    node_num = nxG.number_of_nodes()
    as_dom_node_count = {node: 0 for node in nxG.nodes()}

    for min_dom_set in all_dom_set:
        for dom_node in min_dom_set:
            as_dom_node_count[dom_node] += 1

    for node in as_dom_node_count:
        as_dom_node_count[node] /= num_dom_set

    print(as_dom_node_count)
    return as_dom_node_count

def save_dominating_sets(all_dom_set: list[set[int]], result_path: str, data_file: str) -> None:
    """保存最小支配集结果到 TXT 文件"""
    file_path = os.path.join(result_path, f"{data_file}_dominating_sets.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        for dom_set in all_dom_set:
            dom_set_str = ', '.join(map(str, dom_set))
            file.write(f"{dom_set_str}\n")