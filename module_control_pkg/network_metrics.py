# network_analysis_pkg/network_metrics.py

import networkx as nx
import csv
import os
from typing import Dict

def network_analysis(nxG):
    network_analysis_result = {}
    # 聚集系数
    clustering = nx.clustering(nxG)
    # 接近中心性
    closeness = nx.closeness_centrality(nxG)
    # 介数中心性
    betweenness = nx.betweenness_centrality(nxG)
    # 度中心性（这里要补充进强度）
    degree = nx.degree_centrality(nxG)
    # 平均强度
    average_strength = {}
    for node in nxG.nodes():
        if nxG.degree(node) != 0:
            average_strength[node] = nxG.degree(node, weight='weight') / nxG.degree(node)
        else:
            average_strength[node] = 0
    # k core
    nG_nonself = nxG.copy()
    nG_nonself.remove_edges_from(nx.selfloop_edges(nxG))
    kcore = nx.core_number(nG_nonself)

    network_analysis_result["clustering"] = clustering
    network_analysis_result["closeness"] = closeness
    network_analysis_result["betweenness"] = betweenness
    network_analysis_result["degree"] = degree
    network_analysis_result["average_strength"] = average_strength
    network_analysis_result["kcore"] = kcore

    return network_analysis_result

def save_network_metrics(metrics: dict[str, dict[int, float]], louvain_communities: dict[int, int], as_dom_node_count: dict[int, float], result_path: str, data_file: str) -> None:
    """保存网络度量指标到 CSV 文件"""
    file_path = os.path.join(result_path, f"{data_file}_metrics.csv")
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["item", "degree_centrality", "average_strength", "clustering", "closeness", "betweenness", "kcore", "module", "CF"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for node in metrics["degree"]:
            writer.writerow({
                "item": node,
                "degree_centrality": metrics["degree"][node],
                "average_strength": metrics["average_strength"][node],
                "clustering": metrics["clustering"][node],
                "closeness": metrics["closeness"][node],
                "betweenness": metrics["betweenness"][node],
                "kcore": metrics["kcore"][node],
                "module": louvain_communities.get(node, -1),  # 默认模块为-1
                "CF": as_dom_node_count.get(node, 0.0)        # 默认CF为0.0
            })
