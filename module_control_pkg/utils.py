# network_control_pkg/utils.py

import numpy as np
import os
import csv
import networkx as nx

def ensure_directory_exists(directory: str) -> None:
    """确保指定的目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def matrix_preprocess(matrix):
    number_of_nodes = matrix.shape[1]
    matrix_result = matrix.copy()
    # 去对角线
    np.fill_diagonal(matrix_result, 0)
    # 取绝对值
    matrix_result = np.abs(matrix_result)
    return matrix_result

def save_edges(nxG: nx.Graph, result_path: str, data_file: str) -> None:
    """保存边权值到 CSV 文件"""
    file_path = os.path.join(result_path, f"{data_file}_edges.csv")
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["source", "target", "weight"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for edge in nxG.edges(data=True):
            writer.writerow({
                "source": edge[0],
                "target": edge[1],
                "weight": edge[2].get('weight', 1)  # 默认权重为1
            })