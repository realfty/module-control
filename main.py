# main.py

import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.covariance import GraphicalLasso
import community  # python-louvain

from module_control_pkg import (
    matrix_preprocess,
    all_min_dominating_set,
    network_analysis,
    module_controllability,
    plot_network,
    plot_heatmap,
    plot_module,
    save_network_metrics,
    save_edges,
    save_controllability,
    save_dominating_sets,
    ensure_directory_exists,
    greedy_minimum_dominating_set,
    dominating_frequency
)
import os
from sklearn.covariance import GraphicalLasso
import numpy as np
import community  # python-louvain

def main():
    ## 文件路径，要计算的数据的文件夹
    data_path = "D:/module-control/"
    ## 文件路径，文件要保存到的文件夹
    result_path = "D:/module-control/results/"
    ## 要计算的网络名字
    data_file = "test"

    # 确保结果目录存在
    ensure_directory_exists(result_path)

    # 读取csv,生成pd
    pd_data = pd.read_csv(data_path+data_file+".csv")
    print("数据读取完成。")

    # 计算网络模型，返回矩阵
    ################ 可选参数1 ###################
    ## GraphicalLassoCV()使用交叉验证自动确定alpha
    # estimator = GraphicalLassoCV()
    ## GraphicalLasso()需要给定参数alpha，默认alpha=0.01，alpha越大网络越稀疏
    estimator = GraphicalLasso(alpha=0.05)
    estimator.fit(pd_data)
    print(estimator.precision_)
    print(estimator.precision_.shape)

    #### 精准矩阵要进行处理，从逆协方差变成偏相关
    # 将精度矩阵对角线元素开根号
    diag_sqrt = np.sqrt(np.diag(estimator.precision_))
    # 计算偏相关矩阵
    partial_corr_matrix = -estimator.precision_ / np.outer(diag_sqrt, diag_sqrt)
    # 将对角线元素设置为1
    np.fill_diagonal(partial_corr_matrix, 1)
    print(partial_corr_matrix)
    print("部分相关矩阵计算完成。")

    # 预处理矩阵
    matrix = matrix_preprocess(partial_corr_matrix)
    print("矩阵预处理完成。")

    # 生成网络
    nxG = nx.Graph(matrix, weight=True)
    print("网络图生成完成。")

    # 网络分析
    metrics = network_analysis(nxG)
    print("网络分析完成：")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 社区划分
    louvain_communities = community.best_partition(nxG)
    print("社区划分完成。")

    # 最小支配集
    # all_dom_set, strengths = all_min_dominating_set(nxG)
    # algorithm = "precise"
    # print(f"找到 {len(all_dom_set)} 个最小支配集。")
    # 贪心算法
    all_dom_set = greedy_minimum_dominating_set(nxG, times=1000)
    algorithm = "greedy"
    print(f"贪心算法找到的最小支配集: {all_dom_set}")

    # ACF
    as_dom_node_count = dominating_frequency(all_dom_set, nxG)
    print("frequency as dom nodes: "+str(as_dom_node_count))

    # 可控性分析
    controllability = module_controllability(nxG, all_dom_set, louvain_communities)
    print("可控性分析完成。")

    # 计算支配集频率
    as_dom_node_count = dominating_frequency(all_dom_set, nxG)

    # 保存结果
    # 1. 保存网络度量指标
    # 需要将 'module' 和 'CF' 添加到度量结果中
    for node in metrics["degree"]:
        metrics["module"] = {node: louvain_communities[node] for node in metrics["degree"].keys()}
        metrics["CF"] = as_dom_node_count

    save_network_metrics(metrics, louvain_communities, as_dom_node_count, result_path, data_file)
    print("网络度量指标已保存。")

    # 2. 保存边权值
    save_edges(nxG, result_path, data_file)
    print("边权值已保存。")

    # 3. 保存模块间的平均控制强度
    save_controllability(controllability, result_path, data_file)
    print("模块间的平均控制强度已保存。")

    # 4. 保存最小支配集结果
    save_dominating_sets(all_dom_set, result_path, data_file)
    print("最小支配集结果已保存。")

    # 可视化
    plot_network(nxG, louvain_communities, result_path, "network_graph")
    plot_heatmap(partial_corr_matrix, result_path, "partial_corr_matrix")
    plot_module(controllability, louvain_communities, result_path, data_file)
    print(f"网络图、热力图和模块控制图已保存至 {result_path} 目录。")

if __name__ == "__main__":
    main()
