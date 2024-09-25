# module_control_pkg/visualize.py

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import os
from typing import Dict

def plot_heatmap(matrix: np.ndarray, result_path: str, data_file: str) -> None:
    """绘制偏相关矩阵热力图并保存"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 5})
    
    plt.xticks(ticks=np.arange(0, matrix.shape[1], 5), labels=np.arange(0, matrix.shape[1], 5))
    plt.yticks(ticks=np.arange(0, matrix.shape[0], 5), labels=np.arange(0, matrix.shape[0], 5))
    
    plt.title('Partial Correlation Matrix Heatmap')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    
    # 保存热力图
    heatmap_file = os.path.join(result_path, f"heatmap_{data_file}.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Heatmap saved to {heatmap_file}")

def plot_network(nxG: nx.Graph, louvain_communities: dict[int, int], result_path: str, data_file: str) -> None:
    """绘制网络图并保存，节点颜色区分模块，显示节点编号，边颜色深浅反映权重"""
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(nxG, seed=42)  # 确保布局一致

    # 获取社区
    communities = {}
    for node, community in louvain_communities.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    # 选择颜色映射
    cmap = plt.get_cmap('Set3')  # 使用Set3色图，具有较高对比度
    colors = [cmap(i % cmap.N) for i in range(len(communities))]

    # 绘制节点
    for idx, (community, nodes) in enumerate(communities.items()):
        nx.draw_networkx_nodes(
            nxG, pos,
            nodelist=nodes,
            node_color=[colors[idx]],
            node_size=500,
            label=f"Module {community}"
        )

    # 绘制边，边颜色深浅根据权重
    edges = nxG.edges(data=True)
    weights = [edge_data['weight'] for _, _, edge_data in edges]
    max_weight = max(weights) if weights else 1
    edge_colors = [weight / max_weight for weight in weights]  # 归一化颜色
    nx.draw_networkx_edges(
        nxG, pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        alpha=0.6,
        width=3  # 增加边的粗细
    )

    # 绘制节点标签
    nx.draw_networkx_labels(nxG, pos, font_size=12, font_weight='bold')

    # 创建图例
    patches = [mpatches.Patch(color=colors[idx], label=f'Module {community}') 
               for idx, (community, _) in enumerate(communities.items())]
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))

    # 在图的下方添加模块包含的节点号
    legend_text = ""
    for idx, (community, nodes) in enumerate(communities.items()):
        nodes_str = ", ".join(map(str, nodes))
        legend_text += f"Module {community}: {nodes_str}\n"
    
    plt.figtext(0.1, 0.02, legend_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.title("Network Visualization with Louvain Communities", fontsize=20)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 调整布局，留出下方空间

    # 保存网络图
    network_file = os.path.join(result_path, f"network_communities_{data_file}.png")
    plt.savefig(network_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network graph saved to {network_file}")

def plot_module(controllability: dict[str, float], louvain_communities: dict[int, int], result_path: str, data_file: str) -> None:
    """
    绘制模块间的平均控制强度图并保存。

    参数:
        controllability (dict[str, float]): 模块间的平均控制强度，键为 "source_target" 格式的字符串。
        louvain_communities (dict[int, int]): 节点到社区的映射。
        result_path (str): 结果保存路径。
        data_file (str): 数据文件名，用于命名保存的图像文件。
    """
    plt.figure(figsize=(12, 12))

    # 创建有向图
    G = nx.DiGraph()

    # 获取所有模块
    modules = set()
    for key in controllability.keys():
        source, target = key.split('_')
        if source != target:  # 去除自环边
            modules.add(int(source))
            modules.add(int(target))
    modules = sorted(modules)
    module_labels = {module: f"module_{module}" for module in modules}

    # 添加节点
    for module in modules:
        G.add_node(module, label=module_labels[module])

    # 添加边
    for key, value in controllability.items():
        source, target = map(int, key.split('_'))
        if source != target and value > 0:  # 去除自环边并且权重大于0
            G.add_edge(source, target, weight=value)

    # 设置布局
    pos = nx.spring_layout(G, k=2, iterations=50)  # 增加 k 值和迭代次数以增加节点间距

    # 获取社区
    unique_communities = sorted(set(louvain_communities.values()))
    cmap = plt.get_cmap('Set3')
    community_colors = {comm: cmap(i % cmap.N) for i, comm in enumerate(unique_communities)}

    # 节点颜色
    node_colors = [community_colors[module] for module in G.nodes()]

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3500,
        node_shape='o',  # 使用圆形节点
        alpha=0.8,
    )

    # 绘制边，边颜色根据源节点颜色，边的宽度根据权重
    edges = list(G.edges(data=True))
    edge_colors = [community_colors[u] for u, _, _ in edges]
    weights = [edge_data['weight'] for _, _, edge_data in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [8 * (weight / max_weight) for weight in weights]  # 减小边宽度

    # 绘制有向边
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=edge_colors,
        width=edge_widths,
        arrowsize=80,  # 减小箭头大小
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',  # 使用弧线连接
    )

    # 绘制节点标签
    labels = {module: f"module_{module}" for module in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')

    # 添加边上的标签（当控制强度 > 0.3 时）
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if data['weight'] > 0.3:
            edge_labels[(u, v)] = f"{data['weight']:.2f}"
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10, label_pos=0.5)

    plt.title("Module Controllability Graph", fontsize=20)
    plt.axis('off')
    plt.tight_layout()

    # 创建颜色图例
    patches = [mpatches.Patch(color=community_colors[comm], label=f"Module {comm}") for comm in unique_communities]
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))

    # 保存模块控制图
    module_graph_file = os.path.join(result_path, f"module_controllability_{data_file}.png")
    plt.savefig(module_graph_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Module controllability graph saved to {module_graph_file}")