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
            node_size=300,
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
    nx.draw_networkx_labels(nxG, pos, font_size=10, font_weight='bold')

    # 创建图例
    patches = [mpatches.Patch(color=colors[idx], label=f'Module {community}') 
               for idx, (community, _) in enumerate(communities.items())]
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))

    # 在图的下方添加模块包含的节点号
    legend_text = ""
    for idx, (community, nodes) in enumerate(communities.items()):
        nodes_str = ", ".join(map(str, nodes))
        legend_text += f"Module {community}: {nodes_str}\n"
    
    plt.figtext(0.1, 0.05, legend_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.title("Network Visualization with Louvain Communities", fontsize=16)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # 留出下方空间

    # 保存网络图
    network_file = os.path.join(result_path, f"network_communities_{data_file}.png")
    plt.savefig(network_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network graph saved to {network_file}")