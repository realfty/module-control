# network_analysis_pkg/controllability.py

import networkx as nx
import csv
import os

def module_controllability(nxG, all_dom_set, louvain_communities):
    number_of_communities = max(louvain_communities.values()) + 1
    print(f"module number: {number_of_communities}")
    # 初始化模块
    module = {i: [] for i in range(number_of_communities)}
    for node, community in louvain_communities.items():
        module[community].append(node)

    for i in range(number_of_communities):
        print(f"module {i} has {module[i]} nodes")

    # 初始化结果
    average_module_controllability_result = {f"{source}_{target}": 0 for source in module for target in module}

    for min_dom_set in all_dom_set:
        dominated_area = {}
        for dom_node in min_dom_set:
            dominated_area[dom_node] = set(nxG.neighbors(dom_node)) | {dom_node}

        # 计算社团支配域
        modules_control_area = {}
        for module_index, nodes in module.items():
            single_module_control_area = set()
            for node in nodes:
                if node in min_dom_set:
                    single_module_control_area |= dominated_area[node]
            modules_control_area[module_index] = single_module_control_area

        # 计算社团间支配能力
        temp_module_controllability_result = {}
        for module_source in module:
            for module_target in module:
                control_area = modules_control_area[module_source]
                target_module_area = set(module[module_target])
                intersection = control_area.intersection(target_module_area)
                controllability = len(intersection) / len(target_module_area) if len(target_module_area) > 0 else 0
                key = f"{module_source}_{module_target}"
                average_module_controllability_result[key] += controllability

        print(f"dom_set: {min_dom_set}   module_controllability: {temp_module_controllability_result}")

    # 计算平均值
    for key in average_module_controllability_result:
        average_module_controllability_result[key] /= len(all_dom_set)

    print(f"average_module_controllability: {average_module_controllability_result}")
    return average_module_controllability_result

def save_controllability(controllability: dict[str, float], result_path: str, data_file: str) -> None:
    """保存模块间的平均控制强度到 CSV 文件"""
    file_path = os.path.join(result_path, f"{data_file}_controllability.csv")
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["source_module", "target_module", "average_controllability"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in controllability.items():
            source, target = key.split('_')
            writer.writerow({
                "source_module": source,
                "target_module": target,
                "average_controllability": value
            })