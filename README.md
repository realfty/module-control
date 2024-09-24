README.md

# 模块控制

模块控制 (`module-control`) 是一个用于网络分析的 Python 包，涵盖了支配集（dominating sets）和可控性分析等功能。通过此包，用户可以轻松地进行网络建模、分析、可视化以及结果的保存与导出。

## 特色

- **网络建模**：基于部分相关矩阵构建网络图。
- **网络分析**：计算网络的度量指标，如中心性、聚类系数等。
- **社区检测**：使用 Louvain 算法进行社区划分。
- **支配集计算**：支持精确算法和贪心算法计算最小支配集。
- **可控性分析**：分析模块间的控制强度。
- **可视化**：生成部分相关矩阵热力图和带社区划分的网络图。
- **结果保存**：将分析结果保存为 CSV 和 TXT 文件，便于后续使用。

## 安装

使用 `pip` 安装 `module-control`：

```bash
pip install module-control
```
或者使用 `Poetry` 安装：

```bash
poetry add module-control
```
## 使用示例

以下是一个使用 `module-control` 包的完整示例，展示如何进行网络分析和可视化。

### 1. 准备数据

确保你的数据文件（例如 `test.csv`）位于指定的目录中。数据文件应为 CSV 格式，每行代表一个数据点，每列代表一个变量。

### 2. 创建并运行 `main.py`

创建一个 `main.py` 文件，并编写以下代码：

```
# main.py

import pandas as pd
import networkx as nx
from module_control_pkg import (
    network_analysis,
    module_controllability,
    plot_network,
    plot_heatmap,
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
    # 文件路径，要计算的数据的文件夹
    data_path = "D:/module-control/"
    # 文件路径，文件要保存到的文件夹
    result_path = "D:/module-control/results/"
    # 要计算的网络名字
    data_file = "test"

    # 确保结果目录存在
    ensure_directory_exists(result_path)

    # 读取 CSV 数据
    pd_data = pd.read_csv(os.path.join(data_path, f"{data_file}.csv"))
    print("数据读取完成。")

    # 计算部分相关矩阵
    estimator = GraphicalLasso(alpha=0.05)
    estimator.fit(pd_data)
    partial_corr_matrix = -estimator.precision_ / np.outer(np.sqrt(np.diag(estimator.precision_)), np.sqrt(np.diag(estimator.precision_)))
    np.fill_diagonal(partial_corr_matrix, 1)
    print("部分相关矩阵计算完成。")

    # 生成网络
    nxG = nx.from_numpy_array(partial_corr_matrix)
    print("网络图生成完成。")

    # 网络分析
    metrics = network_analysis(nxG)
    print("网络分析完成。")

    # 社区划分
    louvain_communities = community.best_partition(nxG)
    print("社区划分完成。")

    # 计算最小支配集（使用贪心算法）
    all_dom_set = greedy_minimum_dominating_set(nxG, times=1000)
    print(f"贪心算法找到的最小支配集: {all_dom_set}")

    # 计算支配集频率
    as_dom_node_count = dominating_frequency(all_dom_set, nxG)
    print("支配集频率计算完成。")

    # 可控性分析
    controllability = module_controllability(nxG, all_dom_set, louvain_communities)
    print("可控性分析完成。")

    # 保存结果
    # 1. 保存网络度量指标
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
    print(f"网络图和热力图已保存至 {result_path} 目录。")

if __name__ == "__main__":
    main()
```

### 3. 运行 `main.py`

在命令行中导航到包含 `main.py` 的目录，然后运行：

```bash
python main.py
```

运行完成后，结果将保存在 `D:/module-control/results/` 目录下，包括：

- `test_metrics.csv`：包含节点的中心性和其他指标
- `test_edges.csv`：包含边的权重
- `test_controllability.csv`：包含模块间的平均控制强度
- `test_dominating_sets.txt`：包含最小支配集结果
- `network_communities_test.png`：带社区划分的网络图
- `heatmap_test.png`：部分相关矩阵热力图

## 使用流程

1. **安装包**：

    ```bash
    pip install module-control
    ```

2. **准备数据**：

    - 准备一个 CSV 文件，每行代表一个数据点，每列代表一个变量。

3. **编写脚本**：

    - 使用 `module-control` 提供的函数进行网络分析和可视化。
    - 参考上述 `main.py` 示例进行编写。

4. **运行脚本**：

    ```bash
    python main.py
    ```

## 示例代码

以下是一些常用功能的示例代码，帮助你快速上手：

### 1. 读取数据并计算部分相关矩阵

```python
import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLasso
from module_control_pkg import ensure_directory_exists

# 设置路径
data_path = "D:/module-control/"
data_file = "test"
result_path = "D:/module-control/results/"

# 确保结果目录存在
ensure_directory_exists(result_path)

# 读取数据
pd_data = pd.read_csv(os.path.join(data_path, f"{data_file}.csv"))

# 计算部分相关矩阵
estimator = GraphicalLasso(alpha=0.05)
estimator.fit(pd_data)
partial_corr_matrix = -estimator.precision_ / np.outer(np.sqrt(np.diag(estimator.precision_)), np.sqrt(np.diag(estimator.precision_)))
np.fill_diagonal(partial_corr_matrix, 1)
```

### 2. 生成网络图并进行社区划分

```python
import networkx as nx
import community  # python-louvain
from module_control_pkg import network_analysis, plot_network

# 生成网络
nxG = nx.from_numpy_array(partial_corr_matrix)

# 网络分析
metrics = network_analysis(nxG)

# 社区划分
louvain_communities = community.best_partition(nxG)

# 可视化网络
plot_network(nxG, louvain_communities, result_path, "network_graph")
```

### 3. 计算并保存最小支配集

```python
from module_control_pkg import greedy_minimum_dominating_set, save_dominating_sets

# 计算最小支配集
all_dom_set = greedy_minimum_dominating_set(nxG, times=1000)

# 保存最小支配集
save_dominating_sets(all_dom_set, result_path, "dominating_sets")
```

## 贡献

欢迎提交问题（Issues）和拉取请求（Pull Requests）！如果你有新的功能建议或发现了 bug，请随时告知。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请联系 (mailto:fty1730935466@gmail.com)。

---

感谢使用 `module-control`！希望此工具能助你在网络分析领域取得更大的成果。
