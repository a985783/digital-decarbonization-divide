# 数字化脱碳机制（复现包）

本仓库包含论文的数据和代码：
**《数字化脱碳机制：基于高维面板数据与双重机器学习的实证研究》**

## 🔑 核心发现

使用高维因果推断框架（面板DML），基于40个经济体的60个变量的扩展数据集：

1. **基准效应**：在排除能源控制变量的基准规格中，ICT服务出口与CO₂排放呈**显著负相关**（θ ≈ −0.028，p < 0.01）。这意味着ICT每增长1个百分点，人均排放减少约28公斤。

2. **敏感性**：当加入能源使用作为协变量时，效应衰减并失去显著性。辅助回归显示ICT与总体能源使用之间没有显著因果关系，表明基准结果对能源强度混杂因素敏感。

3. **阈值（探索性）**：SHAP分析表明在ICT服务出口约**6%**处可能存在机制转换，这是假设生成，需要进一步验证。

## 📊 主要结果

### 表：DML因果估计
| 规格 | θ | 标准误 | p值 | 解释 |
| :--- | :--- | :--- | :--- | :--- |
| (1) Lasso选择 | −0.007 | 0.006 | 0.248 | 含能源控制 |
| (2) Lasso（无能源）| **−0.027*** | 0.009 | 0.003 | 基准（显著）|
| (3) 完整高维 | −0.013 | 0.009 | 0.145 | 含能源控制 |
| (4) 完整高维（无能源）| **−0.028*** | 0.010 | 0.005 | 基准（显著）|

## 📂 仓库结构

```
├── data/
│   ├── wdi_expanded_raw.csv       # 扩展WDI/WGI数据（60+变量）
│   └── clean_data_v3_imputed.csv  # 最终MICE插补数据集（N=960）
├── scripts/
│   ├── solve_wdi_v4_expanded_zip.py # 阶段1：数据下载
│   ├── impute_mice.py             # 阶段1：MICE插补
│   ├── lasso_selection.py         # 阶段2：变量选择
│   ├── dml_causal_v2.py           # 阶段2：DML因果推断
│   ├── xgboost_shap_v3.py         # 阶段3：SHAP分析
│   └── mechanism_check.py         # 阶段3：机制分析
├── results/
│   ├── dml_results_v3.csv         # 主要DML估计
│   ├── mechanism_results.csv      # 机制分析结果
│   └── figures/                   # 生成的图表
├── paper.tex                      # LaTeX源文件（中文版）
└── requirements.txt               # 依赖项
```

## 🚀 复现指南

**前置条件**：推荐Python 3.10+

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整流程

**阶段1：数据工程**
```bash
python scripts/solve_wdi_v4_expanded_zip.py
python scripts/impute_mice.py
```

**阶段2：变量选择与因果推断**
```bash
python scripts/lasso_selection.py
python scripts/dml_causal_v2.py
```

**阶段3：机制与探索性分析**
```bash
python scripts/mechanism_check.py
python scripts/xgboost_shap_v3.py
```

## ⚠️ 方法论说明

### 识别策略
- **双向固定效应**：国家和年份FE吸收未观测异质性
- **GroupKFold（K=5）**：按国家分组的交叉验证，防止泄漏
- **聚类稳健标准误**：国家层面聚类，含小样本校正

### 缺失数据处理
- **MICE**：仅对控制变量进行链式方程多重插补
- **不插补Y/T**：结果和处理变量严格使用完整案例
- **防泄漏**：插补模型仅在训练折内训练

---
**维护者**：崔庆松  
**更新日期**：2026年1月21日
