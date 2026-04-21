# Support Vector Machine

## 核心定义  
Support Vector Machine（SVM）是一种监督学习分类方法，其核心思想是：**使用超平面分离数据，并最大化分类间隔（margin）**；SVM 的决策边界仅依赖于部分关键样本（称为支持向量，support vectors）；对于线性不可分的数据，可通过核函数（kernel function）映射到高维空间实现线性可分。

## 关键要点  
- **最大间隔原则**：在所有能正确划分训练样本的超平面中，选择几何间隔（geometric margin）最大的一个，以提升泛化能力。  
- **支持向量**：距离最优超平面最近的样本点；决策函数完全由这些点决定，其余样本不影响模型。  
- **核技巧（Kernel Trick）**：无需显式计算高维映射 $\phi(\mathbf{x})$，仅通过核函数 $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ 即可实现非线性分类。  
- **软间隔（Soft Margin）**：引入松弛变量 $\xi_i \geq 0$ 和惩罚参数 $C$，允许少量误分类，提升对噪声和异常值的鲁棒性。  
- **稀疏性与鲁棒性**：模型仅依赖支持向量，存储和预测开销小；对高维数据表现良好，且不易受维度灾难影响。

## 相关方法/公式  
### 线性 SVM 原始问题（硬间隔）  
给定训练集 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，其中 $y_i \in \{-1, +1\}$：  
$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1,\; i = 1,\dots,n
$$

### 对偶问题（硬间隔）  
引入拉格朗日乘子 $\alpha_i \geq 0$，等价优化：  
$$
\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^\top \mathbf{x}_j) - \sum_{i=1}^n \alpha_i \\
\text{s.t.} \quad \alpha_i \geq 0,\; \sum_{i=1}^n \alpha_i y_i = 0
$$  
解得：$\mathbf{w}^* = \sum_i \alpha_i y_i \mathbf{x}_i$，支持向量满足 $\alpha_i > 0$。

### 软间隔原始问题  
$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i,\; \xi_i \geq 0
$$

### 对偶问题（软间隔）  
$$
\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^\top \mathbf{x}_j) - \sum_i \alpha_i \\
\text{s.t.} \quad 0 \leq \alpha_i \leq C,\; \sum_i \alpha_i y_i = 0
$$

### 决策函数  
$$
f(\mathbf{x}) = \operatorname{sign}\left( \sum_{i \in S} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b \right), \quad \text{其中 } S = \{i \mid \alpha_i > 0\}
$$  
偏置项 $b$ 可由任一支持向量 $s$ 计算：$b = y_s - \sum_{i \in S} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x}_s$，或取均值提升稳定性。

### 常用核函数  
| 核函数 | 表达式 | 参数 |
|--------|--------|------|
| 多项式核（Polynomial） | $(\mathbf{x}_1^\top \mathbf{x}_2 + 1)^d$ | $d \in \mathbb{Z}^+$ |
| 高斯径向基核（RBF/Gaussian） | $\exp\left(-\frac{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}{2\delta^2}\right)$ | $\delta > 0$ |
| 拉普拉斯核（Laplacian） | $\exp\left(-\frac{\|\mathbf{x}_1 - \mathbf{x}_2\|}{\delta^2}\right)$ | $\delta > 0$ |
| 双曲正切核（Fisher/tanh） | $\tanh(\beta \mathbf{x}_1^\top \mathbf{x}_2 + \theta)$ | $\beta > 0,\; \theta < 0$ |

## 与其他概念的关系  
- **与 Logistic Regression 对比**：二者均为线性分类器，但 SVM 基于几何间隔最大化（判别式），LR 基于概率建模（$P(y=1|\mathbf{x})$）和最大似然估计；SVM 解稀疏（仅支持向量），LR 解稠密（全样本参与）。  
- **与 Linear Discriminant Analysis（LDA）对比**：LDA 是生成式方法，假设类条件分布为高斯且共享协方差；SVM 是判别式方法，不建模数据分布，更关注边界附近的样本。  
- **与 k-Nearest Neighbor（kNN）对比**：kNN 是惰性学习（无显式训练），决策依赖全部邻近样本；SVM 是积极学习，训练后仅保留支持向量，预测高效。  
- **与 Decision Trees 对比**：决策树产生轴平行的分段常数决策边界；SVM（尤其使用 RBF 核时）可生成复杂、光滑的非线性边界，且对特征缩放更敏感。  
- **与 Naive Bayes 对比**：Naive Bayes 基于贝叶斯定理和条件独立假设，输出概率；SVM 不提供概率解释（需额外校准），但对特征相关性鲁棒性更强。  

> 注：资料中未充分覆盖该主题 —— SVM 的概率输出校准（如 Platt scaling）、多分类策略（one-vs-one / one-vs-rest）、以及与深度学习的结合等内容未在课程资料中出现。