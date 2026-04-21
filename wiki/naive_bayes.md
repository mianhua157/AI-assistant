# Naive Bayes

## 核心定义
Naive Bayes 是一种基于贝叶斯定理（Bayes Theorem）和**条件独立性假设**（conditional independency assumption）的分类器。其核心思想是估计后验概率 $P(Y|X)$，其中 $X = \{X_1, \dots, X_d\}$ 为特征向量。  
根据贝叶s 定理：  
$$
P(Y|X = x) \propto P(X = x|Y) P(Y)
$$  
Naive Bayes 进一步假设各特征在给定类别 $Y = c$ 下相互独立：  
$$
P(X = x|Y = c) = \prod_{i=1}^{d} P(X_i = x_i|Y = c)
$$  
因此，Naive Bayes 模型的预测规则为：  
$$
\hat{y} = \arg\max_{c} \left[ P(Y = c) \prod_{i=1}^{d} P(X_i = x_i|Y = c) \right]
$$  
该模型也被称为“Idiot’s Bayes”，但因其在高维稀疏数据（如文本）中表现稳健而被广泛使用。

## 训练/预测流程
- **训练阶段**：  
  - 使用最大似然估计（MLE）从训练集 $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$ 中估计参数：  
    - 先验概率：$P(Y = c) = \frac{\sum_{i=1}^n I(y_i = c)}{n}$  
    - 条件概率：  
      - 若 $X_i$ 为离散变量（取值 $\{v_1, \dots, v_K\}$），则  
        $P(X_i = v_k|Y = c) = \frac{\sum_{i=1}^n I(x_{ij} = v_k, y_i = c)}{\sum_{i=1}^n I(y_i = c)}$  
      - 若 $X_i$ 为连续变量，则常假设其服从正态分布 $N(\mu, \sigma^2)$，并用 MLE 估计 $\mu$ 和 $\sigma^2$；或先离散化再按离散方法处理。  

- **预测阶段**：  
  对于新样本 $x = (x_1, \dots, x_d)$，计算每个类别 $c$ 的联合得分：  
  $$
  \text{score}(c) = P(Y = c) \prod_{i=1}^{d} P(X_i = x_i|Y = c)
  $$  
  最终预测标签为：$\hat{y} = \arg\max_c \text{score}(c)$。

## 关键要点
- **适用场景**：广泛用于文本分析、垃圾邮件过滤（spam filtering）、推荐系统与医学诊断。
- **稳定性与鲁棒性**：对异常值（outliers）和缺失值（missing values）稳定；对不相关特征鲁棒（因 $P(X_i|Y)$ 独立于 $Y$，不影响后验概率）。
- **性能特点**：即使条件独立性假设不成立，Naive Bayes 仍可能优于更复杂的模型；但当该假设严重违反时，性能会下降。
- **参数依赖性**：性能高度依赖参数估计质量（如 $P(X_i = x_i|Y = c)$ 的估计是否准确）。
- **与 LDA 的联系**：Naive Bayes 可视为 LDA 的特例——它假设各类别内特征服从独立同方差高斯分布（即协方程矩阵为对角阵且相同），从而简化密度估计。

## 与其他概念的关系
- **与 Bayes 分类器（Oracle 分类器）的关系**：Naive Bayes 是 Bayes 分类器的近似实现，通过引入条件独立性假设来降低 $P(X|Y)$ 的估计难度；Bayes 分类器本身需直接估计 $P(Y|X)$，而 Naive Bayes 将其分解为先验 $P(Y)$ 和类条件似然 $P(X|Y)$ 的乘积形式。
- **与 Logistic Regression 的关系**：两者均建模 $P(Y|X)$，但 Logistic Regression 直接拟合 logit 函数（无分布假设），而 Naive Bayes 基于生成式建模（显式假设 $P(X_i|Y)$ 形式）。Logistic Regression 更灵活，但 Naive Bayes 在高维下更易估计。
- **与 Kernel Density Classification 的关系**：Naive Bayes 是 kernel density classification 的一种特例，其中核函数退化为各维度上的独立一维核（如 Gaussian 或 histogram），且不考虑特征间交互。
- **与 Generalized Additive Models（GAM）的关系**：Naive Bayes 的 logit 变换可导出广义加性模型形式：  
  $$
  \log \frac{P(G = \ell|X)}{P(G = J|X)} = \log \frac{\pi_\ell}{\pi_J} + \sum_{k=1}^p \log \frac{f_{\ell k}(X_k)}{f_{Jk}(X_k)} = \alpha_\ell + \sum_{k=1}^p g_{\ell k}(X_k)
  $$  
  因此，Naive Bayes 可看作 GAM 的一种特殊形式（各 $g_{\ell k}(X_k)$ 由 $P(X_k|Y)$ 决定）。
- **与 Dimensionality Reduction 的关系**：Naive Bayes 不进行降维，而是通过条件独立性假设将高维联合密度估计解耦为多个低维（甚至一维）密度估计，从而规避维数灾难（curse of dimensionality）。

## 资料中未充分覆盖的部分
- **平滑与拉普拉斯校正**（Laplace smoothing）：资料未提及对离散条件概率 $P(X_i = v_k|Y = c)$ 的零频次问题（即未观测到某词在某类中出现）如何处理，这是文本分类中 Naive Bayes 的关键实践技巧。
- **特征工程与文本预处理**：资料仅简要提到 word2vec，但未说明 Naive Bayes 在文本任务中通常依赖词袋（bag-of-words）、TF-IDF 加权、停用词去除、词干提取等前处理步骤。
- **多类扩展细节**：虽提及 K-class Logistic Regression，但未给出 Naive Bayes 多类情形下的完整 logit 推导及归一化策略（如 softmax vs. one-vs-rest）。
- **朴素贝叶斯的变体**：如半朴素贝ays（semi-naive Bayes）、树增强朴素贝叶斯（TAN）等改进模型未被涉及。
- **与决策树的对比机制**：资料未明确指出 Naive Bayes 与 ID3/C4.5 等树算法在信息增益（information gain）与 Gini 指标上的本质差异——前者假设独立性以简化似然，后者通过贪心分裂显式优化 impurity。