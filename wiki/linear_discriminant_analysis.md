# Linear Discriminant Analysis

## 核心定义

- LDA 是一种监督学习方法，基于类别标签进行线性投影，以在低维空间中最大化类间点散度（between-class point scatter）并最小化类内点散度（within-class point scatter）。
- 其核心思想是：假设每个类别的样本服从多元高斯分布 $f_k(x) = P(X = x \mid Y = k)$，且各类别共享一个协方程矩阵 $\Sigma_k = \Sigma$；通过贝叶斯定理，最优分类器为 $k^* = \arg\max_k P(Y = k \mid X)$，其判别函数为线性形式：
  $$
  \delta_k(x) = x^T \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k + \log \pi_k
  $$
- 其中 $\pi_k = P(Y = k)$ 为先验概率，$\mu_k = \frac{1}{n_k} \sum_{i: x_i \in C_k} x_i$ 为第 $k$ 类的均值向量，$\mu = \sum_{k=1}^K \frac{n_k}{n} \mu_k$ 为所有样本的总体均值。

## 训练/预测流程

1. **参数估计**（使用训练数据）：
   - 估计先验概率：$\hat{\pi}_k = N_k / N$，其中 $N_k$ 是第 $k$ 类样本数，$N = \sum_{k=1}^K N_k$；
   - 估计类条件均值：$\hat{\mu}_k = \frac{1}{N_k} \sum_{y_i = k} x_i$；
   - 估计共同协方差矩阵：$\hat{\Sigma} = \frac{1}{N - K} \sum_{k=1}^K \sum_{y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$。

2. **判别函数计算**：
   - 对于任意输入 $x$，计算各线性判别函数 $\delta_k(x)$；
   - 分类决策：$k^* = \arg\max_k \delta_k(x)$。

3. **计算优化技巧**（“Sphere the data”）：
   - 对数据进行球化（whitening）变换：$X^* = D^{-1/2} U^T X$，其中 $\hat{\Sigma} = U D U^T$ 是特征分解；
   - 在变换后的空间中，LDA 等价于寻找离 $x$ 最近的类中心（考虑先验 $\pi_k$），即对球化后数据做最近质心分类。

## 关键要点

- **Fisher 准则**：LDA 的几何本质是寻找投影方向 $a$，使得类间方差与类内方差之比（Rayleigh 商）最大：
  $$
  \max_a \frac{a^T B a}{a^T W a}, \quad \text{其中 } B \text{ 为类间协方差，} W \text{ 为类内协方差}
  $$
- **降维上限**：$K$ 类问题下，LDA 最多可产生 $K-1$ 个非退化判别方向（canonical variates），因此存在根本性的维度压缩。
- **与 PCA 的区别**：
  - PCA 是无监督方法，仅基于样本协方差矩阵，寻找最大方差方向；
  - LDA 是有监督方法，利用类别信息，寻找使类间分离最显著的方向。
- **二分类特例**：当 $K=2$ 时，LDA 判别规则等价于线性回归对指示变量的拟合（Exercise 4.2），但截距项不同；判别方向为 $\beta = \hat{\Sigma}^{-1}(\hat{\mu}_2 - \hat{\mu}_1)$。
- **鲁棒性权衡**：LDA 假设强（高斯+同协方差），若成立则估计效率更高；但若违反（如存在粗大异常值），则不如 Logistic Regression 鲁棒。

## 与其他概念的关系

- **与 Bayes 分类器的关系**：LDA 是 Bayes 分类器在高斯类条件密度和共同协方差假设下的具体实现；Bayes 错分率 $1 - \Phi\left( \beta^T (\mu_2 - \mu_1) / (\beta^T \Sigma \beta)^{1/2} \right)$ 给出了理论下界。
- **与 Logistic Regression 的关系**：二者都导出线性判别边界，但推导路径不同——Logistic Regression 直接建模后验概率 $P(Y=k \mid X=x)$，而 LDA 廾建模类条件密度 $f_k(x)$ 和先验 $\pi_k$。当真实分布满足 LDA 假设时，LDA 更高效；否则 Logistic Regression 更稳健（Section 4.4.5）。
- **与 QDA 的关系**：QDA 放弃了 LDA 的共同协方差假设，允许每类使用独立协方差矩阵 $\Sigma_k$，从而得到二次判别函数 $\delta_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2}(x - \theta_k)^T \Sigma_k^{-1}(x - \theta_k) + \log \pi_k$。
- **与 RDA 的关系**：正则化判别分析（Regularized Discriminant Analysis）是 LDA 与 QDA 的折中，通过收缩 $\hat{\Sigma}_k$ 向 $\hat{\Sigma}$ 实现：$\hat{\Sigma}_k(\alpha) = \alpha \hat{\Sigma}_k + (1 - \alpha)\hat{\Sigma}$，其中 $\alpha \in [0,1]$ 控制收缩强度。
- **与 Reduced-Rank LDA 的关系**：LDA 天然支持降维，其前 $L < K-1$ 个判别坐标（canonical variates）构成最优子空间，用于后续分类或可视化（Section 4.3.3）。

## 资料中未充分覆盖的部分

- **LDA 的实际应用场景细节**（如图像识别、基因表达分析中的具体预处理步骤、标准化策略、缺失值处理）未展开说明；
- **LDA 的模型评估指标**（如混淆矩阵、ROC/AUC、Cohen’s Kappa）虽在课程其他章节提及，但未明确关联到 LDA 模型性能量化；
- **LDA 的软件实现与调参经验**（如 `R` 中 `MASS::lda` 或 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 的超参数含义、交叉验证策略、过拟合诊断）未提供；
- **LDA 的扩展变体**（如 Penalized Discriminant Analysis、Flexible Discriminant Analysis）仅简要提及，缺乏数学推导与算法流程；
- **LDA 在高维小样本（$p \gg N$）下的失效机制及应对方案**（如 Diagonal LDA、Nearest Shrunken Centroids）虽在另一份资料中出现，但未整合进本主题页的统一框架。