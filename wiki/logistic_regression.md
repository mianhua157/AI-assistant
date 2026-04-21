# Logistic Regression

## 核心定义
- Logistic regression 是一种分类方法，而非回归方法。
- 其核心思想是：对二元响应变量 $y$（取值为 0 或 1），线性回归模型给出的拟合值 $\hat{y} = x^T \hat{\beta}$ 可以解释为后验概率 $P(y=1|x)$ 的估计；但该拟合值可能超出 $[0,1]$ 区间，不满足概率约束。
- 因此，logistic regression 引入 sigmoid 函数 $g(z) = \frac{1}{1+e^{-z}}$ 将线性组合映射回 $[0,1]$ 区间，其中 $z = w_0 + w_1x_1 + \dots + w_dx_d$。
- 等价地，它建模的是 logit（对数几率）变换：$\log \frac{P(y=1|x)}{1-P(y=1|x)} = w_0 + w_1x_1 + \dots + w_dx_d$。
- 对于 $K$ 类问题，其扩展形式为：
  $$
  \log \frac{P(y = k|x)}{P(y = K|x)} = w_k^T x, \quad k = 1,2,\dots,K-1
  $$
  概率模型为：
  $$
  P(y = k|x) = \frac{e^{w_k^T x}}{1 + \sum_{k=1}^{K-1} e^{w_k^T x}}, \quad k = 1,2,\dots,K-1
  $$

## 训练/预测流程
- **训练**：通过最大似然估计（MLE）求解参数 $w$。定义似然函数：
  $$
  L(w) = \prod_{i=1}^n p_{y_i}(x_i; w)
  $$
  其中 $p_k(x; w) = P(y=k|X=x)$。MLE 估计为 $\hat{w} = \arg\max_w L(w)$。
- **优化方法**：使用 Newton-Raphson 方法求解 $\nabla_w \log L(w) = 0$。
- **预测**：对于新样本 $x$，计算所有类别的后验概率 $P(y=k|x)$，并选择概率最大的类别作为预测结果：
  $$
  \hat{y} = \arg\max_k P(y=k|x)
  $$

## 关键要点
- **与 LDA 的关系**：logistic regression 和 LDA 都能产生线性决策边界，但估计方式不同。LDA 假设各类别服从高斯分布且协方差矩阵相同，而 logistic regression 假设更少，仅要求后验概率具有 logit-linear 形式，因此更鲁棒。
- **与线性回归的关系**：当数据可被完美分离时，logistic regression 的最大似然估计会发散（趋于无穷），而 LDA 的系数则有明确定义，说明 logistic regression 在强假设下可能失效。
- **正则化**：可引入 $L_1$ 正则化（Lasso）进行变量选择和收缩：
  $$
  \max_{\beta_0,\beta}
  \left\{
  \sum_{i=1}^N [y_i(\beta_0 + \beta^T x_i) - \log(1 + e^{\beta_0+\beta^T x_i})]
  - \lambda \sum_{j=1}^p |\beta_j|
  \right\}
  $$
- **多类扩展**：通过 multilogit（multinomial logistic regression）建模，即对 $K$ 类问题，将第 $K$ 类作为参考类，其余 $K-1$ 类分别建模。

## 与其他概念的关系
- **与 Bayes 分类器的关系**：Bayes 分类器基于后验概率 $P(Y=c|X=x)$ 进行决策，而 logistic regression 直接建模该后验概率（或其单调变换），其目标是逼近 $E[Y|X]$。
- **与 LDA 的关系**：二者在形式上一致（log-posterior odds 均为 $x$ 的线性函数），但 LDA 基于联合密度建模，logistic regression 基于条件密度建模。LDA 更高效（低方差），但依赖更强假设；logistic regression 更鲁棒，但效率略低。
- **与 SVM 的关系**：SVM 的最优分离超平面与 logistic regression 的最大间隔方向类似，但 logistic regression 使用概率输出，SVM 则直接优化几何间隔。
- **与神经网络的关系**：单层感知机（Perceptron）的输出层若采用 sigmoid 激活函数，则等价于 logistic regression；多层网络可视为 logistic regression 的非线性推广。

## 资料中未充分覆盖的部分
- 未详细讨论 logistic regression 的具体实现细节（如 `glm`、`glmnet` 等软件包的用法）。
- 未提供 logistic regression 的典型应用场景（如信用评分、疾病风险预测）的具体案例分析。
- 未涉及 logistic regression 的特征工程策略（如多项式特征、交互项、分箱等）及其对模型性能的影响。
- 未明确说明如何处理 logistic regression 中的多重共线性问题（如 LDA 中的 sphering 技术）。
- 未提及 logistic regression 的校准（calibration）方法（如 Platt scaling、isotonic regression）及评估其概率输出质量的指标（如 Brier score）。