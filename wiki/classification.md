# Classification

## 核心定义
- **Classification** 是监督学习（supervised learning）的一种，其目标是根据输入特征 $x$ 预测离散的类别标签 $y$（也称为响应 $G$ 或输出 $Y$），即构建一个分类器（classifier）函数 $y = f(x)$。
- 分类问题中，输出变量 $Y$ 取值于有限的离散集合 $\mathbf{G} = \{1, 2, ..., K\}$，例如：预测肿瘤是良性还是恶性、预测用户是否会违约、区分垃圾邮件与正常邮件等。
- 最优分类规则由贝叶斯决策理论给出：$f^*(x) = \arg\max_c P(Y = c|X = x)$，即选择在给定 $x$ 下后验概率最大的类别。

## 训练/预测流程
1. **训练阶段（Training Stage）**：
   - 给定带标签的数据集 $D = \{(x_i, y_i)\}_{i=1}^N$，将其划分为训练集 $D_{\text{train}}$ 和测试集 $D_{\text{test}}$。
   - 在训练集上寻找一个分类器 $f(x)$，使其能最好地拟合训练数据，即最小化经验风险（如误分类率）。
2. **预测阶段（Predicting Stage）**：
   - 对于无标签的新样本 $x_{\text{pred}}$，应用训练好的分类器 $f(x)$ 得到预测标签 $y_{\text{pred}} = f(x_{\text{pred}})$。
   - 对于多类问题，通常将每个类别 $k$ 的预测转化为一个 $K$ 维向量，再通过最大后验概率或加权投票进行最终分类。

## 关键要点
- **核心思想**：不同模型对 $f$ 的假设不同，导致不同的分类方法。常见模型包括 Logistic Regression、k-Nearest Neighbor (kNN)、Decision Trees、Naive Bayes、Linear Discriminant Analysis (LDA)、Support Vector Machine (SVM) 等。
- **贝叶斯最优性**：贝叶斯分类器 $f^*(x)$ 基于后验概率 $P(Y=c|X=x)$，其错误率（Bayes error rate）为 $\inf_f E(f) = 1 - P(Y = f^*(X))$。
- **决策边界（Decision Boundary）**：分类器将输入空间划分为若干区域，每个区域对应一个类别。例如，Logistic Regression 和 LDA 的决策边界是线性的；kNN 的边界随 $k$ 增大而更平滑；Decision Tree 的边界是平行于坐标轴的矩形分割；SVM 的边界是最大化间隔的超平面。
- **评估指标**：
  - **混淆矩阵（Confusion Matrix）**：用于二分类，包含 TP（真阳性）、TN（真阴性）、FP（假阳性）、FN（假阴性）。
  - **准确率（Accuracy）**：$\frac{TP+TN}{TP+TN+FP+FN}$，但在样本不平衡时不可靠。
  - **精确率（Precision）**：$\frac{TP}{TP+FP}$，衡量预测为正例的样本中真正为正例的比例。
  - **召回率（Recall / Sensitivity）**：$\frac{TP}{TP+FN}$，衡量所有真实正例中被正确识别的比例。
  - **F1 分数（F1 score）**：精确率和召回率的调和平均。
  - **ROC 曲线与 AUC**：通过设定不同阈值 $t$，绘制 TPR（召回率）vs. FPR（1-特异度），AUC 越大表示模型越鲁棒。
- **偏差-方差权衡（Bias–Variance Tradeoﬀ）**：kNN 中，小 $k$ 导致低偏差、高方差（过拟合）；大 $k$ 导致高偏差、低方差（欠拟合）。该权衡是模型选择的核心。

## 与其他概念的关系
- **与回归（Regression）的关系**：两者同属监督学习，但回归预测连续值，分类预测离散标签。某些分类方法（如 Logistic Regression）可视为对指示变量（indicator matrix）的回归。
- **与维度约减（Dimensionality Reduction）的关系**：LDA 是一种有监督的维度约减方法，它通过投影使类间散度最大化、类内散度最小化；PCA 是无监督的，仅基于协方差矩阵寻找最大方差方向。
- **与集成学习（Ensemble Learning）的关系**：Bagging（如 Random Forest）和 Boosting（如 AdaBoost）均通过组合多个基学习器（base learners）来提升性能。Random Forest 通过随机抽样和随机特征选择降低树之间的相关性；AdaBoost 则通过重采样关注前一轮中被误分类的样本，以降低偏差。
- **与神经网络（Neural Network）的关系**：单层感知机（perceptron）本质上是一个线性分类器；深度神经网络则通过非线性激活函数和多层结构实现更复杂的分类能力，其损失函数（如交叉熵）与 Logistic Regression 直接相关。
- **与密度估计（Density Estimation）的关系**：Kernel Density Classification 和 Naive Bayes 均基于对各类别密度 $f_k(x)$ 的估计，并结合先验概率 $\pi_k$，利用贝叶斯定理计算后验概率 $P(G=k|X=x)$。

## 资料中未充分覆盖的部分
- **深度学习中的分类任务**：资料中虽提及 CNN、RNN 等架构，但未系统阐述其在分类任务中的端到端训练流程、反向传播细节、以及如何从 softmax 输出导出概率和决策边界。
- **评价指标的统计显著性检验**：资料介绍了 McNemar 检验（Ex. 10.6）和 Wilcoxon 检验（Ex. 15.4），但未提供一个完整的框架来比较不同分类器的性能差异是否具有统计意义。
- **在线学习（Online Learning）与流式分类（Streaming Classification）**：资料中讨论了批量学习（batch learning）下的各种算法，但未涉及数据动态到达场景下的增量式更新策略。
- **类别不平衡（Class Imbalance）的专门处理技术**：除 ROC/AUC 外，资料未深入介绍代价敏感学习（cost-sensitive learning）、SMOTE 过采样、或焦点损失（focal loss）等现代方法。
- **可解释性（Interpretability）的量化评估**：虽然提到了特征重要性（feature importance）和部分依赖图（partial dependence plots），但未涵盖 SHAP、LIME 等局部可解释性方法的原理与应用。