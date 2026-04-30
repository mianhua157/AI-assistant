\# RAG QA Evaluation Questions



本问题集用于测试机器学习 RAG 问答系统的检索能力、生成质量、跨语言鲁棒性和边界处理能力。



\---



\## 1. 高频 Query



这些是用户最可能经常问的问题，必须稳定回答。



```text

What is classification?

What is regression?

What is supervised learning?

What is logistic regression?

What is k-nearest neighbor?

What is decision tree?

What is Naive Bayes?

What is support vector machine?

什么是分类？

什么是回归？

什么是监督学习？

什么是逻辑回归？

什么是 KNN？

什么是决策树？

什么是朴素贝叶斯？

什么是支持向量机？



\\## 2. 长尾 Query



这些问题频率较低，但可以测试系统对复杂表达和泛化问题的处理能力。



Why is logistic regression used for classification?

How does k-nearest neighbor make predictions?

How does a decision tree choose split points?

Why can SVM handle non-linear classification with kernels?

What assumptions does Naive Bayes make?

What is the relationship between classification and supervised learning?

Why is classification output discrete?

How is model performance evaluated in classification tasks?

KNN 和 decision tree 在 inductive bias 上有什么区别？

为什么 logistic regression 也叫 regression，却常用于 classification？

SVM 的 margin 是什么，为什么重要？

朴素贝叶斯为什么叫 naive？

决策树为什么容易过拟合？

分类任务为什么常用 accuracy、precision、recall 和 F1？



\## 3. Easy / Medium / Hard 三档
### Easy
答案比较直接，资料中通常能直接找到对应定义。

What is classification?
What is supervised learning?
What is k-nearest neighbor?
What is decision tree?
什么是分类？
什么是监督学习？
什么是决策树？
什么是 KNN？

### Medium
需要系统理解并改写资料，而不是直接复制原文。

Compare classification and regression.
What is the difference between logistic regression and linear regression?
Compare kNN and decision tree.
Compare Naive Bayes and SVM.
Why is logistic regression a classification algorithm?
分类和回归的区别是什么？
逻辑回归和线性回归有什么区别？
KNN 和决策树有什么区别？
朴素贝叶斯和 SVM 有什么区别？

### Hard
需要跨多个知识点综合，测试系统是否能真正组织答案。

Compare common classification algorithms in terms of assumptions and decision boundaries.
Why do different classification algorithms have different inductive biases?
How do loss functions differ between classification and regression?
Why can the same model family sometimes be used for both classification and regression?
Explain how classification methods differ in training objective, output space, and evaluation metrics.
为什么不同分类算法的决策边界不同？
分类和回归在输出空间、损失函数和评估指标上有什么系统性差异？
为什么有些算法既可以做分类，也可以做回归？
从模型假设、训练目标和预测输出三个角度比较分类和回归。

\## 4. OOD 或跨域样本
OOD 指 Out-of-Distribution，即问题和当前课程资料分布不完全一致。
这些问题用于测试系统是否会乱编，以及能否合理说明资料不足。

What is dropout in deep learning?
How is transformer related to classification?
What is reinforcement learning?
What is clustering?
What is PCA used for?
What is overfitting in neural networks?
How does attention mechanism work?
什么是强化学习？
什么是聚类？
PCA 和分类有什么关系？
dropout 和过拟合有什么关系？
Transformer 可以用于分类任务吗？
深度学习中的 batch normalization 是什么？

\## 5. Hard Negatives
Hard negatives 是“看起来相关，但容易检索到错误内容”的问题。
它们可以测试系统是否会被相似概念干扰。

What is logistic regression?
What is linear regression?
What is the difference between logistic regression and linear regression?
What is precision?
What is recall?
What is the difference between precision and recall?
What is accuracy?
What is AUC?
What is LDA?
What is PCA?
What is the difference between LDA and PCA?
What is SVM?
What is SVR?
What is the difference between SVM and SVR?
什么是逻辑回归？
什么是线性回归？
逻辑回归和线性回归有什么区别？
precision 和 recall 的区别是什么？
accuracy 和 AUC 的区别是什么？
LDA 和 PCA 有什么区别？
SVM 和 SVR 有什么区别？

\## 6. 中英混合 Query
用于测试 Query Rewrite 和跨语言检索能力。

classification 是什么？
regression 是什么？
What is 分类?
Compare 分类 and regression.
classification 和 回归 有什么区别？
Why logistic regression 用于分类？
SVM 的 margin 是什么？
What is 朴素贝叶斯?
KNN 怎么 make prediction?
What is overfitting 和 model complexity 的关系？

\## 7. Source Grounding 测试
这些问题用于检查回答是否正确引用来源，以及是否把 wiki 和 raw 的作用区分开。

According to the provided materials, what is classification?
Based on the retrieved sources, summarize classification.
Use the course materials to explain kNN.
根据课程资料解释什么是 classification。
请基于检索到的资料总结 decision tree。
请说明你的回答主要来自 wiki 还是 raw PDF。

\## 8. 推荐固定测试子集
每次修改系统后，建议固定测试以下 10 个问题：

What is classification?
What is regression?
Compare classification and regression.
What is logistic regression?
Why is logistic regression used for classification?
What is k-nearest neighbor?
Compare kNN and decision tree.
What is Naive Bayes?
What is support vector machine?
什么是分类？






