"""
RAG 系统优化对比测试脚本

测试 20 个问题，对比三种方案的效果：
1. Baseline（无优化）- 直接用原始问题检索 + 简单 prompt
2. + Query Rewrite - 添加查询重写
3. + Prompt 优化 - 完整优化版本
"""

import os
from http import HTTPStatus
import dashscope
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# API 配置 - 从环境变量读取
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 加载 FAISS 向量存储
# 注意：向量库已离线构建，此处仅演示加载流程
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# 测试问题列表（20 个机器学习相关问题）
TEST_QUESTIONS = [
    "What is regression?",
    "What is classification?",
    "What is overfitting?",
    "What is k-nearest neighbor?",
    "What is decision tree?",
    "What is logistic regression?",
    "What is support vector machine?",
    "What is naive bayes?",
    "What is linear discriminant analysis?",
    "What is cross-validation?",
    "What is precision and recall?",
    "What is F1 score?",
    "What is confusion matrix?",
    "What is bias-variance tradeoff?",
    "What is regularization?",
    "What is gradient descent?",
    "What is feature engineering?",
    "What is ensemble learning?",
    "What is bagging and boosting?",
    "What is neural network?",
]


def evaluate_answer(answer: str) -> bool:
    """
    简单评估答案是否有效
    规则：
    - 不是"我不知道"或类似拒答
    - 长度超过 20 字
    """
    if not answer:
        return False

    answer_lower = answer.lower()

    # 检查是否是拒答
    refuse_phrases = ["我不知道", "i don't know", "无法回答", "资料不足", "没有相关信息"]
    for phrase in refuse_phrases:
        if phrase in answer_lower:
            return False

    # 检查长度
    if len(answer) < 20:
        return False

    return True


# ==================== 方案 1: Baseline（无优化）====================
def ask_rag_baseline(question, k=5):
    """基础版本 - 无优化"""
    docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return {"answer": "未找到相关文档", "success": False}

    context = "\n\n".join([doc.page_content[:300] for doc in docs])

    prompt = f"""根据以下资料回答问题：
{context}

问题：{question}
回答："""

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0.5
    )

    if response.status_code == HTTPStatus.OK:
        answer = response.output.text
        success = evaluate_answer(answer)
        return {"answer": answer, "success": success}
    else:
        return {"answer": f"请求失败：{response.message}", "success": False}


# ==================== 方案 2: + Query Rewrite ====================
def rewrite_query(question: str) -> str:
    """重写查询以提高语义搜索效果"""
    prompt = f"""Rewrite the following question to improve semantic search.
Add related terms, synonyms, and more specific expressions.
Keep it concise but informative.

Original question:
{question}

Rewritten query:"""

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.text.strip()
    return question


def ask_rag_with_rewrite(question, k=5):
    """方案 2 - 添加 Query Rewrite"""
    new_query = rewrite_query(question)
    docs = vectorstore.similarity_search(new_query, k=k)

    if not docs:
        return {"answer": "未找到相关文档", "success": False, "rewritten_query": new_query}

    context = "\n\n".join([doc.page_content[:300] for doc in docs])

    prompt = f"""根据以下资料回答问题：
{context}

问题：{question}
回答："""

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0.5
    )

    if response.status_code == HTTPStatus.OK:
        answer = response.output.text
        success = evaluate_answer(answer)
        return {"answer": answer, "success": success, "rewritten_query": new_query}
    else:
        return {"answer": f"请求失败：{response.message}", "success": False}


# ==================== 方案 3: + Prompt 优化（完整版）====================
def translate_to_chinese(text: str) -> str:
    """将英文资料翻译成中文"""
    prompt = f"请把以下内容翻译成中文：\n\n{text}"

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.text.strip()
    return text


def ask_rag_full_optimized(question, k=5):
    """方案 3 - 完整优化版本（Query Rewrite + Prompt 优化 + 翻译）"""
    new_query = rewrite_query(question)
    docs = vectorstore.similarity_search(new_query, k=k)

    if not docs:
        return {"answer": "未找到相关文档", "success": False, "rewritten_query": new_query}

    context = "\n\n".join([doc.page_content[:300] for doc in docs])
    context_cn = translate_to_chinese(context)

    prompt = f"""你是一个机器学习助教，请根据提供的资料回答问题。

规则：
1. 必须全部使用中文回答（不能出现英文句子）
2. 如果资料是英文，请翻译后再回答
3. 优先使用资料内容
4. 可以进行合理补充解释
5. 不要轻易回答"我不知道"

资料：
{context_cn}

问题：
{question}

请用中文回答："""

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0.1,
        max_tokens=500
    )

    if response.status_code == HTTPStatus.OK:
        answer = response.output.text
        success = evaluate_answer(answer)
        return {"answer": answer, "success": success, "rewritten_query": new_query}
    else:
        return {"answer": f"请求失败：{response.message}", "success": False}


# ==================== 运行测试 ====================
def run_benchmark():
    """运行完整的基准测试"""
    print("=" * 60)
    print("RAG 系统优化对比测试")
    print("=" * 60)
    print(f"测试问题数量：{len(TEST_QUESTIONS)}")
    print()

    results = {
        "baseline": {"success": 0, "fail": 0, "details": []},
        "with_rewrite": {"success": 0, "fail": 0, "details": []},
        "full_optimized": {"success": 0, "fail": 0, "details": []},
    }

    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] 问题：{q}")

        # 方案 1
        print("  测试 Baseline...", end=" ")
        r1 = ask_rag_baseline(q)
        results["baseline"]["details"].append({
            "question": q,
            "success": r1["success"],
            "answer_preview": r1["answer"][:50] if r1["answer"] else ""
        })
        if r1["success"]:
            results["baseline"]["success"] += 1
            print("✅")
        else:
            results["baseline"]["fail"] += 1
            print("❌")

        # 方案 2
        print("  测试 +Query Rewrite...", end=" ")
        r2 = ask_rag_with_rewrite(q)
        results["with_rewrite"]["details"].append({
            "question": q,
            "success": r2["success"],
            "answer_preview": r2["answer"][:50] if r2["answer"] else ""
        })
        if r2["success"]:
            results["with_rewrite"]["success"] += 1
            print("✅")
        else:
            results["with_rewrite"]["fail"] += 1
            print("❌")

        # 方案 3
        print("  测试 完整优化...", end=" ")
        r3 = ask_rag_full_optimized(q)
        results["full_optimized"]["details"].append({
            "question": q,
            "success": r3["success"],
            "answer_preview": r3["answer"][:50] if r3["answer"] else ""
        })
        if r3["success"]:
            results["full_optimized"]["success"] += 1
            print("✅")
        else:
            results["full_optimized"]["fail"] += 1
            print("❌")

    # 计算比例
    total = len(TEST_QUESTIONS)
    summary = {
        "baseline": round(results["baseline"]["success"] / total * 100),
        "with_rewrite": round(results["with_rewrite"]["success"] / total * 100),
        "full_optimized": round(results["full_optimized"]["success"] / total * 100),
    }

    return results, summary


def generate_readme_table(summary: dict) -> str:
    """生成 README 中的对比表格"""
    table = """| 方法 | 能回答比例 |
|----------------------|------------|
| baseline（无优化） | {}% |
| + query rewrite | {}% |
| + prompt 优化 | {}% |""".format(
        summary['baseline'],
        summary['with_rewrite'],
        summary['full_optimized']
    )
    return table


def update_readme(table: str, readme_path: str = "README.md"):
    """更新 README.md 文件中的表格"""
    import os

    if not os.path.exists(readme_path):
        print(f"警告：{readme_path} 不存在，将创建新文件")
        content = f"# RAG 系统优化对比测试\n\n## 测试结果\n\n{table}\n"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 查找并替换现有的表格部分
    import re
    pattern = r'(\| 方法.*?\|.*?能回答比例.*?\n\|.*?\n(\|.*?\n)+)'

    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, table, content, flags=re.MULTILINE)
        print("✓ 已更新 README.md 中的表格")
    else:
        # 如果没有找到现有表格，在"优化效果对比"标题下插入
        if "### 优化效果对比" in content:
            content = content.replace(
                "### 优化效果对比\n",
                f"### 优化效果对比\n\n{table}\n"
            )
            print("✓ 已在 README.md 中添加表格")
        else:
            # 添加到文件末尾
            content += f"\n## 测试结果\n\n{table}\n"
            print("✓ 已在 README.md 末尾添加表格")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    results, summary = run_benchmark()

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print()
    # 简洁版表格（匹配图片格式）
    print("| 方法 | 能回答比例 |")
    print("|----------------------|------------|")
    print(f"| baseline（无优化） | {summary['baseline']}% |")
    print(f"| + query rewrite | {summary['with_rewrite']}% |")
    print(f"| + prompt 优化 | {summary['full_optimized']}% |")
    print()
    print("测试完成！")

    # 保存详细结果到 JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, ensure_ascii=False, indent=2)
    print("✓ 详细结果已保存到 benchmark_results.json")

    # 生成并更新 README 表格
    table = generate_readme_table(summary)
    update_readme(table)
    print("✓ README.md 已更新")
