import os
from http import HTTPStatus
import dashscope
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# API 配置 - 从环境变量读取（避免硬编码泄露）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 调试模式开关
DEBUG = False  # 设为 True 时打印详细调试信息


# 加载已保存的 FAISS 向量存储
# 注意：向量库已离线构建，此处仅演示加载流程
# 实际使用的 embedding 模型与 FAISS 索引创建时保持一致
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)


def rewrite_query(question: str) -> str:
    """重写查询以提高语义搜索效果

    添加相关术语、同义词和更具体的表达。
    保持简洁但有信息量。
    """
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
    else:
        # 如果重写失败，返回原始问题
        if DEBUG:
            print(f"⚠️ Query 重写失败，使用原始问题")
        return question


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
    else:
        # 如果翻译失败，返回原文
        if DEBUG:
            print(f"⚠️ 翻译失败，返回原文")
        return text


def ask_fallback(question: str, fallback_reason: str = "检索资料不足") -> dict:
    """Fallback 机制：当检索不到资料时，用模型自身知识回答

    设计目的：在 retrieval failure 场景下切换为模型知识回答，
    避免因为检索不到资料而无法回答或直接拒答。

    Args:
        question: 用户问题
        fallback_reason: 说明为什么使用 fallback

    Returns:
        dict: 包含 answer 和 fallback 标记的字典
    """
    if DEBUG:
        print(f"⚠️ 触发 fallback 机制：{fallback_reason}")

    prompt = f"""你是一个机器学习助教，检索资料中没有找到相关内容。

但你可以基于自己的知识来回答这个问题。请注意：
1. 必须全部使用中文回答
2. 开头先说明"检索资料中没有相关信息，但我可以基于已有知识解释："
3. 然后给出清晰、准确的解释
4. 如果确实不确定，也要诚实说明

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
        return {
            "answer": answer,
            "sources": [],  # 没有检索来源
            "doc_count": 0,
            "fallback_used": True
        }
    else:
        if DEBUG:
            print(f"❌ fallback 也失败了：{response.message}")
        return {
            "answer": "抱歉，暂时无法回答这个问题。",
            "sources": [],
            "doc_count": 0,
            "fallback_used": True
        }


# 定义 RAG 查询函数（带引用来源）
def ask_rag(question, k=5, return_sources=True):
    """基于向量搜索结果回答问题，并标注引用来源（带 fallback 机制）

    Args:
        question: 用户问题
        k: 检索文档数量
        return_sources: 是否返回来源列表

    Returns:
        dict: 包含 answer 和 sources 的字典
    """
    # Step 1: 重写查询以提高检索效果
    new_query = rewrite_query(question)

    # === 调试代码：显示检索结果 ===
    if DEBUG:
        print("\n=== ORIGINAL QUERY ===")
        print(question)
        print("\n=== REWRITTEN QUERY ===")
        print(new_query)

        docs = vectorstore.similarity_search(new_query, k=k)

        print("\n=== RETRIEVED DOCS ===")
        for i, doc in enumerate(docs):
            print(f"\n--- Doc {i+1} ---")
            print(doc.page_content[:300])
    else:
        docs = vectorstore.similarity_search(new_query, k=k)
    # ===========================

    # Step 1: 判断"检索是否靠谱"
    # 判断 1: 没有检索到任何文档
    if len(docs) == 0:
        return ask_fallback(question, "暂无相关检索资料")

    # 判断 2: 检索到的文档太短或质量差
    # 说明：总长度小于 100 可能意味着检索到的内容不够充分
    total_content_len = sum(len(doc.page_content) for doc in docs)
    if total_content_len < 100:
        return ask_fallback(question, "检索资料内容过少")

    # 实际检索到的文档数量
    actual_count = len(docs)

    # 构建来源列表（完整内容，用于显示）
    sources_list = []
    for i, doc in enumerate(docs):
        ref_num = i + 1
        source_info = {
            "id": ref_num,
            "content": doc.page_content.strip(),
            "preview": doc.page_content[:150].strip().replace("\n", " ")
        }
        if len(source_info["preview"]) > 150:
            source_info["preview"] += "..."
        sources_list.append(source_info)

    # 构建带引用编号的上下文（用于 prompt）
    context_parts = []
    for src in sources_list:
        context_parts.append(f"[{src['id']}] {src['content']}")
    context = "\n\n".join(context_parts)

    # 先翻译 context 为中文（方案 2：更稳定）
    context_cn = translate_to_chinese(context)

    # 优化后的 prompt - 带 fallback 机制
    prompt = f"""你是一个机器学习助教，请根据提供的资料回答问题。

规则：
1. 必须全部使用中文回答（不能出现英文句子）
2. 如果资料是英文，请翻译后再回答
3. 优先使用资料内容
4. 可以进行合理补充解释
5. 如果资料不足，可以补充自己的知识，但要说明

资料：
{context_cn}

问题：
{question}

请用中文回答："""

    # 使用 DashScope 调用通义千问模型，设置 temperature=0 保证稳定性
    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0.1,  # 低温度，更稳定
        max_tokens=500
    )

    if response.status_code == HTTPStatus.OK:
        answer = response.output.text
        result = {
            "answer": answer,
            "sources": sources_list,
            "doc_count": actual_count,
            "fallback_used": False  # 标记是否使用了 fallback
        }
        return result
    else:
        if DEBUG:
            print(f"❌ 请求失败:")
            print(f"   状态码：{response.status_code}")
            print(f"   错误代码：{response.code}")
            print(f"   错误信息：{response.message}")
        return None
