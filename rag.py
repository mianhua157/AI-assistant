import os
from http import HTTPStatus
import dashscope
import faiss
import pickle
from pathlib import Path
import numpy as np
from typing import List

# API 配置 - 从环境变量读取（避免硬编码泄露）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 调试模式开关
DEBUG = False  # 设为 True 时打印详细调试信息


class LocalEmbeddings:
    """使用本地 SentenceTransformer 模型生成 embedding（384 维）"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (may download on first run)...")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"加载本地 embedding 模型失败：{e}")

    def embed_query(self, query: str) -> List[float]:
        """生成 query 的 embedding"""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


class SimpleFAISSRetriever:
    """简单的 FAISS 检索器，绕过 langchain 的兼容性问题"""

    def __init__(self, index_path="faiss_index"):
        # 加载 FAISS 索引
        index_file = Path(index_path) / "index.faiss"
        pkl_file = Path(index_path) / "index.pkl"

        # 加载 FAISS 索引文件
        self.index = faiss.read_index(str(index_file))

        # 加载 docstore 和 index_to_docstore_id
        # pkl 文件结构：(docstore dict, index_to_docstore_id dict)
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                self.docstore, self.index_to_docstore_id = data
            elif isinstance(data, dict):
                # 兼容旧格式
                self.docstore = data.get("_dict", {})
                self.index_to_docstore_id = data.get("index_to_docstore_id", {})
            else:
                self.docstore = {}
                self.index_to_docstore_id = {}

        # 使用本地 embedding 模型（384 维，和新索引一致）
        self.embeddings = LocalEmbeddings()

    def similarity_search(self, query, k=5):
        """执行相似度搜索"""
        # 生成 query 的 embedding
        query_embedding = self.embeddings.embed_query(query)

        # 转换为 numpy array 并添加 batch 维度
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # 搜索最近的 k 个向量
        distances, indices = self.index.search(
            query_embedding_np,
            k=min(k, len(self.index_to_docstore_id))
        )

        # 获取文档内容
        docs = []
        for idx in indices[0]:
            if idx < 0:  # FAISS 返回 -1 表示没有找到足够的结果
                continue
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id and doc_id in self.docstore:
                docs.append(self.docstore[doc_id])

        return docs


# 加载已保存的 FAISS 向量存储
# 注意：向量库已离线构建，此处仅演示加载流程
# 实际使用的 embedding 模型与 FAISS 索引创建时保持一致
def load_vectorstore():
    """加载向量存储，带错误处理"""
    try:
        # 优先尝试使用简单检索器（避免 langchain 兼容性问题）
        return SimpleFAISSRetriever("faiss_index")
    except Exception as e:
        print(f"⚠️ 使用 SimpleFAISSRetriever 失败：{e}，尝试重新创建索引...")
        raise

vectorstore = load_vectorstore()


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
    # 修复：使用 search 方法避免 filter 兼容性问题
    if DEBUG:
        print("\n=== ORIGINAL QUERY ===")
        print(question)
        print("\n=== REWRITTEN QUERY ===")
        print(new_query)

        # 使用 vectorstore 的 search 方法，type="similarity" 避免 filter 问题
        docs = vectorstore.similarity_search(new_query, k=k)

        print("\n=== RETRIEVED DOCS ===")
        for i, doc in enumerate(docs):
            print(f"\n--- Doc {i+1} ---")
            print(doc.page_content[:300])
    else:
        # 使用 vectorstore 的 search 方法，type="similarity" 避免 filter 问题
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
