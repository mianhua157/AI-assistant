"""
rag.py - RAG 核心逻辑（检索 + 生成）

支持从 PDF + Wiki 混合向量库中检索，并显示来源类型
"""

import os
from http import HTTPStatus
import dashscope
import faiss
import pickle
from pathlib import Path
import numpy as np
from typing import List

# API 配置 - 从环境变量读取
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 调试模式开关
DEBUG = False


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
    """简单的 FAISS 检索器，支持 PDF + Wiki 混合检索"""

    def __init__(self, index_path="faiss_index"):
        # 加载 FAISS 索引
        index_file = Path(index_path) / "index.faiss"
        pkl_file = Path(index_path) / "index.pkl"

        # 加载 FAISS 索引文件
        self.index = faiss.read_index(str(index_file))

        # 加载 docstore 和 index_to_docstore_id
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                self.docstore, self.index_to_docstore_id = data
            else:
                self.docstore = {}
                self.index_to_docstore_id = {}

        # 使用本地 embedding 模型
        self.embeddings = LocalEmbeddings()

    def similarity_search(self, query, k=5) -> List:
        """执行相似度搜索，返回带 metadata 的文档"""
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
            if idx < 0:
                continue
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id and doc_id in self.docstore:
                docs.append(self.docstore[doc_id])

        return docs


# ============== 加载向量库 ==============

def load_vectorstore():
    """加载混合向量库（PDF + Wiki）"""
    try:
        return SimpleFAISSRetriever("faiss_index")
    except Exception as e:
        print(f"⚠️ 加载向量库失败：{e}")
        raise


vectorstore = load_vectorstore()


# ============== 格式化上下文 ==============

def format_context(docs) -> str:
    """
    格式化检索结果，带上来源类型
    省 token 配置：每个 chunk 最多 500 字符
    用于构建 prompt 的 context 部分
    """
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "raw")
        # 省 token：限制每个 chunk 最多 500 字符
        content = doc.page_content[:500]
        context += f"[{i+1}] Source Type: {doc_type}, Source: {source}\n{content}\n\n"
    return context


# ============== 检索结果处理 ==============

def format_sources(docs) -> List[dict]:
    """
    格式化检索结果为 sources 列表，用于前端显示
    包含来源类型信息
    """
    sources_list = []
    for i, doc in enumerate(docs):
        ref_num = i + 1
        source_info = {
            "id": ref_num,
            "content": doc.page_content.strip(),
            "preview": doc.page_content[:150].strip().replace("\n", " "),
            "source": doc.metadata.get("source", "unknown"),
            "type": doc.metadata.get("type", "raw")
        }
        if len(source_info["preview"]) > 150:
            source_info["preview"] += "..."
        sources_list.append(source_info)
    return sources_list


# ============== 翻译函数 ==============

def translate_to_chinese(text: str) -> str:
    """
    将英文资料翻译成中文
    省 token 配置：暂时关闭翻译，直接用原文
    """
    # 省 token：跳过翻译，直接返回原文
    return text


# ============== Query Rewrite（统一检索语言） ==============

def rewrite_query_to_chinese(query: str) -> str:
    """
    将英文 query 重写为中文（统一检索语言）
    只有当 query 包含英文时才调用
    """
    # 检查是否包含英文字母
    import re
    if not re.search(r'[a-zA-Z]', query):
        # 纯中文，不需要重写
        return query

    # 判断是否是短问题（单词数 <= 5）
    words = query.split()
    if len(words) > 5:
        return query  # 长句子，不重写

    prompt = f"""把下面的机器学习相关问题翻译成中文，只输出翻译结果：

问题：{query}

翻译：
"""

    try:
        response = dashscope.Generation.call(
            model="qwen-plus",
            prompt=prompt,
            api_key=DASHSCOPE_API_KEY,
            temperature=0
        )

        if response.status_code == HTTPStatus.OK:
            translated = response.output.text.strip()
            # 清理可能的前缀
            if translated.startswith("翻译："):
                translated = translated[3:].strip()
            return translated
        else:
            return query
    except Exception:
        return query  # 失败时返回原文


def rewrite_query_bilingual(query: str) -> str:
    """
    中英一起喂给检索（高级版）
    例如：What is classification? → What is classification? 什么是分类？

    好处：
    - 不丢英文信息
    - 中文也能命中 wiki
    """
    # 检查是否包含英文字母
    import re
    if not re.search(r'[a-zA-Z]', query):
        # 纯中文，直接返回
        return query

    # 判断是否是短问题（单词数 <= 5）
    words = query.split()
    if len(words) > 5:
        # 长句子，只返回原文
        return query

    prompt = f"""把下面的机器学习相关问题翻译成中文，只输出翻译结果：

问题：{query}

翻译：
"""

    try:
        response = dashscope.Generation.call(
            model="qwen-plus",
            prompt=prompt,
            api_key=DASHSCOPE_API_KEY,
            temperature=0
        )

        if response.status_code == HTTPStatus.OK:
            translated = response.output.text.strip()
            # 清理可能的前缀
            if translated.startswith("翻译："):
                translated = translated[3:].strip()
            # 中英一起喂给检索
            return f"{query} {translated}"
        else:
            return query
    except Exception:
        return query  # 失败时返回原文


# ============== Fallback 机制 ==============

def ask_fallback(question: str, fallback_reason: str = "检索资料不足") -> dict:
    """
    Fallback 机制：当检索不到资料时，用模型自身知识回答
    弱容错：如果资料不完全匹配，可以基于已有内容做合理解释
    """
    if DEBUG:
        print(f"⚠️ 触发 fallback 机制：{fallback_reason}")

    prompt = f"""你是一个机器学习助教。资料不足时，可以基于你的知识进行合理补充。

规则：
1. 不要编造不存在的概念
2. 回答要清晰、结构化
3. 对于定义类问题，尽量简洁
4. 对于"what is / 什么是"类问题：只给核心定义，不要展开推导

问题：
{question}

回答：
"""

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
            "sources": [],
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


# ============== RAG 核心函数 ==============

def ask_rag(question, vectorstore):
    """
    基于向量搜索结果回答问题（从 PDF + Wiki 混合库检索）

    Args:
        question: 用户问题
        vectorstore: 向量库

    Returns:
        tuple: (answer, docs) 回答和检索到的文档
    """
    # 中英一起喂给检索（高级版）
    rewritten_query = rewrite_query_bilingual(question)
    if rewritten_query != question:
        print(f"\nQuery 重写：{question} → {rewritten_query}")

    # 智能 k 值：定义类问题 k=3，比较类 k=8，其他 k=6
    q = rewritten_query.lower()
    is_definition = q.startswith("what is") or q.startswith("什么是") or q.startswith("定义")
    is_compare = "compare" in q or "对比" in q or "比较" in q or "vs" in q or "difference" in q

    if is_definition:
        candidates = vectorstore.similarity_search(rewritten_query, k=3)
    elif is_compare:
        candidates = vectorstore.similarity_search(rewritten_query, k=8)
    else:
        candidates = vectorstore.similarity_search(rewritten_query, k=6)

    # 检查 1：打印候选池
    print("\n=== 候选池 ===")
    for i, d in enumerate(candidates):
        print(f"{i+1} {d.metadata.get('type')} {d.metadata.get('source')}")

    # 优先选 wiki：按类型分离
    wiki_docs = [d for d in candidates if d.metadata.get("type") == "wiki"]
    raw_docs = [d for d in candidates if d.metadata.get("type") == "raw_pdf"]

    # 强制"只取最相关 wiki"：按 score 排序
    wiki_docs = sorted(wiki_docs, key=lambda x: x.metadata.get("score", 0), reverse=True)

    docs = []

    # 比较类问题：拿 2 个 wiki；否则只拿 1 个
    if is_compare:
        docs.extend(wiki_docs[:2])
    else:
        if wiki_docs:
            docs.append(wiki_docs[0])

    # 非比较类问题：1 个 raw
    if not is_compare and raw_docs:
        docs.append(raw_docs[0])

    # 剩下的位置补 raw（不再补 wiki）
    remaining = raw_docs[1:]
    docs.extend(remaining[:2])

    # 检查 2：打印最终 docs
    print("\n=== 最终送入模型的 docs ===")
    for i, d in enumerate(docs):
        print(f"{i+1} {d.metadata.get('type')} {d.metadata.get('source')}")

    # 格式化上下文（省 token：每个 doc 最多 600 字符）
    context = "\n\n".join([doc.page_content[:600] for doc in docs])

    # 判断检索质量
    if len(docs) == 0:
        return ask_fallback(question, "暂无相关检索资料"), []

    total_content_len = sum(len(doc.page_content) for doc in docs)
    if total_content_len < 100:
        return ask_fallback(question, "检索资料内容过少"), []

    # 省 token：跳过翻译，直接用 context

    # 构建 prompt（优化版）
    if is_compare:
        prompt = f"""你是一个机器学习助教，请基于检索到的资料回答问题。

规则：
1. 优先使用提供的资料
2. 回答要清晰、结构化
3. 比较类问题请列出两者的异同点（可以用表格或分点）
4. 如果资料不完整，可以基于你的知识进行合理补充

资料：
{context}

问题：
{question}

回答：
"""
    else:
        prompt = f"""你是一个机器学习助教，请基于检索到的资料回答问题。

规则：
1. 优先使用提供的资料
2. 如果资料不完整，可以基于你的知识进行合理补充
3. 不要编造不存在的概念
4. 回答要清晰、结构化
5. 对于定义类问题，尽量简洁

对于"what is / 什么是"类问题：
- 只给核心定义
- 不要展开推导或复杂公式

资料：
{context}

问题：
{question}

回答：
"""

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.text, docs
    else:
        return f"请求失败：{response.code} - {response.message}", docs


# ============== 旧版 ask_rag 兼容（可选删除） ==============

def ask_rag_old(question, k=5):
    """旧版 ask_rag，保留用于兼容（省 token 简化版）"""
    # 检索（减少 k 值）
    docs = vectorstore.similarity_search(question, k=k)

    if len(docs) == 0:
        return ask_fallback(question, "暂无相关检索资料")

    total_content_len = sum(len(doc.page_content) for doc in docs)
    if total_content_len < 100:
        return ask_fallback(question, "检索资料内容过少")

    # 格式化来源（带类型）
    sources_list = format_sources(docs)

    # 构建上下文（省 token：限制每个 chunk 500 字符）
    context_parts = []
    for src in sources_list:
        content = src['content'][:500]  # 限制长度
        context_parts.append(f"[{src['id']}] {content}")
    context = "\n\n".join(context_parts)
    # 省 token：跳过翻译

    prompt = f"""你是机器学习助教。优先根据资料回答。
如果资料部分相关，可以合理总结。用中文回答。

资料：
{context}

问题：
{question}

回答：
"""

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
            "sources": sources_list,
            "doc_count": len(docs),
            "fallback_used": False
        }
    else:
        if DEBUG:
            print(f"❌ 请求失败:")
            print(f"   状态码：{response.status_code}")
            print(f"   错误信息：{response.message}")
        return None
