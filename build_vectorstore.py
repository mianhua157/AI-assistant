"""
build_vectorstore.py - 构建混合向量知识库

功能：同时读取 raw/ 里的 PDF 和 wiki/ 里的 markdown，构建 FAISS 向量库
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============== 1. 读取 PDF ==============

def load_raw_documents(raw_dir: str = "raw"):
    """
    读取 raw 目录下的所有 PDF 文件
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    docs = []

    # 如果目录不存在，返回空列表
    if not os.path.exists(raw_dir):
        print(f"警告：目录 {raw_dir} 不存在，跳过 PDF 加载")
        return docs

    for filename in os.listdir(raw_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(raw_dir, filename)
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            # 添加元数据标记
            for doc in pdf_docs:
                doc.metadata["source"] = path
                doc.metadata["type"] = "raw_pdf"
            docs.extend(pdf_docs)
            print(f"已加载 PDF: {filename}")

    return docs


# ============== 2. 读取 wiki markdown ==============

def load_wiki_documents(wiki_dir: str = "wiki"):
    """
    读取 wiki 目录下的所有 markdown 文件
    给每个文档添加 metadata 标签
    """
    from langchain_core.documents import Document

    docs = []

    # 如果目录不存在，返回空列表
    if not os.path.exists(wiki_dir):
        print(f"警告：目录 {wiki_dir} 不存在，跳过 wiki 加载")
        return docs

    for filename in os.listdir(wiki_dir):
        if filename.endswith(".md"):
            path = os.path.join(wiki_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            docs.append(
                Document(
                    page_content="[WIKI]\n" + text,
                    metadata={"source": path, "type": "wiki"}
                )
            )
            print(f"已加载 wiki: {filename}")

    return docs


# ============== 2.5 切分 wiki 文档 ==============

def split_wiki_documents(wiki_docs):
    """
    将 wiki 文档切分成更小的 chunk，增加 wiki 在向量库中的数量
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(wiki_docs)


# ============== 3. 合并两类文档 ==============

def load_all_documents():
    """
    加载所有文档（PDF + wiki）
    wiki 文档会被切分成更小的 chunk 以增加数量
    """
    raw_docs = load_raw_documents("raw")
    wiki_docs = load_wiki_documents("wiki")

    # 对 wiki 文档进行切分，增加 wiki 在向量库中的数量
    if wiki_docs:
        wiki_docs = split_wiki_documents(wiki_docs)
        print(f"  - Wiki 切分后：{len(wiki_docs)} 个 chunk")

    print(f"\n文档加载完成:")
    print(f"  - PDF 文档数：{len(raw_docs)}")
    print(f"  - Wiki 文档数：{len(wiki_docs)}")
    print(f"  - 总计：{len(raw_docs) + len(wiki_docs)}")

    return raw_docs + wiki_docs


# ============== 4. 切 chunk ==============

def split_documents(documents):
    """
    将文档切分为 chunk
    使用 RecursiveCharacterTextSplitter
    省 token 配置：每个 chunk 最多 500 字符
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 省 token：从 800 降到 500
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)
    print(f"\n切分完成：{len(split_docs)} 个 chunk")

    return split_docs


# ============== 5. 建立向量库 ==============

def build_vectorstore(output_dir: str = "faiss_index"):
    """
    构建 FAISS 向量库
    使用本地 sentence-transformers embedding
    """
    from langchain_community.vectorstores import FAISS
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import pickle
    import faiss

    # 加载所有文档
    print("=" * 50)
    print("开始构建向量库")
    print("=" * 50)

    documents = load_all_documents()

    if not documents:
        print("错误：没有加载到任何文档")
        return None

    # 切分文档
    split_docs = split_documents(documents)

    # 加载 embedding 模型
    print("\n加载 embedding 模型...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 生成 embeddings
    print("生成 embeddings...")
    texts = [doc.page_content for doc in split_docs]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # 构建 FAISS 索引
    print("构建 FAISS 索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # 保存索引
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    # 保存 docstore（兼容 langchain 格式）
    docstore = {}
    index_to_docstore_id = {}

    for i, doc in enumerate(split_docs):
        docstore[i] = doc
        index_to_docstore_id[i] = i

    with open(os.path.join(output_dir, "index.pkl"), "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    print(f"\n{'=' * 50}")
    print(f"向量库已保存到：{output_dir}/")
    print(f"  - 向量数量：{index.ntotal}")
    print(f"  - 维度：{dimension}")
    print(f"{'=' * 50}")

    print("当前工作目录:", os.getcwd())
    return index


# ============== 主程序 ==============

if __name__ == "__main__":
    build_vectorstore()
