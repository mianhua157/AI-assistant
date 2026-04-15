"""
重新构建 FAISS 索引，使用本地 embedding 模型
解决 API key 缺失和维度不匹配问题
"""
import os
import pickle
from pathlib import Path
import faiss
import numpy as np

# 先安装必要的包
print("正在检查依赖...")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("正在安装 sentence-transformers...")
    os.system("pip install sentence-transformers -q")
    from sentence_transformers import SentenceTransformer

try:
    import PyPDF2
except ImportError:
    print("正在安装 PyPDF2...")
    os.system("pip install PyPDF2 -q")
    import PyPDF2

print("依赖检查完成！")

# 配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 维
FAISS_INDEX_PATH = "faiss_index"
PDF_FILES = [
    "COINS_CAMERA_READY_IEEE_APPROVED.pdf",
    "NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf",
    "P1_8_Zayoud+et+al._Impact+of+ChatGPT.pdf",
    "classification.pdf"
]

def extract_text_from_pdf(pdf_path):
    """从 PDF 提取文本"""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"读取 {pdf_path} 失败：{e}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """将文本分块"""
    chunks = []
    # 按段落分割
    paragraphs = text.split('\n\n')

    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 保留重叠部分
            if len(para) > chunk_size:
                # 长段落进一步分割
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i+chunk_size].strip())
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def main():
    print("=" * 50)
    print("重新构建 FAISS 索引")
    print("=" * 50)

    # 加载 embedding 模型
    print(f"\n正在加载 embedding 模型：{EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"模型维度：{model.get_sentence_embedding_dimension()}")

    # 提取所有 PDF 文本
    all_chunks = []
    for pdf in PDF_FILES:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"⚠️ 文件不存在：{pdf}")
            continue

        print(f"\n处理：{pdf}")
        text = extract_text_from_pdf(str(pdf_path))
        if text:
            chunks = chunk_text(text)
            print(f"  提取了 {len(chunks)} 个文本块")
            all_chunks.extend(chunks)
        else:
            print(f"  ⚠️ 没有提取到文本")

    if not all_chunks:
        print("\n❌ 没有提取到任何文本，无法构建索引")
        return

    print(f"\n总文本块数：{len(all_chunks)}")

    # 生成 embeddings
    print("\n正在生成 embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)
    print(f"Embeddings 形状：{embeddings.shape}")

    # 构建 FAISS 索引
    print("\n正在构建 FAISS 索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # 保存索引
    print(f"\n正在保存到：{FAISS_INDEX_PATH}")
    index_dir = Path(FAISS_INDEX_PATH)
    index_dir.mkdir(exist_ok=True)

    # 保存 FAISS 索引
    faiss.write_index(index, str(index_dir / "index.faiss"))

    # 保存 docstore（兼容 langchain 格式）
    from langchain_core.documents import Document
    docstore = {}
    index_to_docstore_id = {}

    for i, chunk in enumerate(all_chunks):
        doc = Document(page_content=chunk, metadata={"source": "local_pdf"})
        docstore[i] = doc
        index_to_docstore_id[i] = i

    with open(index_dir / "index.pkl", "wb") as f:
        pickle.dump((docstore, index_to_docstore_id), f)

    print(f"\n✅ 完成！")
    print(f"   - 向量数量：{index.ntotal}")
    print(f"   - 维度：{dimension}")
    print(f"   - 文件位置：{index_dir.absolute()}")

    # 测试检索
    print("\n测试检索...")
    test_query = "What is regression?"
    query_embedding = model.encode([test_query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_embedding, k=2)

    print(f"\n查询：{test_query}")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"\n[{i+1}] 距离：{dist:.4f}")
        print(f"    内容：{all_chunks[idx][:200]}...")

if __name__ == "__main__":
    main()
