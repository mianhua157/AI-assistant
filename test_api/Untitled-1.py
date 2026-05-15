"""
PDF 智能问答系统 - 阿里云版本
"""

import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ========== 配置部分 ==========
# 1. 设置你的阿里云 API 密钥
ALIYUN_API_KEY = "sk-sp-b08258ea5af34c5198e1434b0dc30fd9"  # ⚠️ 替换为你的真实密钥

# 2. 设置 PDF 文件路径
PDF_FILE_PATH = "classification.pdf"  # ⚠️ 确保有这个文件

# 3. 检查文件是否存在
if not os.path.exists(PDF_FILE_PATH):
    print(f"❌ 错误：找不到文件 {PDF_FILE_PATH}")
    print(f"请确保 PDF 文件位于: {os.path.abspath('.')}")
    exit()

# ========== 初始化 ==========
def main():
    print("🚀 开始处理 PDF 文档...")
    
    # 步骤 1：加载 PDF
    print("1. 加载 PDF 文档...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    print(f"   ✅ 成功加载 {len(documents)} 页文档")
    
    # 步骤 2：分割文本
    print("2. 分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每块 500 字符
        chunk_overlap=100,   # 重叠 100 字符
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"   ✅ 分割为 {len(split_docs)} 个文本块")
    
    # 步骤 3：创建阿里云 embeddings
    print("3. 初始化阿里云 embeddings...")
    try:
        # 设置阿里云 API 密钥
        dashscope.api_key = ALIYUN_API_KEY
        
        # 创建 embeddings
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=ALIYUN_API_KEY
        )
        print("   ✅ embeddings 初始化成功")
    except Exception as e:
        print(f"   ❌ embeddings 初始化失败: {e}")
        return
    
    # 步骤 4：创建向量存储
    print("4. 创建向量存储...")
    try:
        vectorstore = FAISS.from_documents(
            split_docs,
            embeddings
        )
        print("   ✅ 向量存储创建成功")
    except Exception as e:
        print(f"   ❌ 向量存储创建失败: {e}")
        return
    
    # 步骤 5：保存向量库
    print("5. 保存向量库...")
    try:
        vectorstore.save_local("faiss_index")
        print("   ✅ 向量库已保存到 'faiss_index' 文件夹")
    except Exception as e:
        print(f"   ❌ 保存失败: {e}")
    
    # 步骤 6：测试查询
    print("6. 测试查询功能...")
    query = "这个文档主要讲什么？"
    print(f"   查询: '{query}'")
    
    try:
        docs = vectorstore.similarity_search(query, k=2)
        print(f"   ✅ 检索到 {len(docs)} 个相关文档")
        for i, doc in enumerate(docs):
            print(f"\n   --- 文档 {i+1} ---")
            print(f"   来源: 第 {doc.metadata.get('page', '未知')} 页")
            print(f"   内容: {doc.page_content[:150]}...")
    except Exception as e:
        print(f"   ❌ 查询失败: {e}")
    
    print("\n🎉 处理完成！")

if __name__ == "__main__":
    main()