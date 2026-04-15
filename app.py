import streamlit as st
import os
import re

# 检查环境变量
if not os.getenv("DASHSCOPE_API_KEY"):
    st.error("❌ 请设置环境变量 DASHSCOPE_API_KEY")
    st.info("💡 设置方法：\n\n- Windows (PowerShell): `$env:DASHSCOPE_API_KEY=\"sk-your-key\"`\n- Linux/Mac: `export DASHSCOPE_API_KEY=\"sk-your-key\"`\n\n或者复制 `.env.example` 为 `.env` 并填入你的 API Key。")
    st.stop()

from rag import ask_rag

# 页面配置
st.set_page_config(
    page_title="机器学习 RAG 问答系统",
    page_icon="📚",
    layout="wide"
)

# 标题和说明
st.title("📚 机器学习 RAG 问答系统")
st.caption("一个基于课程讲义资料的机器学习 RAG 问答系统，支持英文资料检索与中文回答。")
st.info("💡 Loading embedding model, first run may take a while...")

# 侧边栏 - 示例问题
st.sidebar.markdown("**示例问题：**")
st.sidebar.markdown("- What is regression?")
st.sidebar.markdown("- What is classification?")
st.sidebar.markdown("- What is overfitting?")
st.sidebar.markdown("- What is k-nearest neighbor?")
st.sidebar.markdown("- What is decision tree?")

# 缓存字典 - 对同一个问题做缓存
if "cache" not in st.session_state:
    st.session_state.cache = {}


def extract_keywords(question: str) -> list:
    """从问题中提取关键词用于高亮"""
    # 简单实现：提取英文单词（长度>3 的）
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
    # 去除常见停用词
    stopwords = {'what', 'this', 'that', 'with', 'from', 'have', 'been', 'would', 'could'}
    return [w for w in words if w not in stopwords]


def highlight_content(content: str, keywords: list) -> str:
    """高亮内容中的关键词"""
    highlighted = content
    for keyword in keywords:
        # 不区分大小写替换，用**加粗标记
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(f'**{keyword}**', highlighted)
    return highlighted


# 主界面 - 输入框
question = st.text_input("请输入你的问题：", placeholder="e.g., What is classification?")

# 处理回答
if question:
    # 检查缓存
    if question in st.session_state.cache:
        result = st.session_state.cache[question]
        st.info("⚡ 从缓存加载")
    else:
        with st.spinner("正在检索资料并生成回答..."):
            result = ask_rag(question)

        if result:
            # 存入缓存
            st.session_state.cache[question] = result

    if result:
        # 检查是否使用了 fallback 机制
        if result.get("fallback_used", False):
            st.warning("⚠️ 检索资料中没有相关内容，以下是基于模型已有知识的回答")

        # 显示回答
        st.subheader("✅ 回答")
        st.write(result["answer"])

        # 查看参考资料（可折叠）
        if result.get("sources"):
            # 提取关键词用于高亮
            keywords = extract_keywords(question)

            with st.expander("📚 查看参考资料"):
                for src in result["sources"]:
                    st.write(f"**来源 {src['id']}:**")
                    # 高亮关键词后显示
                    highlighted = highlight_content(src["content"], keywords)
                    st.markdown(highlighted)
        elif result.get("fallback_used", False):
            st.info("💡 本次回答没有检索到相关资料，完全基于模型知识生成")
