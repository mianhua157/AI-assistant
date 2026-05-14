import streamlit as st

from rag import ask_rag, load_vectorstore, rewrite_query_bilingual


st.set_page_config(
    page_title="机器学习 RAG 问答系统",
    page_icon="📚",
    layout="wide",
)

st.title("📚 机器学习 RAG 问答系统")
st.caption("基于 PDF + Wiki 混合知识库的课程助教")

with st.sidebar:
    st.header("使用说明")
    st.markdown(
        """
        - 支持中英文提问
        - 英文短问题会自动补充中文检索
        - 示例：
          - What is classification?
          - 什么是回归？
          - Compare classification and regression
        """
    )


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


question = st.text_input("请输入你的问题：")

if question:
    rewritten = rewrite_query_bilingual(question)
    if rewritten != question:
        st.info(f"检索改写：`{rewritten}`")

    try:
        vectorstore = get_vectorstore()
    except Exception as exc:
        st.error(f"向量库加载失败：{exc}")
        st.info("请先运行 `python build_vectorstore.py` 构建本地索引。")
        st.stop()

    with st.spinner("正在检索资料并生成回答..."):
        result = ask_rag(question, vectorstore)

    if result.get("fallback_used"):
        st.warning("未检索到足够相关的课程资料，以下回答部分依赖模型已有知识。")

    st.subheader("回答")
    st.write(result["answer"])

    if result.get("sources"):
        with st.expander("参考资料"):
            for source in result["sources"]:
                st.markdown(f"**来源 {source['id']}**")
                st.write("类型：", source.get("type", "raw"))
                st.write("文件：", source.get("source", "unknown"))
                st.write(source.get("content", "")[:500])
                st.markdown("---")
