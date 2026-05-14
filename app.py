import streamlit as st

from rag import ask_rag, load_vectorstore, rewrite_query_bilingual


st.set_page_config(
    page_title="Machine Learning RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Machine Learning RAG Assistant")
st.caption("Intent-aware course assistant with retrieval planning and coverage checks")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        - Supports Chinese and English questions
        - Short English questions may be expanded with Chinese search terms
        - The system now plans retrieval before answering

        Example questions:
        - What is classification?
        - classification 和 regression 有什么区别？
        - 帮我总结这一章
        - 帮我出 3 道复习题
        - 资料里有没有覆盖 overfitting？
        """
    )


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


question = st.text_input("Ask a course question:")

if question:
    rewritten = rewrite_query_bilingual(question)
    if rewritten != question:
        st.info(f"Search rewrite: `{rewritten}`")

    try:
        vectorstore = get_vectorstore()
    except Exception as exc:
        st.error(f"Vector index failed to load: {exc}")
        st.info("Please run `python build_vectorstore.py` first.")
        st.stop()

    with st.spinner("Planning retrieval, checking coverage, and generating an answer..."):
        result = ask_rag(question, vectorstore)

    coverage = result.get("coverage", {})
    plan = result.get("plan", {})

    meta_left, meta_mid, meta_right = st.columns(3)
    with meta_left:
        st.metric("Detected Intent", result.get("intent", "unknown"))
    with meta_mid:
        st.metric("Coverage", coverage.get("status", "unknown"))
    with meta_right:
        st.metric("Retrieved Sources", str(result.get("doc_count", 0)))

    if result.get("intent_reason"):
        st.caption(f"Intent reason: {result['intent_reason']}")

    if coverage.get("reason"):
        if coverage.get("can_answer", False):
            st.success(f"Coverage check: {coverage['reason']}")
        else:
            st.warning(f"Coverage check: {coverage['reason']}")

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("queries"):
        with st.expander("Queries used for retrieval"):
            for query in result["queries"]:
                st.write(f"- {query}")

    if plan:
        with st.expander("Retrieval plan"):
            st.json(plan)

    if result.get("execution_trace"):
        with st.expander("Retrieval execution trace"):
            st.json(result["execution_trace"])

    if result.get("sources"):
        with st.expander("Retrieved sources"):
            for source in result["sources"]:
                st.markdown(f"**Source {source['id']}**")
                st.write("Type:", source.get("type", "raw"))
                st.write("File:", source.get("source", "unknown"))
                if source.get("page") is not None:
                    st.write("Page:", source.get("page"))
                if source.get("score") is not None:
                    st.write("Score:", source.get("score"))
                if source.get("query"):
                    st.write("Matched by query:", source.get("query"))
                st.write(source.get("content", "")[:600])
                st.markdown("---")
