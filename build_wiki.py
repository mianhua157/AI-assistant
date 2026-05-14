"""
build_wiki.py - 从 PDF 生成机器学习概念知识库

功能：读取 PDF 内容，让模型生成若干个 markdown 概念页，保存到 wiki/
"""

import os
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv


load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


def load_pdf_documents(raw_dir: str = "raw"):
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    if not os.path.exists(raw_dir):
        print(f"警告：目录 {raw_dir} 不存在")
        return docs

    for file in os.listdir(raw_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(raw_dir, file))
            docs.extend(loader.load())
            print(f"已加载 PDF: {file}")

    return docs


def load_pdf_text(pdf_path: str) -> str:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return "\n\n".join(doc.page_content for doc in documents)


def generate_wiki_page(full_text: str, topic: str) -> str:
    prompt = f"""你是一个机器学习课程知识整理助手。

请根据以下课程资料，围绕主题"{topic}"生成一份 markdown 格式的知识页。

要求：
1. 只使用资料中的内容，必要时做简洁整理
2. 输出必须是 markdown
3. 必须按以下结构输出：
   - 标题
   - 核心定义
   - 训练/预测流程
   - 关键要点
   - 与其他概念的关系
   - 资料中未充分覆盖的部分（如果有，必须放在最后）
4. "资料中未充分覆盖的部分"不能放在前面
5. 全部用中文输出，专业术语可保留英文

课程资料：
{full_text}

请直接输出 markdown：
"""

    response = dashscope.Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=DASHSCOPE_API_KEY,
        temperature=0,
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.text
    raise RuntimeError(f"生成失败：{response.code} - {response.message}")


def save_wiki_page(topic: str, content: str, output_dir: str = "wiki"):
    os.makedirs(output_dir, exist_ok=True)
    filename = topic.lower().replace(" ", "_") + ".md"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"已保存：{path}")


def main(topics_override=None):
    import sys

    print("正在读取 raw/ 目录下的所有 PDF...")
    docs = load_pdf_documents("raw")

    if not docs:
        print("Error: 没有找到任何 PDF 文件")
        return

    full_text = "\n\n".join(doc.page_content for doc in docs)
    print(f"PDF 读取完成，总长度：{len(full_text)} 字符")

    key_topics = [
        "regression",
        "classification vs regression",
        "supervised learning",
    ]

    if topics_override:
        topics = topics_override
        print(f"\n>>> 使用指定的 topic 列表（共 {len(topics)} 个）")
    elif len(sys.argv) > 1:
        topics = sys.argv[1:]
        print(f"\n>>> 使用命令行指定的 topic 列表（共 {len(topics)} 个）")
    else:
        topics = key_topics
        print(f"\n>>> 使用关键 topic 列表（共 {len(topics)} 个）- 省 token 模式")

    print(f"开始生成 {len(topics)} 个概念页面...")
    print("=" * 50)

    for topic in topics:
        print(f"\n正在生成：{topic}")
        try:
            content = generate_wiki_page(full_text, topic)
            save_wiki_page(topic, content)
        except Exception as exc:
            print(f"Error: 生成失败 [{topic}]: {exc}")

    print("\n" + "=" * 50)
    print("生成完成！")
    print("输出目录：wiki/")


if __name__ == "__main__":
    main()
