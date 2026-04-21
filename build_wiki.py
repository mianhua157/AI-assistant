"""
build_wiki.py - 从 PDF 生成机器学习概念知识库

功能：读取 PDF 内容，让模型生成若干个 markdown 概念页，保存到 wiki/
"""

import os
from http import HTTPStatus
import dashscope
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# API Key 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# ============== 1. 读取 PDF 的函数 ==============

def load_pdf_documents(raw_dir: str = "raw"):
    """
    读取 raw 目录下的所有 PDF 文件
    """
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
    """
    读取整份 PDF，拼成一个长字符串，后面交给 LLM 做整理
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = "\n\n".join([doc.page_content for doc in documents])
    return full_text


# ============== 2. 调用模型生成 wiki 页的函数 ==============

def generate_wiki_page(full_text: str, topic: str) -> str:
    """
    基于 PDF 资料，为指定主题生成一页 markdown 格式的知识页

    Args:
        full_text: PDF 完整文本内容
        topic: 页面主题，如 "classification"

    Returns:
        markdown 格式的知识页内容
    """
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
        temperature=0  # 为了稳定
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.text
    else:
        raise RuntimeError(f"生成失败：{response.code} - {response.message}")


# ============== 3. 保存 markdown 文件的函数 ==============

def save_wiki_page(topic: str, content: str, output_dir: str = "wiki"):
    """
    将生成的 wiki 页面保存为 markdown 文件

    Args:
        topic: 页面主题
        content: markdown 内容
        output_dir: 输出目录，默认为 "wiki"
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = topic.lower().replace(" ", "_") + ".md"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"已保存：{path}")


# ============== 5. 主程序 ==============

def main(topics_override=None):
    """
    Args:
        topics_override: 可选的 topic 列表，如果提供则只生成指定的 topic
                       可以是命令行参数或直接在代码中指定
    """
    import sys

    # 从 raw/ 目录读取所有 PDF
    print("正在读取 raw/ 目录下的所有 PDF...")
    docs = load_pdf_documents("raw")

    if not docs:
        print("Error: 没有找到任何 PDF 文件")
        return

    # 合并所有 PDF 内容
    full_text = "\n\n".join([doc.page_content for doc in docs])
    print(f"PDF 读取完成，总长度：{len(full_text)} 字符")

    # 关键 topic 列表（阶段 2：只补关键 wiki，小规模花钱）
    # 只需要生成这 3 个就够了，别多做
    key_topics = [
        "regression",
        "classification vs regression",
        "supervised learning"
    ]

    # 如果提供了指定的 topics，则只生成这些；否则默认生成 3 个关键 topic
    if topics_override:
        topics = topics_override
        print(f"\n>>> 使用指定的 topic 列表（共 {len(topics)} 个）")
    elif len(sys.argv) > 1:
        # 支持命令行参数：python build_wiki.py "topic1" "topic2" ...
        topics = sys.argv[1:]
        print(f"\n>>> 使用命令行指定的 topic 列表（共 {len(topics)} 个）")
    else:
        # 默认只生成 3 个关键 topic（省 token 模式）
        topics = key_topics
        print(f"\n>>> 使用关键 topic 列表（共 {len(topics)} 个）- 省 token 模式")

    print(f"开始生成 {len(topics)} 个概念页面...")
    print("=" * 50)

    for topic in topics:
        print(f"\n正在生成：{topic}")
        try:
            content = generate_wiki_page(full_text, topic)
            save_wiki_page(topic, content)
        except Exception as e:
            print(f"Error: 生成失败 [{topic}]: {e}")

    print("\n" + "=" * 50)
    print("生成完成！")
    print(f"输出目录：wiki/")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
