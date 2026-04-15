# 机器学习 RAG 问答系统

一个基于课程讲义资料的机器学习 RAG（检索增强生成）问答系统，支持英文资料检索与中文回答。

## 项目结构

```
pdf_ai_project/
├── app.py                 # Streamlit 应用主入口
├── rag.py                 # RAG 核心逻辑（检索 + 生成）
├── rebuild_faiss.py       # FAISS 索引重建脚本
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量示例
├── .gitignore
└── faiss_index/           # 向量数据库（FAISS 索引）
    ├── index.faiss
    └── index.pkl
```

## 快速开始

### 前置要求

- Python 3.9+
- DashScope API Key（通义千问）

### 1. API Key 设置

本项目使用 DashScope 作为大语言模型后端，需要先获取 API Key：

**获取地址：** https://dashscope.console.aliyun.com/apiKey

#### Windows (PowerShell)

```powershell
# 临时设置（当前终端会话）
$env:DASHSCOPE_API_KEY="sk-your-key"
```

#### Windows (CMD)

```cmd
set DASHSCOPE_API_KEY=sk-your-key
```

#### Linux / macOS

```bash
export DASHSCOPE_API_KEY="sk-your-key"
```

#### 永久设置（推荐）

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入 `DASHSCOPE_API_KEY`。

### 2. 环境准备

#### 克隆仓库

```bash
git clone <repository-url>
cd pdf_ai_project
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行应用

```bash
streamlit run app.py
```

### 4. 重建 FAISS 索引（可选）

如果你需要更新 PDF 数据或更改 embedding 模型：

```bash
python rebuild_faiss.py
```

---

## Bug 修复总结

### 问题

运行时出现 `AssertionError: assert d == self.d` 错误。

### 原因

1. **Embedding 维度不匹配**：原 FAISS 索引为 1536 维，代码使用 384 维模型
2. **LangChain FAISS 兼容性 Bug**：新版本 `langchain-community` 存在内部断言错误

### 解决方案

1. 使用本地 `sentence-transformers/all-MiniLM-L6-v2` 模型（384 维）重新构建 FAISS 索引
2. 重写检索器，绕过 LangChain 兼容性问题

### 修改文件

| 文件 | 修改说明 |
|------|----------|
| `rag.py` | 重写 embedding 和检索逻辑 |
| `rebuild_faiss.py` | 新建 FAISS 索引重建脚本 |
| `requirements.txt` | 更新依赖包 |
| `faiss_index/` | 重建为 384 维本地模型索引 |

---

## 使用指南

### 示例问题

系统内置了以下示例问题供参考：

- What is regression?
- What is classification?
- What is overfitting?
- What is k-nearest neighbor?
- What is decision tree?

### 功能特点

- 支持英文问题输入
- 中文回答输出
- 检索结果引用来源标注
- 问题关键词高亮显示
- 回答缓存机制

---

## 依赖

- **streamlit**: Web 应用框架
- **dashscope**: 通义千问 API
- **sentence-transformers**: 本地 Embedding 模型（384 维）
- **PyPDF2**: PDF 文本提取
- **faiss-cpu**: 向量相似度搜索

---

## 未来改进

- [ ] 支持更多嵌入模型选项
- [ ] 增加 Web UI 配置界面
- [ ] 支持多轮对话
- [ ] 添加评估指标

---

## 联系方式

如有问题，请提交 Issue 或联系开发者。
