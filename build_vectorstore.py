"""
Build a hybrid FAISS vector store from raw PDFs and wiki markdown files.

The metadata is intentionally planner-friendly:
- `type`: original source type (`raw_pdf` or `wiki`)
- `chunk_type`: normalized source family (`raw` or `wiki`)
- `source`: file path
- `source_name`: file stem for debugging
- `chapter`: coarse chapter/topic hint when available
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from embeddings_utils import DEFAULT_EMBEDDING_MODEL, load_sentence_transformer


load_dotenv()


def _ensure_raw_metadata(doc, *, path: str):
    stem = Path(path).stem
    metadata = dict(getattr(doc, "metadata", {}))
    metadata["source"] = path
    metadata["source_name"] = stem
    metadata["type"] = "raw_pdf"
    metadata["chunk_type"] = "raw"
    metadata.setdefault("chapter", stem)
    doc.metadata = metadata
    return doc


def _ensure_wiki_metadata(doc, *, path: str):
    stem = Path(path).stem
    metadata = dict(getattr(doc, "metadata", {}))
    metadata["source"] = path
    metadata["source_name"] = stem
    metadata["type"] = "wiki"
    metadata["chunk_type"] = "wiki"
    metadata.setdefault("chapter", stem)
    doc.metadata = metadata
    return doc


def load_raw_documents(raw_dir: str = "raw"):
    from langchain_community.document_loaders import PyPDFLoader

    docs = []
    if not os.path.exists(raw_dir):
        print(f"Warning: {raw_dir} does not exist, skipping raw PDFs.")
        return docs

    for filename in os.listdir(raw_dir):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(raw_dir, filename)
        loader = PyPDFLoader(path)
        pdf_docs = [_ensure_raw_metadata(doc, path=path) for doc in loader.load()]
        docs.extend(pdf_docs)
        print(f"Loaded PDF: {filename}")

    return docs


def load_wiki_documents(wiki_dir: str = "wiki"):
    from langchain_core.documents import Document

    docs = []
    if not os.path.exists(wiki_dir):
        print(f"Warning: {wiki_dir} does not exist, skipping wiki documents.")
        return docs

    for filename in os.listdir(wiki_dir):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(wiki_dir, filename)
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        doc = Document(page_content="[WIKI]\n" + text, metadata={})
        docs.append(_ensure_wiki_metadata(doc, path=path))
        print(f"Loaded wiki: {filename}")

    return docs


def split_wiki_documents(wiki_docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(wiki_docs)
    for index, doc in enumerate(split_docs):
        doc.metadata = dict(getattr(doc, "metadata", {}))
        doc.metadata["chunk_index"] = index
        doc.metadata["chunk_type"] = "wiki"
    return split_docs


def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    for index, doc in enumerate(split_docs):
        doc.metadata = dict(getattr(doc, "metadata", {}))
        doc.metadata["chunk_index"] = index
        doc.metadata.setdefault("chunk_type", "raw")
    print(f"Split complete: {len(split_docs)} chunks")
    return split_docs


def load_all_documents():
    raw_docs = load_raw_documents("raw")
    wiki_docs = load_wiki_documents("wiki")

    if wiki_docs:
        wiki_docs = split_wiki_documents(wiki_docs)
        print(f"Wiki chunks after splitting: {len(wiki_docs)}")

    print("Document loading complete:")
    print(f"  - Raw pages: {len(raw_docs)}")
    print(f"  - Wiki chunks: {len(wiki_docs)}")
    print(f"  - Total before final split: {len(raw_docs) + len(wiki_docs)}")

    return raw_docs + wiki_docs


def build_vectorstore(output_dir: str = "faiss_index"):
    print("=" * 50)
    print("Building vector store")
    print("=" * 50)

    documents = load_all_documents()
    if not documents:
        print("Error: no documents were loaded.")
        return None

    split_docs = split_documents(documents)

    print("Loading embedding model...")
    embedding_model = load_sentence_transformer(DEFAULT_EMBEDDING_MODEL)

    print("Encoding chunks...")
    texts = [doc.page_content for doc in split_docs]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    docstore = {}
    index_to_docstore_id = {}
    for i, doc in enumerate(split_docs):
        docstore[i] = doc
        index_to_docstore_id[i] = i

    with open(os.path.join(output_dir, "index.pkl"), "wb") as file:
        pickle.dump((docstore, index_to_docstore_id), file)

    print("=" * 50)
    print(f"Vector store saved to: {output_dir}/")
    print(f"  - Vectors: {index.ntotal}")
    print(f"  - Dimension: {dimension}")
    print("=" * 50)
    return index


if __name__ == "__main__":
    build_vectorstore()
