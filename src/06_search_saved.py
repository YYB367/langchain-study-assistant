import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT.parent / "hf-cache"))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = PROJECT_ROOT / "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 从项目目录重新打开已保存的向量库
vector_store = Chroma(
    collection_name="pdf_materials",
    persist_directory=str(DB_PATH),
    embedding_function=embeddings,
)

question = input("请输入检索问题：")
results = vector_store.similarity_search(question, k=3)

for i, doc in enumerate(results, start=1):
    page = doc.metadata.get("page", "?")
    page = page + 1 if isinstance(page, int) else page
    print(f"\n--- 第 {i} 条结果（PDF 第 {page} 页）---")
    print(doc.page_content[:500])
