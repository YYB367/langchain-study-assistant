"""复用已经建立的 PDF 检索库。"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "vector_db"
# 根据当前六页 PDF 的初步检查设定；距离越小，段落越相关。
MAX_DISTANCE = 0.92
# 严格筛选没有结果时，有限度地放宽距离，避免漏掉已保存的原文。
FALLBACK_DISTANCE = 1.07

# 与现有练习脚本共用已下载的中文向量模型缓存。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT.parent / "hf-cache"))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def load_saved_vector_store() -> Chroma:
    """打开 05_build_index.py 已经保存的 Chroma 检索库。"""
    if not (DB_PATH / "chroma.sqlite3").is_file():
        raise FileNotFoundError("找不到检索库，请先运行 src/05_build_index.py")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="pdf_materials",
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )


def retrieve_documents(vector_store: Chroma, question: str, k: int = 3) -> list[Document]:
    """从已打开的检索库中找出与问题最相关的段落。"""
    matches = vector_store.similarity_search_with_score(question, k=k)
    documents = [doc for doc, distance in matches if distance <= MAX_DISTANCE]
    if not documents:
        return [doc for doc, distance in matches if distance <= FALLBACK_DISTANCE]

    # 切分时相邻段有重叠文字；给首段补上紧邻的上一段，保留句子开头。
    first = documents[0]
    page = first.metadata.get("page")
    if isinstance(page, int):
        same_page = vector_store.get(where={"page": page}, include=["documents", "metadatas"])
        for content, metadata in zip(same_page["documents"], same_page["metadatas"]):
            if content in (doc.page_content for doc in documents):
                continue
            max_overlap = min(80, len(content), len(first.page_content))
            if any(content.endswith(first.page_content[:length]) for length in range(max_overlap, 39, -1)):
                documents.append(Document(page_content=content, metadata=metadata))
                break
    return documents


def generate_answer(question: str, documents: list[Document]) -> str:
    """让 DeepSeek 只依据检索到的段落回答问题。"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查本机 .env 配置")

    context_parts = []
    for doc in documents:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "未知"
        context_parts.append(f"[PDF第{page_number}页]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "请只依据提供的资料回答问题。如果资料不足，请明确说明。"
                "回答中的事实要标注对应页码，例如[PDF第3页]。不要编造资料或页码。",
            ),
            ("human", "问题：{question}\n\n参考资料：\n{context}"),
        ]
    )
    model = ChatOpenAI(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0.2,
    )
    answer = (prompt | model).invoke({"question": question, "context": context})
    return str(answer.content)
