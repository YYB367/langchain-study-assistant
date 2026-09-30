import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT.parent / "hf-cache"))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = PROJECT_ROOT / "data" / "六页资料文字版.pdf"
DB_PATH = PROJECT_ROOT / "vector_db"

# 读取 PDF 并切分
pages = PyPDFLoader(str(PDF_PATH)).load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "，", ""],
)
chunks = splitter.split_documents(pages)

# 把段落转换成向量
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 建立 Chroma 向量库并保存在项目目录
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_materials",
    persist_directory=str(DB_PATH),
)

print(f"完成：已将 {len(chunks)} 段保存到 {DB_PATH}")
