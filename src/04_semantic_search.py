import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT.parent / "hf-cache"))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

loader = PyPDFLoader(str(PROJECT_ROOT / "data" / "六页资料文字版.pdf"))
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "，", ""],
)
chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

question = input("请输入检索问题：")
results = vector_store.similarity_search(question, k=3)

for i, doc in enumerate(results, start=1):
    page = doc.metadata.get("page", "?")
    print(f"\n--- 第 {i} 条结果（PDF 第 {page + 1 if isinstance(page, int) else page} 页）---")
    print(doc.page_content[:500])

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

model = ChatOpenAI(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    temperature=0.2,
)

context = "\n\n".join(
    f"[PDF第{doc.metadata.get('page', 0) + 1}页]\n{doc.page_content}"
    for doc in results
)

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

chain = prompt | model
answer = chain.invoke({"question": question, "context": context})

print("\n--- 根据检索资料生成的回答 ---")
print(answer.content)
