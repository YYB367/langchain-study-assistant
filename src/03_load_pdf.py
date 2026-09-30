from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/六页资料文字版.pdf")
pages = loader.load()

print(f"读到了 {len(pages)} 页")
print("第一页内容预览：")
print(pages[0].page_content[:300])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "，", ""],
)

chunks = splitter.split_documents(pages)

print(f"\n切成了 {len(chunks)} 段")
for i, chunk in enumerate(chunks[:5], start=1):
    print(f"\n--- 第 {i} 段；页码信息：{chunk.metadata} ---")
    print(chunk.page_content[:300])