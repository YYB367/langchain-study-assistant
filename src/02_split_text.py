from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
水盐平衡对人体很重要。

运动时，人体会通过出汗散热。大量出汗会使身体流失水分和盐分，
因此运动者需要及时补充水分。

如果身体失水过多，可能影响身体机能。
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5,
    separators=["\n\n", "\n", "。", "，", ""],
)

chunks = splitter.split_text(text)

print(f"切出了 {len(chunks)} 段：")
for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- 第 {i} 段，{len(chunk)} 个字符 ---")
    print(chunk)