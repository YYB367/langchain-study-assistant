"""第 1 步：通过 LangChain 调用 DeepSeek，验证模型连接。"""
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit(
        "没有找到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，"
        "再把你的 DeepSeek API Key 填入 .env。"
    )

model = ChatOpenAI(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    temperature=0.2,
)

question = input("请输入一个问题：")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位耐心的中文 AI 入门老师。请用{language}回答。"),
        ("human", "{question}"),
    ]
)

chain = prompt | model

answer = chain.invoke(
    {
        "language": "简体中文",
        "question": question,
    }
)

print("\n助手回答：")
print(answer.content)
