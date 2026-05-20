# test_api.py
from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL


def test_deepseek_connection():
    print(f"🔄 准备连接 DeepSeek API...")
    print(f"📦 使用模型: {LLM_MODEL}")

    # 实例化 LangChain 的大模型对象
    # 因为 DeepSeek 完美兼容 OpenAI 的 API 格式，所以我们直接用 ChatOpenAI 类
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=256,
        temperature=0.7  # 温度值：0比较严谨，1比较发散
    )

    try:
        # 调用大模型，给它发一句话
        response = llm.invoke("你好，请用一句话证明你是一个高智商的AI助手。")
        print("\n✅ 连接成功！DeepSeek 返回内容：")
        print("---------------------------------")
        print(response.content)
        print("---------------------------------")
    except Exception as e:
        print(f"\n❌ 连接失败，请检查网络或 API Key。错误信息：{e}")


if __name__ == "__main__":
    test_deepseek_connection()