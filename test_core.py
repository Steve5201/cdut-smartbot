# test_core.py
from core_agent import CourseAssistantAgent


def run_chat_test():
    # 实例化 Agent
    agent = CourseAssistantAgent()

    print("\n" + "=" * 50)
    print("🎓 欢迎使用《BERT基础教程》助教系统 (输入 'quit' 退出)")
    print("=" * 50 + "\n")

    while True:
        user_input = input("👨‍🎓 学生提问: ")
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break

        print("\n🤖 助教思考中...\n(请观察下方控制台输出的绿字思考过程)")
        print("-" * 30)

        # 调用核心回答接口
        answer = agent.chat(user_input)

        print("-" * 30)
        print(f"👩‍🏫 助教回答:\n{answer}\n")


if __name__ == "__main__":
    run_chat_test()