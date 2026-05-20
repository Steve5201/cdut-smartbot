# core_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
from rag_manager import RAGManager
from tools import get_course_syllabus, check_assignment_deadline


class CourseAssistantAgent:
    def __init__(self):
        print("🧠 正在唤醒 Agent 大脑...")

        # 1. 初始化大模型
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3,
            max_tokens=1024,
            extra_body={"thinking": {"type": "disabled"}}
        )

        # 2. 初始化 RAG 并将其封装为 Tool (标准优雅写法)
        self.rag = RAGManager()
        retriever = self.rag.get_retriever(top_k=3)
        rag_tool = create_retriever_tool(
            retriever=retriever,
            name="search_bert_course_knowledge",
            description="当学生询问关于 BERT、自然语言处理、提取式摘要任务等专业学术问题时，必须使用此工具查询课件资料。"
        )

        # 3. 组装工具箱
        self.tools = [get_course_syllabus, check_assignment_deadline, rag_tool]

        # 4. 设定系统提示词 (Prompt Engineering)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个名叫'成理智答'的课程助教，负责辅助《BERT基础教程》课程。
            你的职责是：专业、耐心地回答学生的问题。
            请优先使用你手中的工具来获取准确信息，不要自己编造截止日期或课程安排。
            如果学生只是打招呼，你可以直接友善地回应。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 5. 创建 Agent 及执行器
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        # verbose=True 会在控制台打印出 Agent 绿色的思考过程
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        # 6. 初始化短期记忆
        self.chat_history = []
        print("✅ Agent 大脑准备就绪！")

    def chat(self, user_input: str, callbacks=None) -> str:
        """对外暴露的对话接口"""
        try:
            # 运行 Agent，如果有 callbacks 就传进去，方便前端渲染思考过程
            invoke_config = {"callbacks": callbacks} if callbacks else {}

            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": self.chat_history},
                config=invoke_config  # 【改动】注入回调配置
            )

            # 提取回答文本
            answer = response.get("output", "抱歉，大脑运转出现了点异常。")

            # 更新记忆
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=answer))

            return answer
        except Exception as e:
            return f"系统异常: {str(e)}"