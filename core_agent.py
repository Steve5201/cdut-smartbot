# core_agent.py
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, BASE_DIR
from rag_manager import RAGManager
from tools import get_course_syllabus, check_assignment_deadline


class CourseAssistantAgent:
    def __init__(self):
        print("🧠 正在唤醒具有自愈能力的 Agent 大脑...")

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3,
            max_tokens=1024,
            extra_body={"thinking": {"type": "disabled"}}
        )

        self.rag = RAGManager()

        # =================【工具 1：知识检索】=================
        def rag_search_func(query: str) -> str:
            if not self.rag.check_db_exists():
                return "❌ 检索失败：本地数据库不存在。请调用 build_knowledge_base 工具生成数据库后重试。"
            try:
                docs = self.rag.get_retriever().invoke(query)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                return f"❌ 检索遇到异常: {e}。这可能是数据库损坏，请调用 build_knowledge_base 工具重建。"

        rag_tool = Tool(
            name="search_course_knowledge",
            description="当需要回答关于课件内容的专业问题时调用。如果返回失败提示，请先调用重建工具。",
            func=rag_search_func
        )

        # =================【工具 2：状态检查】=================
        def check_db_func(query: str) -> str:
            return "知识库已就绪。" if self.rag.check_db_exists() else "知识库缺失/损坏。"

        check_tool = Tool(
            name="check_knowledge_base",
            description="检查本地向量知识库是否健康存在。输入任意字符即可调用。",
            func=check_db_func
        )

        # =================【工具 3：触发自愈重建】=================
        def build_db_func(query: str) -> str:
            pdf_path = os.path.join(BASE_DIR, "test.pdf")
            return self.rag.ingest_pdf(pdf_path)

        build_tool = Tool(
            name="build_knowledge_base",
            description="当知识库损坏、缺失或搜索报错时，调用此工具重新读取 PDF 构建知识库。输入任意字符调用。",
            func=build_db_func
        )

        self.tools = [get_course_syllabus, check_assignment_deadline, rag_tool, check_tool, build_tool]

        # 【核心：系统提示词升级，教会它自救逻辑】
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个名叫'成理智答'的课程助教。
            你的职责是回答问题。
            【极其重要的行动规范】：
            1. 当用户询问专业知识时，调用 search_course_knowledge。
            2. 如果工具返回“数据库不存在”或报错，你必须**立即**调用 build_knowledge_base 重建数据库！
            3. 重建成功后，你必须**再次调用** search_course_knowledge 查找答案，最后将答案整理并输出给用户。不要告诉用户你在修数据库，假装什么事都没发生地把答案给他。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        self.chat_history = []

    def chat(self, user_input: str, callbacks=None) -> str:
        try:
            invoke_config = {"callbacks": callbacks} if callbacks else {}
            response = self.agent_executor.invoke(
                {"input": user_input, "chat_history": self.chat_history},
                config=invoke_config
            )
            answer = response.get("output", "抱歉，大脑运转异常。")
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=answer))
            return answer
        except Exception as e:
            return f"系统异常: {str(e)}"