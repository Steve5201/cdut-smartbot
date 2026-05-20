# ui_app.py
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from core_agent import CourseAssistantAgent
import os
from config import BASE_DIR

# 1. 设置网页标题和全局配置 (给网信处领导看的排面)
st.set_page_config(
    page_title="成理智答 | BERT课程助教",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 CDUT 成理智答 - 智能课程助教")
st.markdown("基于 **LangChain Agent** 架构构建，支持课程进度查询、作业查询及专业知识检索。")

# 2. 初始化 Agent 大脑并存入 Session State (内存模式)
if "agent_instance" not in st.session_state:
    with st.spinner("⚙️ 正在云端内存中构建向量知识库，约需 10-20 秒，请稍候..."):
        agent_obj = CourseAssistantAgent()

        # 强制每次启动会话时，把 PDF 读入内存
        pdf_path = os.path.join(BASE_DIR, "test.pdf")
        try:
            agent_obj.rag.ingest_pdf(pdf_path)
            st.session_state.agent_instance = agent_obj
            st.success("✅ 云端知识库加载成功！")
        except Exception as e:
            st.error(f"❌ 读取 PDF 失败: {e}")

agent = st.session_state.agent_instance

# 3. 渲染侧边栏
with st.sidebar:
    st.header("⚙️ 助教状态监控")
    st.write("🟢 大模型: 已连接")
    st.write("🟢 向量库: 已挂载")
    st.write("🟢 核心工具数: 3")
    st.divider()
    st.write("👨‍💻 开发人员: 成都理工大学 地质专业毕业生")
    st.write("✨ 核心技术: ReAct Agent, RAG, Tool Calling")

    # 清空记忆按钮
    if st.button("🗑️ 清空历史对话"):
        agent.chat_history = []
        st.success("记忆已清空")

# 4. 渲染聊天气泡 (根据 Agent 的历史记忆自动绘制)
for msg in agent.chat_history:
    # 判断是用户还是 AI
    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 5. 接收用户输入并触发响应
user_input = st.chat_input("请向助教提问 (例如：文本摘要作业啥时候交？什么是BERT？)")

if user_input:
    # 先把用户的问题在界面上画出来
    with st.chat_message("user"):
        st.markdown(user_input)

    # 再准备画 AI 的回答
    with st.chat_message("assistant"):
        # 这个神奇的组件，会把 Agent 的内部思考过程截获并显示在网页上！
        st_callback = StreamlitCallbackHandler(st.container())

        # 调用核心大脑，传入 callback 画笔
        answer = agent.chat(user_input, callbacks=[st_callback])

        # 最终回答显示出来
        st.markdown(answer)