# ui_app.py
import os
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from core_agent import CourseAssistantAgent
from config import BASE_DIR

st.set_page_config(page_title="成理智答 | BERT课程助教", page_icon="🎓", layout="wide")
st.title("🎓 CDUT 成理智答 - 智能课程助教")
st.markdown("基于 **LangChain Agent** 架构构建，支持课程进度查询、作业查询及专业知识检索。")


# 【核心性能优化】：使用 cache_resource 确保全局只初始化一次大脑和知识库！
@st.cache_resource(show_spinner=False)
def get_global_agent():
    print("🚀 首次启动，正在执行耗时的底层初始化...")
    agent_obj = CourseAssistantAgent()
    pdf_path = os.path.join(BASE_DIR, "test.pdf")
    try:
        agent_obj.rag.ingest_pdf(pdf_path)
    except Exception as e:
        st.error(f"知识库加载异常: {e}")
    return agent_obj


# 加载动画
with st.spinner("⚙️ 系统状态：正在连接云端向量记忆体..."):
    agent = get_global_agent()

# 侧边栏监控
with st.sidebar:
    st.header("⚙️ 助教状态监控")
    st.write("🟢 内存知识库: 已挂载并锁定")
    st.write("🟢 响应模式: 极速缓存模式")
    st.divider()
    st.write("👨‍💻 开发: 成理毕业生")

    # 清空当前会话的历史记忆
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