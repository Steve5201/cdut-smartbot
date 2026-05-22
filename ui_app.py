# ui_app.py
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from core_agent import CourseAssistantAgent

st.set_page_config(page_title="成理智答 | BERT课程助教", page_icon="🎓", layout="wide")
st.title("🎓 CDUT 成理智答 - 智能课程助教")

# 去掉所有恶心的判断逻辑，只实例化 Agent！
@st.cache_resource(show_spinner=False)
def get_global_agent():
    return CourseAssistantAgent()

agent = get_global_agent()

with st.sidebar:
    st.header("⚙️ 助教状态监控")
    st.write("🟢 智能调度: 自愈合 Agent 模式")
    if st.button("🗑️ 清空历史对话"):
        agent.chat_history = []
        st.success("记忆已清空")

for msg in agent.chat_history:
    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("请向助教提问...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        answer = agent.chat(user_input, callbacks=[st_callback])
        st.markdown(answer)