# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 大模型 (LLM) 配置 - 使用 DeepSeek 官网最新配置
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # 兼容 OpenAI SDK 的入口
LLM_MODEL = "deepseek-v4-flash"                 # 官网推荐最新高速模型

# ==========================================
# 2. 检索增强 (RAG) 向量数据库配置
# ==========================================
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")            # 向量数据库本地持久化保存路径
CHUNK_SIZE = 500                                                            # 长文档切分时，每块文本的长度
CHUNK_OVERLAP = 50                                                          # 文本块之间的重叠字数（防止切断关键句子）