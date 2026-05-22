# rag_manager.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class RAGManager:
    def __init__(self):
        print("⚙️ 正在初始化 RAG 模块...")
        # 1. 初始化 Embedding 模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )

        # 2. 初始化 Chroma 向量数据库 (回归最纯粹、最稳定的原版写法！)
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="cdut_docs"
        )

        # 3. 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print("✅ RAG 模块初始化完成！")

    def check_db_exists(self) -> bool:
        """【新增】给 Agent 用的检查函数：判断本地是否有真实的数据库文件"""
        db_file = os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3")
        # 检查文件是否存在，且大小大于 0 字节
        return os.path.exists(db_file) and os.path.getsize(db_file) > 0

    def ingest_pdf(self, file_path: str):
        """读取 PDF 并存入向量数据库 (回归原版 add_documents，保证落盘！)"""
        if not os.path.exists(file_path):
            return f"❌ 找不到文件: {file_path}"

        print(f"📄 正在读取文档: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        print(f"✂️ 正在切分文档...")
        split_docs = self.text_splitter.split_documents(docs)
        print(f"✅ 文档被切分为 {len(split_docs)} 个文本块。")

        print(f"💾 正在计算向量并写入数据库...")
        # 原版最稳的写入方式，绝对能生成文件
        self.vector_store.add_documents(split_docs)
        print("🎉 写入完成！文档知识已入库。")
        return "✅ 知识库构建成功！请重新发起检索。"

    def get_retriever(self, top_k=3):
        """获取检索器"""
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def test_search(self, query: str):
        """用于测试的搜索方法"""
        print(f"\n🔍 正在数据库中搜索: '{query}'")
        docs = self.vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(docs):
            print(f"\n--- 匹配结果 {i + 1} ---")
            print(f"来源: {doc.metadata.get('source')} (第 {doc.metadata.get('page')} 页)")
            print(f"内容: {doc.page_content}")