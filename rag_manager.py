# rag_manager.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from chromadb.config import Settings


class RAGManager:
    def __init__(self):
        print("⚙️ 正在初始化 RAG 模块...")

        print(f"📁 向量库将被持久化保存至绝对路径: {CHROMA_PERSIST_DIR}")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # 1. 初始化 Embedding 模型 (把文字变成数学向量)
        # 我们使用 shibing624/text2vec-base-chinese，这是一个非常经典且轻量级的中文向量模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )

        # 2. 初始化 Chroma 向量数据库 (保存在本地文件夹)
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="cdut_docs",
            client_settings=Settings(anonymized_telemetry=False)
        )

        # 3. 初始化文本切分器 (把几百页的PDF切成一小块一小块)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print("✅ RAG 模块初始化完成！")

    def ingest_pdf(self, file_path: str):
        """读取 PDF 并存入向量数据库"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 找不到文件: {file_path}")

        print(f"📄 正在读取文档: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        print(f"✂️ 正在切分文档...")
        split_docs = self.text_splitter.split_documents(docs)
        print(f"✅ 文档被切分为 {len(split_docs)} 个文本块。")

        print(f"💾 正在计算向量并写入数据库 (可能需要几秒到十几秒)...")
        self.vector_store.add_documents(split_docs)
        print("🎉 写入完成！文档知识已入库。")

    def get_retriever(self, top_k=3):
        """获取检索器，后续供 Agent 调用"""
        # top_k=3 表示每次用户提问，搜索最相关的 3 个文本块
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    def test_search(self, query: str):
        """用于测试的搜索方法"""
        print(f"\n🔍 正在数据库中搜索: '{query}'")
        docs = self.vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(docs):
            print(f"\n--- 匹配结果 {i + 1} ---")
            print(f"来源: {doc.metadata.get('source')} (第 {doc.metadata.get('page')} 页)")
            print(f"内容: {doc.page_content}")