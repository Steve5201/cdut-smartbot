# rag_manager.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHUNK_SIZE, CHUNK_OVERLAP


class RAGManager:
    def __init__(self):
        print("⚙️ 正在初始化 RAG 模块...")

        # 1. 初始化 Embedding 模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )

        # 2. 初始化 Chroma 向量数据库
        # 【核心修改】：去掉了 persist_directory，启用极速且安全的“内存模式”！
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            collection_name="cdut_docs"
        )

        # 3. 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print("✅ RAG 模块初始化完成！")

    def ingest_pdf(self, file_path: str):
        """读取 PDF 并存入向量数据库"""
        abs_file_path = os.path.abspath(file_path)
        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(f"❌ 找不到文件: {abs_file_path}")

        print(f"📄 正在读取文档: {abs_file_path}")
        loader = PyPDFLoader(abs_file_path)
        docs = loader.load()

        print(f"✂️ 正在切分文档...")
        split_docs = self.text_splitter.split_documents(docs)

        print(f"💾 正在计算向量并写入内存数据库...")
        self.vector_store.add_documents(split_docs)
        print("🎉 写入完成！文档知识已入内存。")

    def get_retriever(self, top_k=3):
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})