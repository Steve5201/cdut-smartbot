# test_rag.py
from rag_manager import RAGManager


def run_rag_test():
    # 实例化 RAG 管理器
    # 第一次运行会从 HuggingFace 自动下载模型文件（大概 100多MB），需要稍微等一会
    rag = RAGManager()

    # 1. 测试灌入 PDF 文档
    # 假设你放了一个叫 test.pdf 的文件在项目根目录
    pdf_path = "./test.pdf"
    rag.ingest_pdf(pdf_path)

    # 2. 测试搜索
    # 请把下面的问题改成你 test.pdf 文档里实际包含的内容！！
    test_question = "提取一个文档里面的关键词作为问题写在这里"
    rag.test_search(test_question)


if __name__ == "__main__":
    run_rag_test()