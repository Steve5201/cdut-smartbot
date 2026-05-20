# tools.py
from langchain_core.tools import tool


# 使用 @tool 装饰器，LangChain 会自动把这个 Python 函数转换成大模型能理解的工具
# 注意：函数下方的注释 (Docstring) 非常重要！大模型就是通过看这些注释，来决定什么时候调用这个工具的。

@tool
def get_course_syllabus(week: int) -> str:
    """
    获取《BERT基础教程》课程某一周的教学进度或教学大纲。
    当学生询问“第X周学什么”、“进度是什么”时，调用此工具。
    参数 week: 周数（整数类型），例如：1, 2, 3。
    """
    # 真实场景中这里可以查数据库或 API，这里我们用字典模拟数据库
    syllabus = {
        1: "第一周：自然语言处理 (NLP) 基础与 Transformer 架构入门",
        2: "第二周：BERT 预训练模型原理、掩码语言模型 (MLM)与下一句预测 (NSP)",
        3: "第三周：使用 HuggingFace 和 ktrain 库加载预训练模型",
        4: "第四周：BERT 的微调应用实战（文本分类与文本摘要）",
        5: "第五周：跨语言 BERT (mBERT) 与模型部署"
    }
    # 返回查询结果
    return syllabus.get(week, f"未查询到第 {week} 周的教学安排，请提醒学生询问教务老师。")


@tool
def check_assignment_deadline(assignment_name: str) -> str:
    """
    查询特定课程作业的提交截止日期。
    当学生询问“某某作业什么时候交”、“截止日期是多少”时，调用此工具。
    参数 assignment_name: 作业名称或关键字，例如："文本分类", "期中大作业"。
    """
    # 模拟数据库中的作业截止日期
    deadlines = {
        "文本分类": "2024年5月20日 23:59",
        "文本摘要": "2024年6月10日 23:59",
        "期中大作业": "2024年6月30日 23:59",
    }

    # 模糊匹配作业名称
    for key, val in deadlines.items():
        if key in assignment_name:
            return f"系统查询到作业【{key}】的截止日期是：{val}"

    return "未找到相关作业的截止日期，可能暂未发布或作业名称不准确。"