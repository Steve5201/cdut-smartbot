# test_tools.py
from tools import get_course_syllabus, check_assignment_deadline


def run_tools_test():
    print("🛠️ 正在测试工具调用...")

    # 测试工具 1
    print("\n--- 测试: 查询课程进度 ---")
    # LangChain 工具调用时，参数以字典形式传入
    result_1 = get_course_syllabus.invoke({"week": 4})
    print(f"输入: 第4周")
    print(f"输出: {result_1}")

    # 测试工具 2
    print("\n--- 测试: 查询作业截止日期 ---")
    result_2 = check_assignment_deadline.invoke({"assignment_name": "老师，请问那个文本分类作业啥时候交啊？"})
    print(f"输入: 文本分类作业")
    print(f"输出: {result_2}")


if __name__ == "__main__":
    run_tools_test()