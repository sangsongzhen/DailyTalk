import sys
from collections import Counter
import datetime
import numpy as np

def analyze_list(input_list):
    """
    分析Python列表的各种属性
    
    参数:
        input_list (list): 要分析的列表
        
    返回:
        dict: 包含列表各种属性的字典
    """
    analysis = {}
    
    # 基础属性
    analysis['长度'] = len(input_list)
    analysis['内存占用(估计, bytes)'] = sys.getsizeof(input_list)
    
    # 元素类型分析
    type_counter = Counter()
    for item in input_list:
        type_counter[type(item).__name__] += 1
    
    analysis['元素类型分布'] = dict(type_counter)
    
    # 嵌套列表分析
    nested_info = {
        '是否包含嵌套列表': any(isinstance(item, list) for item in input_list),
        '最大嵌套深度': get_max_depth(input_list) if any(isinstance(item, list) for item in input_list) else 0
    }
    analysis['嵌套信息'] = nested_info
    
    # 数值列表的特殊分析
    if all(isinstance(item, (int, float, np.number)) for item in input_list if not isinstance(item, bool)):
        try:
            numerical_list = [float(item) for item in input_list]
            analysis['数值分析'] = {
                '总和': sum(numerical_list),
                '平均值': np.mean(numerical_list),
                '中位数': np.median(numerical_list),
                '最小值': min(numerical_list),
                '最大值': max(numerical_list),
                '标准差': np.std(numerical_list) if len(numerical_list) > 1 else 0,
                '唯一值数量': len(set(numerical_list))
            }
        except:
            pass
    
    # 字符串列表的特殊分析
    if all(isinstance(item, str) for item in input_list):
        analysis['字符串分析'] = {
            '总字符数': sum(len(s) for s in input_list),
            '平均长度': sum(len(s) for s in input_list) / len(input_list),
            '最短字符串长度': min(len(s) for s in input_list),
            '最长字符串长度': max(len(s) for s in input_list)
        }
    
    # 时间信息
    analysis['分析时间'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return analysis

def get_max_depth(lst):
    """
    计算列表的最大嵌套深度
    
    参数:
        lst (list): 要分析的列表
        
    返回:
        int: 最大嵌套深度
    """
    if not isinstance(lst, list):
        return 0
    return 1 + max((get_max_depth(item) for item in lst), default=0)

def print_analysis_results(results):
    """
    打印分析结果
    
    参数:
        results (dict): analyze_list函数返回的结果字典
    """
    print("\n列表分析结果:")
    print("=" * 50)
    
    # 打印基础信息
    print(f"{'长度':<20}: {results['长度']}")
    print(f"{'内存占用(估计)':<20}: {results['内存占用(估计, bytes)']} bytes")
    
    # 打印类型分布
    print("\n元素类型分布:")
    for type_name, count in results['元素类型分布'].items():
        print(f"  {type_name:<15}: {count} ({count/results['长度']:.1%})")
    
    # 打印嵌套信息
    print("\n嵌套信息:")
    print(f"  是否包含嵌套列表: {'是' if results['嵌套信息']['是否包含嵌套列表'] else '否'}")
    if results['嵌套信息']['是否包含嵌套列表']:
        print(f"  最大嵌套深度: {results['嵌套信息']['最大嵌套深度']}")
    
    # 打印数值分析
    if '数值分析' in results:
        print("\n数值分析:")
        for key, value in results['数值分析'].items():
            print(f"  {key:<10}: {value:.4f}" if isinstance(value, float) else f"  {key:<10}: {value}")
    
    # 打印字符串分析
    if '字符串分析' in results:
        print("\n字符串分析:")
        for key, value in results['字符串分析'].items():
            print(f"  {key:<15}: {value}")
    
    print("\n分析时间:", results['分析时间'])
    print("=" * 50)

if __name__ == "__main__":

    
    print("正在分析示例列表...")
    x = list()
    results = analyze_list(x)
    print_analysis_results(results)
    