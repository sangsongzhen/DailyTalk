import numpy as np
import os
import sys

def analyze_npy_file(file_path):
    """
    分析.npy文件的各种属性
    
    参数:
        file_path (str): .npy文件的路径
        
    返回:
        dict: 包含文件各种属性的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {"error": f"文件 {file_path} 不存在"}
        
        # 检查文件扩展名
        if not file_path.lower().endswith('.npy'):
            return {"error": "文件扩展名不是.npy"}
        
        # 加载npy文件
        data = np.load(file_path, mmap_mode='r')
        
        # 收集文件属性
        file_stats = os.stat(file_path)
        
        result = {
            "文件名": os.path.basename(file_path),
            "文件路径": os.path.abspath(file_path),
            "文件大小 (bytes)": file_stats.st_size,
            "文件创建时间": file_stats.st_ctime,
            "文件修改时间": file_stats.st_mtime,
            "数组形状": data.shape,
            "数组维度": data.ndim,
            "数据类型": str(data.dtype),
            "元素总数": data.size,
            "数组大小 (估计, bytes)": data.size * data.itemsize,
            "是否连续存储": np.isfortran(data),
            "字节顺序": data.dtype.byteorder,
            "元素字节大小": data.itemsize,
            "最小值": np.min(data) if data.size > 0 else "空数组",
            "最大值": np.max(data) if data.size > 0 else "空数组",
            "平均值": np.mean(data) if data.size > 0 and np.issubdtype(data.dtype, np.number) else "不适用",
            "标准差": np.std(data) if data.size > 0 and np.issubdtype(data.dtype, np.number) else "不适用"
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def print_analysis_results(results):
    """
    打印分析结果
    
    参数:
        results (dict): analyze_npy_file函数返回的结果字典
    """
    if "error" in results:
        print(f"错误: {results['error']}")
        return
    
    print("\n.npy文件分析结果:")
    print("=" * 50)
    for key, value in results.items():
        if key in ["文件创建时间", "文件修改时间"]:
            from datetime import datetime
            value = datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{key:>20}: {value}")
    print("=" * 50)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("使用方法: python analyze_npy.py <npy文件路径>")
    #     sys.exit(1)
    
    # file_path = sys.argv[1]
    file_path = '/home/ssz/TTS/DailyTalk/preprocessed_data/DailyTalk/text_emb/0-text_emb-0_0_d2.npy'
    results = analyze_npy_file(file_path)
    print_analysis_results(results)