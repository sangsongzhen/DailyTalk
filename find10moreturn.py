import os

# 设定生成结果根目录
root_dir = "./output/result/7.5/900000"

# 设置阈值：对话轮数 > 10
threshold = 15

# 结果列表
long_dialogs = []

# 遍历子目录
for subdir in sorted(os.listdir(root_dir)):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        # 统计 .wav 文件数量
        wav_files = [f for f in os.listdir(subdir_path) if f.endswith(".wav")]
        num_files = len(wav_files)

        if num_files > threshold:
            long_dialogs.append((subdir, num_files))

# 输出结果
print(f"📊 找到 {len(long_dialogs)} 个说话轮数大于 {threshold} 的对话文件夹：\n")
for name, count in long_dialogs:
    print(f"📁 对话 {name}: {count} 个 .wav 文件")
