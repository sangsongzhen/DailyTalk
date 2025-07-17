import os
import torch
import numpy as np
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from scipy.spatial.distance import cosine

# 设置路径
pred_root = "./output/result/DailyTalk/900000"
ref_root = "./raw_data/DailyTalk/data"

# 指定要评估的对话编号（可以替换为任意 10 个）
# dialog_ids = [23, 403, 590, 877, 1046, 1172, 1618, 1829, 1983, 2463]
# 15 more
dialog_ids = [1112, 1126, 1238, 1298, 1325, 1452, 1898, 1908, 1936, 1987]

# 加载 speaker encoder 模型
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def extract_embedding(wav_path):
    """提取单个 wav 的 speaker embedding"""
    waveform = read_audio(wav_path)
    return classifier.encode_batch(waveform.unsqueeze(0)).squeeze().cpu().numpy()

def get_common_files(pred_dir, ref_dir):
    """获取预测和参考中共有的音频文件名"""
    pred_files = set(f for f in os.listdir(pred_dir) if f.endswith(".wav"))
    ref_files = set(f for f in os.listdir(ref_dir) if f.endswith(".wav"))
    return sorted(list(pred_files & ref_files))

def compute_similarity(emb1, emb2):
    """计算余弦相似度"""
    return 1 - cosine(emb1, emb2)

# 主函数
if __name__ == '__main__':
    all_similarities = []
    dialog_sim_dict = {}

    for dialog_id in dialog_ids:
        pred_dir = os.path.join(pred_root, str(dialog_id))
        ref_dir = os.path.join(ref_root, str(dialog_id))

        if not os.path.exists(pred_dir) or not os.path.exists(ref_dir):
            print(f"⚠️ 对话 {dialog_id} 的路径不存在，跳过。")
            continue

        common_files = get_common_files(pred_dir, ref_dir)
        print(f"\n📁 对话 {dialog_id} - 共有 {len(common_files)} 个音频文件")

        similarities = []
        for file in tqdm(common_files, desc=f"[{dialog_id}] Comparing embeddings", leave=False):
            pred_path = os.path.join(pred_dir, file)
            ref_path = os.path.join(ref_dir, file)
            try:
                emb_pred = extract_embedding(pred_path)
                emb_ref = extract_embedding(ref_path)
                sim = compute_similarity(emb_pred, emb_ref)
                similarities.append(sim)
            except Exception as e:
                print(f"❌ 处理 {file} 时出错: {e}")

        if similarities:
            avg_sim = np.mean(similarities)
            dialog_sim_dict[dialog_id] = avg_sim
            all_similarities.extend(similarities)
            print(f"✅ 对话 {dialog_id} 的平均相似度: {avg_sim:.4f}")
        else:
            print(f"⚠️ 对话 {dialog_id} 无有效数据")

    if all_similarities:
        overall_avg = np.mean(all_similarities)
        print("\n📊 每个对话的相似度：")
        for k, v in dialog_sim_dict.items():
            print(f" - 对话 {k}: {v:.4f}")
        print(f"\n🎯 所有对话的平均相似度: {overall_avg:.4f}")
    else:
        print("⚠️ 没有处理成功的音频")
