import os
import torch
import numpy as np
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from scipy.spatial.distance import cosine

# 设置路径
pred_root = "./output/result/7.5/900000/30"
ref_root = "./raw_data/DailyTalk/data/30"

# 加载 speaker encoder 模型
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# def extract_embedding(wav_path):
#     """提取单个 wav 的 speaker embedding"""
#     return classifier.encode_batch(wav_path).squeeze().cpu().numpy()
def extract_embedding(wav_path):
    """提取单个 wav 的 speaker embedding"""
    waveform = read_audio(wav_path)  # 加载音频为 Tensor
    return classifier.encode_batch(waveform.unsqueeze(0)).squeeze().cpu().numpy()

def get_common_files(pred_dir, ref_dir):
    """获取预测和参考中共有的文件名列表"""
    pred_files = set(f for f in os.listdir(pred_dir) if f.endswith(".wav"))
    ref_files = set(f for f in os.listdir(ref_dir) if f.endswith(".wav"))
    return sorted(list(pred_files & ref_files))

def compute_similarity(emb1, emb2):
    """计算余弦相似度"""
    return 1 - cosine(emb1, emb2)

if __name__ == '__main__':
    common_files = get_common_files(pred_root, ref_root)
    print(f"🔍 共有 {len(common_files)} 个对齐的音频文件")

    similarities = []

    for file in tqdm(common_files, desc="Comparing embeddings"):
        pred_path = os.path.join(pred_root, file)
        ref_path = os.path.join(ref_root, file)
        try:
            emb_pred = extract_embedding(pred_path)
            emb_ref = extract_embedding(ref_path)
            sim = compute_similarity(emb_pred, emb_ref)
            similarities.append(sim)
            print(f"{file}: similarity = {sim:.4f}")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    if similarities:
        avg_sim = np.mean(similarities)
        print(f"\n✅ 平均风格相似度（Cosine Similarity）: {avg_sim:.4f}")
    else:
        print("⚠️ 没有成功处理的音频文件")
