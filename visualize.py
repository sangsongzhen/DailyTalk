import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from tqdm import tqdm

# 初始化模型
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir"
)

def get_embedding(path):
    wav = read_audio(path)
    emb = classifier.encode_batch(wav.unsqueeze(0)).squeeze().cpu().numpy()
    return emb

# 设置路径
gt_dir = "./raw_data/DailyTalk/data/23"
base_dir = "/home/ssz/TTS/DailyTalk_old/output/result/DailyTalk/900000/23"
ours_dir = "./output/result/7.5/900000/23"

files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".wav")])

# 存储 embedding 和标签
embeddings = []
labels = []
pairs = []  # 用于连线: [(gt_idx, base_idx), (gt_idx, ours_idx)]

print(f"共发现 {len(files)} 个 GT 文件，开始提取 embedding 并对比...")

for fname in tqdm(files):
    gt_path = os.path.join(gt_dir, fname)
    base_path = os.path.join(base_dir, fname)
    ours_path = os.path.join(ours_dir, fname)

    if not os.path.exists(base_path) or not os.path.exists(ours_path):
        print(f"⚠️ 缺少对应文件: {fname}")
        continue

    try:
        emb_gt = get_embedding(gt_path)
        emb_base = get_embedding(base_path)
        emb_ours = get_embedding(ours_path)

        # 存储并记录对应关系
        gt_idx = len(embeddings)
        embeddings.append(emb_gt)
        labels.append("GT")

        base_idx = len(embeddings)
        embeddings.append(emb_base)
        labels.append("Base")
        pairs.append((gt_idx, base_idx))

        ours_idx = len(embeddings)
        embeddings.append(emb_ours)
        labels.append("Ours")
        pairs.append((gt_idx, ours_idx))

    except Exception as e:
        print(f"❌ 处理 {fname} 失败: {e}")

# 转为数组并降维
embeddings = np.array(embeddings)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(10, 8))

for (gt_idx, other_idx) in pairs:
    gt = emb_2d[gt_idx]
    other = emb_2d[other_idx]
    label = labels[other_idx]

    color = 'red' if label == "Base" else 'green'
    plt.plot([gt[0], other[0]], [gt[1], other[1]], color=color, linewidth=0.8, alpha=0.5)

    plt.scatter(gt[0], gt[1], color='blue', label='GT' if gt_idx == 0 else "", marker='o')
    plt.scatter(other[0], other[1], color=color, label=label if gt_idx == 0 else "", marker='x')

plt.title("GT - Base vs GT - Ours Embedding Comparison (PCA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_speaker_embedding.png")
plt.show()
