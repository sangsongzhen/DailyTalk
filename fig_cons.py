import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
pred_dir = "./output/result/7.5/900000/64"
gt_dir = "/home/ssz/TTS/DailyTalk/raw_data/DailyTalk/data/64"

embeddings = []
labels = []  # 'Generated' or 'GroundTruth'
filenames = []

files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".wav")])

print(f"共发现 {len(files)} 对音频文件，开始提取 embedding 并对比...")

for fname in tqdm(files):
    pred_path = os.path.join(pred_dir, fname)
    gt_path = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"⚠️ 缺少真值文件: {fname}")
        continue

    try:
        emb_pred = get_embedding(pred_path)
        emb_gt = get_embedding(gt_path)

        embeddings.append(emb_gt)
        labels.append("GroundTruth")
        filenames.append(fname)

        embeddings.append(emb_pred)
        labels.append("Generated")
        filenames.append(fname)

    except Exception as e:
        print(f"❌ 处理 {fname} 失败: {e}")

# 转为数组
embeddings = np.array(embeddings)

# 降维
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
emb_2d = tsne.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(10, 8))

for i in range(0, len(emb_2d), 2):
    gt = emb_2d[i]
    gen = emb_2d[i+1]
    fname = filenames[i]

    # 连线
    plt.plot([gt[0], gen[0]], [gt[1], gen[1]], color='gray', linewidth=0.8, alpha=0.6)

    # 点
    plt.scatter(gt[0], gt[1], color='blue', label='GroundTruth' if i == 0 else "", marker='o')
    plt.scatter(gen[0], gen[1], color='red', label='Generated' if i == 0 else "", marker='x')

plt.title("Speaker Embedding Comparison (t-SNE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_speaker_embedding.png")
plt.show()
