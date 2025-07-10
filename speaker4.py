import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from tqdm import tqdm

# 初始化 speaker embedding 模型
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir"
)

def get_embedding(path):
    wav = read_audio(path)
    emb = classifier.encode_batch(wav.unsqueeze(0)).squeeze().cpu().numpy()
    return emb

# 设置路径
base_dir = "/home/ssz/TTS/DailyTalk_old/output/result/DailyTalk/900000/23"
ours_dir = "./output/result/7.5/900000/23"

# 获取文件
base_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".wav")])
ours_files = sorted([f for f in os.listdir(ours_dir) if f.endswith(".wav")])

# 存储
embeddings = []
labels = []

print("开始提取 Base 和 Ours 的说话人 embedding...")

def get_speaker_id(filename):
    """从文件名中提取说话人编号，比如 xxx_0.wav -> 0"""
    if '_' in filename:
        return filename.split('_')[-1].replace('.wav', '')
    return 'unknown'

# Base
for f in tqdm(base_files):
    speaker_id = get_speaker_id(f)
    path = os.path.join(base_dir, f)
    try:
        emb = get_embedding(path)
        if np.isnan(emb).any() or np.allclose(emb, 0):
            print(f"⚠️ Base 中 {f} 的 embedding 无效")
            continue
        embeddings.append(emb)
        labels.append(f'Base_Speaker {speaker_id}')
    except:
        print(f"❌ 处理 Base 文件失败: {f}")

# Ours
for f in tqdm(ours_files):
    speaker_id = get_speaker_id(f)
    path = os.path.join(ours_dir, f)
    try:
        emb = get_embedding(path)
        if np.isnan(emb).any() or np.allclose(emb, 0):
            print(f"⚠️ Ours 中 {f} 的 embedding 无效")
            continue
        embeddings.append(emb)
        labels.append(f'Ours_Speaker {speaker_id}')
    except:
        print(f"❌ 处理 Ours 文件失败: {f}")

# 转为数组并降维
if len(embeddings) == 0:
    raise ValueError("没有有效的 embedding 数据，请检查音频路径或模型输出")

embeddings = np.array(embeddings)

# 打印分布范围
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding min/max: {embeddings.min()}/{embeddings.max()}")

# 降维
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(10, 8))
colors = {
    'Base_Speaker 0': 'red',
    'Base_Speaker 1': 'orange',
    'Ours_Speaker 0': 'blue',
    'Ours_Speaker 1': 'green'
}
markers = {
    'Base_Speaker 0': 'o',
    'Base_Speaker 1': 'o',
    'Ours_Speaker 0': 'x',
    'Ours_Speaker 1': 'x'
}

for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    points = emb_2d[indices]
    color = colors.get(label, 'gray')
    marker = markers.get(label, '.')
    plt.scatter(points[:, 0], points[:, 1], color=color, marker=marker, label=label, alpha=0.8)

plt.title("Speaker Embedding Comparison: Base vs Ours (PCA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
