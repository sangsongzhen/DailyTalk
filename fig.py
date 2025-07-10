import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from speechbrain.pretrained import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from tqdm import tqdm

# 初始化模型
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="tmpdir"
)

# 设置路径
target_dir = "./output/result/7.5/900000/23"
embeddings = []
labels = []

# 遍历文件夹
for file in tqdm(os.listdir(target_dir)):
    if file.endswith(".wav"):
        speaker_label = file.split("_")[1]  # 0 or 1 from file name
        filepath = os.path.join(target_dir, file)
        try:
            waveform = read_audio(filepath)
            emb = classifier.encode_batch(waveform.unsqueeze(0)).squeeze().cpu().numpy()
            embeddings.append(emb)
            labels.append(speaker_label)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# 降维
embeddings = np.array(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化
colors = {'0': 'blue', '1': 'red'}
plt.figure(figsize=(8, 6))
for label in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], 
                label=f"Speaker {label}", c=colors[label], alpha=0.7)

plt.title("t-SNE of Speaker Embeddings")
plt.legend()
plt.grid(True)
plt.savefig("tsne_speaker_embedding.png")
plt.show()
