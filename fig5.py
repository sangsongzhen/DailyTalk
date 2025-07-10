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

# ✅ 固定选择的子目录编号
selected_ids = ["23", "321", "1949", "898", "1455"]
base_dir = "./output/result/7.5/900000"

# 存储 embedding 和标签
embeddings = []
speaker_labels = []
dialogue_ids = []

for dialog_id in selected_ids:
    target_dir = os.path.join(base_dir, dialog_id)
    if not os.path.isdir(target_dir):
        print(f"⚠️ 跳过不存在的目录: {target_dir}")
        continue

    for file in os.listdir(target_dir):
        if file.endswith(".wav"):
            parts = file.split("_")
            if len(parts) < 3:
                continue
            speaker = parts[1]  # 0 或 1
            filepath = os.path.join(target_dir, file)
            try:
                waveform = read_audio(filepath)
                emb = classifier.encode_batch(waveform.unsqueeze(0)).squeeze().cpu().numpy()
                embeddings.append(emb)
                speaker_labels.append(speaker)
                dialogue_ids.append(dialog_id)
            except Exception as e:
                print(f"Error processing {file}: {e}")

# 降维可视化
embeddings = np.array(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(10, 8))
colors = {'0': 'blue', '1': 'red'}
markers = ['o', 's', 'D', '^', 'v']
dialogue_set = list(sorted(set(dialogue_ids)))

for i, dialog_id in enumerate(dialogue_set):
    for speaker in ['0', '1']:
        idxs = [j for j in range(len(embeddings_2d)) if dialogue_ids[j] == dialog_id and speaker_labels[j] == speaker]
        if idxs:
            plt.scatter(
                embeddings_2d[idxs, 0], embeddings_2d[idxs, 1],
                c=colors[speaker],
                marker=markers[i % len(markers)],
                label=f"Dialog {dialog_id} - Speaker {speaker}",
                alpha=0.7
            )

plt.title("t-SNE of Speaker Embeddings from 5 Dialogues")
plt.legend()
plt.grid(True)
plt.savefig("tsne_5_dialogues.png")
plt.show()
