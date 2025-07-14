# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from speechbrain.pretrained import EncoderClassifier
# from speechbrain.dataio.dataio import read_audio

# # 初始化模型
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir"
# )

# def extract_embedding(path):
#     wav = read_audio(path)
#     emb = classifier.encode_batch(wav.unsqueeze(0)).squeeze().cpu().numpy()
#     return emb

# def load_embeddings_by_speaker(folder_path):
#     embeddings = []
#     labels = []

#     for fname in sorted(os.listdir(folder_path)):
#         if not fname.endswith(".wav"):
#             continue

#         try:
#             emb = extract_embedding(os.path.join(folder_path, fname))
#             speaker = "speaker_1" if "_1_" in fname else "speaker_0"
#             embeddings.append(emb)
#             labels.append(speaker)
#         except Exception as e:
#             print(f"⚠️ Failed on {fname}: {e}")

#     return np.array(embeddings), labels

# def visualize_tsne(embeddings, labels, title):
#     tsne = TSNE(n_components=2, random_state=42, perplexity=5)
#     emb_2d = tsne.fit_transform(embeddings)

#     label_colors = {"speaker_0": "blue", "speaker_1": "red"}

#     plt.figure(figsize=(8, 6))
#     for emb, lbl in zip(emb_2d, labels):
#         plt.scatter(emb[0], emb[1], c=label_colors[lbl], label=lbl, marker='o' if lbl=="speaker_0" else 'x')

#     # 去重 legend
#     handles, legend_labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(legend_labels, handles))
#     plt.legend(by_label.values(), by_label.keys())

#     plt.title(title)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # base 模型生成的音频路径
# base_dir = "/home/ssz/TTS/DailyTalk_old/output_base/result/DailyTalk/900000/23"
# # ours 模型生成的音频路径
# ours_dir = "./output_before_7_14/result/7.5/900000/23"

# # 处理 base
# base_embeddings, base_labels = load_embeddings_by_speaker(base_dir)
# visualize_tsne(base_embeddings, base_labels, title="🔵 Base 模型 - Speaker Embeddings (t-SNE)")

# # 处理 ours
# ours_embeddings, ours_labels = load_embeddings_by_speaker(ours_dir)
# visualize_tsne(ours_embeddings, ours_labels, title="🔴 Ours 模型 - Speaker Embeddings (t-SNE)")


'''
single comparison
'''
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from speechbrain.pretrained import EncoderClassifier
# from speechbrain.dataio.dataio import read_audio
# from tqdm import tqdm

# # 初始化模型
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir"
# )

# def extract_embedding(path):
#     wav = read_audio(path)
#     emb = classifier.encode_batch(wav.unsqueeze(0)).squeeze().cpu().numpy()
#     return emb

# def load_embeddings(folder_path, model_label):
#     embeddings = []
#     speakers = []
#     models = []

#     for fname in sorted(os.listdir(folder_path)):
#         if not fname.endswith(".wav"):
#             continue

#         try:
#             emb = extract_embedding(os.path.join(folder_path, fname))
#             speaker = "speaker_1" if "_1_" in fname else "speaker_0"
#             embeddings.append(emb)
#             speakers.append(speaker)
#             models.append(model_label)
#         except Exception as e:
#             print(f"⚠️ Failed on {fname}: {e}")

#     return embeddings, speakers, models

# # 模型目录
# base_dir = "/home/ssz/TTS/DailyTalk_old/output_base/result/DailyTalk/900000/126"
# ours_dir = "./output_before_7_14/result/FiLM/900000/126"

# # 提取 embedding
# base_embs, base_speakers, base_models = load_embeddings(base_dir, "base")
# ours_embs, ours_speakers, ours_models = load_embeddings(ours_dir, "ours")

# # 合并
# all_embeddings = np.array(base_embs + ours_embs)
# all_speakers = base_speakers + ours_speakers
# all_models = base_models + ours_models

# # t-SNE 降维
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# emb_2d = tsne.fit_transform(all_embeddings)

# # 可视化
# plt.figure(figsize=(10, 8))

# for emb, speaker, model in zip(emb_2d, all_speakers, all_models):
#     color = "blue" if speaker == "speaker_0" else "red"
#     marker = "o" if model == "base" else "x"
#     plt.scatter(emb[0], emb[1], c=color, marker=marker, label=f"{model}_{speaker}")

# # 去重 legend
# handles, labels = plt.gca().get_legend_handles_labels()
# unique = dict(zip(labels, handles))
# plt.legend(unique.values(), unique.keys())

# plt.title(" Speaker Embedding Comparison: Base vs Ours")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

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

def extract_embedding(path):
    wav = read_audio(path)
    emb = classifier.encode_batch(wav.unsqueeze(0)).squeeze().cpu().numpy()
    return emb

def average_embeddings_in_dialog(dialog_path):
    """
    读取一个对话文件夹，返回两个说话人的平均embedding
    返回字典：{'spk0': emb0, 'spk1': emb1}
    """
    spk0_embs, spk1_embs = [], []

    for fname in os.listdir(dialog_path):
        if not fname.endswith(".wav"):
            continue
        fpath = os.path.join(dialog_path, fname)
        try:
            emb = extract_embedding(fpath)
            if "_0_" in fname:
                spk0_embs.append(emb)
            elif "_1_" in fname:
                spk1_embs.append(emb)
        except Exception as e:
            print(f"❌ 处理 {fname} 失败: {e}")

    result = {}
    if spk0_embs:
        result['spk0'] = np.mean(spk0_embs, axis=0)
    if spk1_embs:
        result['spk1'] = np.mean(spk1_embs, axis=0)

    return result

def load_average_embeddings(root_dir, tag, dialog_ids):
    """
    从多个对话中提取每位 speaker 的平均 embedding
    返回 embeddings 和 对应的 labels
    """
    embeddings, labels = [], []
    for dialog_id in dialog_ids:
        dialog_path = os.path.join(root_dir, dialog_id)
        if not os.path.isdir(dialog_path):
            print(f"⚠️ 缺少对话文件夹: {dialog_path}")
            continue

        avg_embs = average_embeddings_in_dialog(dialog_path)
        for spk, emb in avg_embs.items():
            label = f"{tag}_{spk}_{dialog_id}"
            embeddings.append(emb)
            labels.append(label)
    return embeddings, labels

# ==== 你可以在这里修改对话编号 ====
# dialog_ids = ["23", "30", "59", "64"]  # 选取你想分析的多个对话段
dialog_ids = ["1112", "1126", "1238", "1298", "1325", "1452", "1898", "1908", "1936", "1987"]

# ==== 根路径设置 ====
base_root = "/home/ssz/TTS/DailyTalk_old/output_base/result/DailyTalk/900000"
ours_root = "./output_before_7_14/result/7.5/900000"

# ==== 提取数据 ====
print("📥 提取 base 平均 embedding ...")
base_embs, base_labels = load_average_embeddings(base_root, "base", dialog_ids)

print("📥 提取 ours 平均 embedding ...")
ours_embs, ours_labels = load_average_embeddings(ours_root, "ours", dialog_ids)

all_embeddings = np.array(base_embs + ours_embs)
all_labels = base_labels + ours_labels

# ==== t-SNE 降维 ====
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
emb_2d = tsne.fit_transform(all_embeddings)

# ==== 可视化 ====
plt.figure(figsize=(12, 10))

colors = {"spk0": "blue", "spk1": "red"}
markers = {"base": "o", "ours": "x"}

for emb, label in zip(emb_2d, all_labels):
    model, spk, dialog_id = label.split("_")
    color = colors[spk]
    marker = markers[model]
    plt.scatter(emb[0], emb[1], c=color, marker=marker,
                label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("🧭 Dialog-level Speaker Embedding Comparison (Base vs Ours)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
