# import os
# import torch
# import torchaudio
# import numpy as np
# from tqdm import tqdm
# from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# # === 配置路径 ===
# wav_root = "/home/ssz/TTS/speech-datasets/dailytalk/data"  # 修改为你的实际路径
# output_emb_dir = "./preprocessed_data/DailyTalk/audio_emb"
# os.makedirs(output_emb_dir, exist_ok=True)

# # === 加载 feature extractor 和模型 ===
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
# model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
# model.eval()
# if torch.cuda.is_available():
#     model.cuda()

# # === 收集需要处理的 wav 文件 ===
# wav_files = []
# for speaker in sorted(os.listdir(wav_root)):
#     speaker_path = os.path.join(wav_root, speaker)
#     if not os.path.isdir(speaker_path):
#         continue
#     for file in os.listdir(speaker_path):
#         if file.endswith(".wav"):
#             basename = file[:-4]
#             output_filename = f"{speaker}-audio_emb-{basename}.npy"
#             output_path = os.path.join(output_emb_dir, output_filename)
#             if not os.path.exists(output_path):  # 跳过已处理文件
#                 wav_files.append((os.path.join(speaker_path, file), speaker, basename))

# print(f"待处理语音数量：{len(wav_files)}")

# # === 提取音频 embedding ===
# for wav_path, speaker, basename in tqdm(wav_files):
#     waveform, sr = torchaudio.load(wav_path)
#     if sr != 16000:
#         waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

#     inputs = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
#     if torch.cuda.is_available():
#         inputs = {k: v.cuda() for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
#     hidden_states = outputs.last_hidden_state[0]  # [T, 768]
#     audio_emb = hidden_states.mean(dim=0).cpu().numpy()  # [768]

#     # 保存 embedding
#     output_filename = f"{speaker}-audio_emb-{basename}.npy"
#     np.save(os.path.join(output_emb_dir, output_filename), audio_emb)

import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# === 配置路径 ===
wav_root = "/home/ssz/TTS/speech-datasets/dailytalk/data"
output_emb_dir = "./preprocessed_data/DailyTalk/audio_emb"
os.makedirs(output_emb_dir, exist_ok=True)

# === 加载模型 ===
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
model.eval()
if torch.cuda.is_available():
    model.cuda()

# === 收集 wav 文件 ===
wav_files = []
for speaker in sorted(os.listdir(wav_root)):
    speaker_path = os.path.join(wav_root, speaker)
    if not os.path.isdir(speaker_path):
        continue
    for file in os.listdir(speaker_path):
        if file.endswith(".wav"):
            basename = file[:-4]  # 去掉 .wav
            true_speaker = basename.split("_")[1]  # 从 basename 中提取 speaker_id
            output_filename = f"{true_speaker}-audio_emb-{basename}.npy"
            output_path = os.path.join(output_emb_dir, output_filename)
            if not os.path.exists(output_path):
                wav_files.append((os.path.join(speaker_path, file), true_speaker, basename))

print(f"待处理语音数量：{len(wav_files)}")

# === 提取 embedding ===
for wav_path, speaker_id, basename in tqdm(wav_files):
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

    inputs = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state[0]  # [T, 768]
    audio_emb = hidden_states.mean(dim=0).cpu().numpy()

    output_filename = f"{speaker_id}-audio_emb-{basename}.npy"
    np.save(os.path.join(output_emb_dir, output_filename), audio_emb)

