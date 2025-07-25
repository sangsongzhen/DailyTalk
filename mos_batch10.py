# import torch
# import librosa
# import os
# import random


# def load_audio(path):
#     wav, sr = librosa.load(path, sr=None, mono=None)
#     return wav, sr


# def evaluate(model, wav, sr):
#     with torch.no_grad():
#         score = model(torch.from_numpy(wav).unsqueeze(0), sr)
#     return score.item()


# def collect_wav_paths(root_dir):
#     wav_paths = []
#     for root, _, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith(".wav"):
#                 wav_paths.append(os.path.join(root, file))
#     return wav_paths


# if __name__ == '__main__':
#     print("------- Loading predictor -------")
#     predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

#     # 顶层目录
#     base_dir = "./output/result/7.5/900000"

#     # 获取所有子目录（非递归，只获取一级子目录）
#     subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
#                if os.path.isdir(os.path.join(base_dir, d))]

#     # 随机选取 10 个子目录（或最多 10 个）
#     sampled_subdirs = random.sample(subdirs, min(10, len(subdirs)))
#     print(f"🎯 Sampled {len(sampled_subdirs)} subdirectories.")

#     total_scores = []
#     for subdir in sampled_subdirs:
#         wav_files = collect_wav_paths(subdir)
#         print(f"\n📂 Processing '{subdir}' ({len(wav_files)} files)")
#         subdir_scores = []
#         for wav_path in wav_files:
#             try:
#                 wav, sr = load_audio(wav_path)
#                 score = evaluate(predictor, wav, sr)
#                 subdir_scores.append(score)
#                 print(f"  [{os.path.basename(wav_path)}] MOS: {score:.3f}")
#             except Exception as e:
#                 print(f"  ❌ Error processing {wav_path}: {e}")
#         if subdir_scores:
#             avg_subdir_score = sum(subdir_scores) / len(subdir_scores)
#             total_scores.append(avg_subdir_score)
#             print(f"✅ Avg MOS for {os.path.basename(subdir)}: {avg_subdir_score:.3f}")
#         else:
#             print("⚠️ No valid .wav files found in this folder.")

#     # 全部子目录的平均值
#     if total_scores:
#         overall_avg = sum(total_scores) / len(total_scores)
#         print(f"\n🎉 Overall Average MOS across {len(total_scores)} folders: {overall_avg:.3f}")
#     else:
#         print("\n❗ No valid data found.")


import torch
import librosa
import os


def load_audio(path):
    wav, sr = librosa.load(path, sr=None, mono=None)
    return wav, sr


def evaluate(model, wav, sr):
    with torch.no_grad():
        score = model(torch.from_numpy(wav).unsqueeze(0), sr)
    return score.item()


def collect_wav_paths(root_dir):
    wav_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(root, file))
    return wav_paths


if __name__ == '__main__':
    print("------- Loading predictor -------")
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    # 顶层目录
    base_dir = "./output/result/DailyTalk/900000"
    # base_dir = "./raw_data/DailyTalk/data"


    # ❗ 手动指定你要评估的子目录（10个）
    # target_subdirs = ["23", "403", "590", "877", "1091", "1298", "1618", "1829", "1949", "2463"]
    target_subdirs = ["1112", "1126", "1238", "1298", "1325", "1452", "1898", "1908", "1936", "1987"]

    total_scores = []
    for subdir_name in target_subdirs:
        subdir = os.path.join(base_dir, subdir_name)
        if not os.path.exists(subdir):
            print(f"❌ Folder not found: {subdir}")
            continue

        wav_files = collect_wav_paths(subdir)
        print(f"\n📂 Processing '{subdir}' ({len(wav_files)} files)")
        subdir_scores = []
        for wav_path in wav_files:
            try:
                wav, sr = load_audio(wav_path)
                score = evaluate(predictor, wav, sr)
                subdir_scores.append(score)
                print(f"  [{os.path.basename(wav_path)}] MOS: {score:.3f}")
            except Exception as e:
                print(f"  ❌ Error processing {wav_path}: {e}")
        if subdir_scores:
            avg_subdir_score = sum(subdir_scores) / len(subdir_scores)
            total_scores.append(avg_subdir_score)
            print(f"✅ Avg MOS for {subdir_name}: {avg_subdir_score:.3f}")
        else:
            print("⚠️ No valid .wav files found in this folder.")

    if total_scores:
        overall_avg = sum(total_scores) / len(total_scores)
        print(f"\n🎯 Overall Average MOS across {len(total_scores)} folders: {overall_avg:.3f}")
    else:
        print("\n❗ No valid data found.")
