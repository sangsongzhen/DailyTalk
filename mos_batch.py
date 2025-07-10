import torch
import librosa
import os


def load_audio(path):
    wav, sr = librosa.load(path, sr=None, mono=None)
    return wav, sr


def evaluate(model, wav, sr):
    with torch.no_grad():
        score = model(torch.from_numpy(wav).unsqueeze(0), sr)
    return score.item()  # 转为 float


def collect_wav_paths(root_dir):
    """递归获取所有子目录下的.wav文件路径"""
    wav_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_paths.append(os.path.join(root, file))
    return wav_paths


if __name__ == '__main__':
    print("------- Loading predictor -------")
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    # 文件夹路径
    file_path = "./output/result/7.5/900000/155"

    # 遍历所有 .wav 文件
    wav_files = collect_wav_paths(file_path)
    print(f"Found {len(wav_files)} .wav files.")

    scores = []
    for wav_path in wav_files:
        try:
            wav, sr = load_audio(wav_path)
            score = evaluate(predictor, wav, sr)
            scores.append(score)
            print(f"[{wav_path}] MOS: {score:.3f}")
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n✅ Average MOS Score: {avg_score:.3f}")
    else:
        print("⚠️ No valid wav files were processed.")
