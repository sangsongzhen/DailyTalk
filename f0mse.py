# evaluate_f0_mse.py

import os
import numpy as np
import librosa
import pyworld as pw
from fastdtw import fastdtw


def extract_f0(y, sr):
    _f0, _ = pw.harvest(y, sr)
    return _f0


def compute_f0_mse(f0_1, f0_2):
    distance, path = fastdtw(f0_1, f0_2, dist=lambda x, y: abs(x - y))
    aligned1 = np.array([f0_1[i] for i, _ in path])
    aligned2 = np.array([f0_2[j] for _, j in path])

    mask = (aligned1 > 0) & (aligned2 > 0)
    if np.sum(mask) == 0:
        return None
    mse = np.mean((aligned1[mask] - aligned2[mask]) ** 2)
    return mse


def evaluate_f0_mse(pred_path, ref_path):
    y_pred, sr = librosa.load(pred_path, sr=None)
    y_ref, _ = librosa.load(ref_path, sr=sr)

    y_pred = y_pred.astype(np.float64)
    y_ref = y_ref.astype(np.float64)

    f0_pred = extract_f0(y_pred, sr)
    f0_ref = extract_f0(y_ref, sr)
    return compute_f0_mse(f0_pred, f0_ref)


if __name__ == '__main__':
    pred_root = "./output/result/7.4/900000/23"
    ref_root = "/home/ssz/TTS/speech-datasets/dailytalk/data/23"

    f0_mse_scores = []

    for root, _, files in os.walk(pred_root):
        for file in files:
            if file.endswith(".wav"):
                rel_path = os.path.relpath(os.path.join(root, file), pred_root)
                pred_path = os.path.join(pred_root, rel_path)
                ref_path = os.path.join(ref_root, rel_path)

                if os.path.exists(ref_path):
                    try:
                        f0_mse = evaluate_f0_mse(pred_path, ref_path)
                        if f0_mse is not None:
                            print(f"{rel_path}: F0 MSE = {f0_mse:.3f}")
                            f0_mse_scores.append(f0_mse)
                        else:
                            print(f"{rel_path}: skipped (voiceless regions only)")
                    except Exception as e:
                        print(f"Error processing {rel_path}: {e}")
                else:
                    print(f"Missing reference file: {ref_path}")

    print("\n✅ F0 MSE Average:", np.mean(f0_mse_scores) if f0_mse_scores else "N/A")
