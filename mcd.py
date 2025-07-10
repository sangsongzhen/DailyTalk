
import os
import numpy as np
import librosa
import pyworld as pw
from librosa.sequence import dtw


def extract_mcep(y, sr, dim=13):
    # 提取 F0、时长
    _f0, t = pw.harvest(y.astype(np.float64), sr)
    sp = pw.cheaptrick(y.astype(np.float64), _f0, t, sr)
    mcep = pw.code_spectral_envelope(sp, sr, dim + 1)  # +1 是因为我们要去掉第0维
    return mcep[:, 1:]  # 去掉第0维（能量），只保留 1~13 维


def compute_mcd_librosa(mcep1, mcep2):
    D, wp = dtw(X=mcep1.T, Y=mcep2.T, metric='euclidean')
    aligned1 = mcep1[wp[:, 0]]
    aligned2 = mcep2[wp[:, 1]]
    diff = aligned1 - aligned2
    mse = np.mean(np.sum(diff ** 2, axis=1))
    mcd = (10.0 / np.log(10)) * np.sqrt(2 * mse)
    return mcd


def evaluate_mcd(pred_path, ref_path):
    y_pred, sr = librosa.load(pred_path, sr=None)
    y_ref, _ = librosa.load(ref_path, sr=sr)
    mcep_pred = extract_mcep(y_pred, sr)
    mcep_ref = extract_mcep(y_ref, sr)
    return compute_mcd_librosa(mcep_pred, mcep_ref)


if __name__ == '__main__':
    pred_root = "./output/result/7.4/900000/23"
    ref_root = "/home/ssz/TTS/speech-datasets/dailytalk/data/23"

    mcd_scores = []

    for root, _, files in os.walk(pred_root):
        for file in files:
            if file.endswith(".wav"):
                rel_path = os.path.relpath(os.path.join(root, file), pred_root)
                pred_path = os.path.join(pred_root, rel_path)
                ref_path = os.path.join(ref_root, rel_path)

                if os.path.exists(ref_path):
                    try:
                        mcd = evaluate_mcd(pred_path, ref_path)
                        print(f"{rel_path}: MCD = {mcd:.3f}")
                        mcd_scores.append(mcd)
                    except Exception as e:
                        print(f"❌ Error processing {rel_path}: {e}")
                else:
                    print(f"⚠️ Missing reference file: {ref_path}")

    print("\n✅ MCD Average:", np.mean(mcd_scores) if mcd_scores else "N/A")
