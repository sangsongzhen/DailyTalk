import torch
import librosa
import os



def load_audio(path):
    wav, sr= librosa.load(path, sr=None, mono=None)
    return wav, sr

def evalutate(model, wav, sr):
    score = model(torch.from_numpy(wav).unsqueeze(0), sr)
    return score

if __name__ == '__main__':
    print("------- evaluating -------")
    # print("------- Loading predictor -------")
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    # print("--------Loaded ---------------------")
    # file path
    file_path = "./output/result/7.4/900000"
    path = os.path.join(file_path, "23/11_0_d23.wav")

    wav, sr = load_audio(path)

    score = evalutate(predictor, wav, sr)
    print(score)

sampled_subdirs = ["23", "280", "590", "877", "1091", "1251", "1618", "1829", "1987", "2463"]



