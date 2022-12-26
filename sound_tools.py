from pydub import AudioSegment
from pydub.playback import play
from os import listdir
import random
import json
import pandas as pd
import numpy as np
import librosa
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt






def featurize(wavfile):
    #initialize features 
    hop_length = 512
    n_fft=2048
    #load file 
    y, sr = librosa.load(wavfile)
    #extract mfcc coefficients 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc) 
    #extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
    mfcc_features=np.array([np.mean(mfcc[0]),np.std(mfcc[0]),np.amin(mfcc[0]),np.amax(mfcc[0]),
                            np.mean(mfcc[1]),np.std(mfcc[1]),np.amin(mfcc[1]),np.amax(mfcc[1]),
                            np.mean(mfcc[2]),np.std(mfcc[2]),np.amin(mfcc[2]),np.amax(mfcc[2]),
                            np.mean(mfcc[3]),np.std(mfcc[3]),np.amin(mfcc[3]),np.amax(mfcc[3]),
                            np.mean(mfcc[4]),np.std(mfcc[4]),np.amin(mfcc[4]),np.amax(mfcc[4]),
                            np.mean(mfcc[5]),np.std(mfcc[5]),np.amin(mfcc[5]),np.amax(mfcc[5]),
                            np.mean(mfcc[6]),np.std(mfcc[6]),np.amin(mfcc[6]),np.amax(mfcc[6]),
                            np.mean(mfcc[7]),np.std(mfcc[7]),np.amin(mfcc[7]),np.amax(mfcc[7]),
                            np.mean(mfcc[8]),np.std(mfcc[8]),np.amin(mfcc[8]),np.amax(mfcc[8]),
                            np.mean(mfcc[9]),np.std(mfcc[9]),np.amin(mfcc[9]),np.amax(mfcc[9]),
                            np.mean(mfcc[10]),np.std(mfcc[10]),np.amin(mfcc[10]),np.amax(mfcc[10]),
                            np.mean(mfcc[11]),np.std(mfcc[11]),np.amin(mfcc[11]),np.amax(mfcc[11]),
                            np.mean(mfcc[12]),np.std(mfcc[12]),np.amin(mfcc[12]),np.amax(mfcc[12]),
                            np.mean(mfcc_delta[0]),np.std(mfcc_delta[0]),np.amin(mfcc_delta[0]),np.amax(mfcc_delta[0]),
                            np.mean(mfcc_delta[1]),np.std(mfcc_delta[1]),np.amin(mfcc_delta[1]),np.amax(mfcc_delta[1]),
                            np.mean(mfcc_delta[2]),np.std(mfcc_delta[2]),np.amin(mfcc_delta[2]),np.amax(mfcc_delta[2]),
                            np.mean(mfcc_delta[3]),np.std(mfcc_delta[3]),np.amin(mfcc_delta[3]),np.amax(mfcc_delta[3]),
                            np.mean(mfcc_delta[4]),np.std(mfcc_delta[4]),np.amin(mfcc_delta[4]),np.amax(mfcc_delta[4]),
                            np.mean(mfcc_delta[5]),np.std(mfcc_delta[5]),np.amin(mfcc_delta[5]),np.amax(mfcc_delta[5]),
                            np.mean(mfcc_delta[6]),np.std(mfcc_delta[6]),np.amin(mfcc_delta[6]),np.amax(mfcc_delta[6]),
                            np.mean(mfcc_delta[7]),np.std(mfcc_delta[7]),np.amin(mfcc_delta[7]),np.amax(mfcc_delta[7]),
                            np.mean(mfcc_delta[8]),np.std(mfcc_delta[8]),np.amin(mfcc_delta[8]),np.amax(mfcc_delta[8]),
                            np.mean(mfcc_delta[9]),np.std(mfcc_delta[9]),np.amin(mfcc_delta[9]),np.amax(mfcc_delta[9]),
                            np.mean(mfcc_delta[10]),np.std(mfcc_delta[10]),np.amin(mfcc_delta[10]),np.amax(mfcc_delta[10]),
                            np.mean(mfcc_delta[11]),np.std(mfcc_delta[11]),np.amin(mfcc_delta[11]),np.amax(mfcc_delta[11]),
                            np.mean(mfcc_delta[12]),np.std(mfcc_delta[12]),np.amin(mfcc_delta[12]),np.amax(mfcc_delta[12])])
    
    return mfcc_features


def add_512_features():
    print("Start")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    MALES_PATH = r"data\males"
    FEMALES_PATH = r"data\females"
    MALES_OUT_PATH = r"data\males_out"
    FEMALES_OUT_PATH = r"data\females_out"
    male_files = listdir(MALES_PATH)
    female_files = listdir(FEMALES_PATH)
    random.shuffle(male_files)
    random.shuffle(female_files)
    min_amount = 99999
    male = []
    female = []
    count = 0
    for file in male_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = featurize(f"{MALES_PATH}/{file}")
            signal, fs =torchaudio.load(f"{MALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            male.append(features.tolist() + embedding.tolist())
            print(file)
            sound = AudioSegment.from_wav(f"{MALES_PATH}/{file}")
            sound.export(f"{MALES_OUT_PATH}/{file}", format='wav')
            count += 1
    
    count = 0
    for file in female_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = featurize(f"{FEMALES_PATH}/{file}")
            signal, fs =torchaudio.load(f"{FEMALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            female.append(features.tolist() + embedding.tolist())
            sound = AudioSegment.from_wav(f"{FEMALES_PATH}/{file}")
            sound.export(f"{FEMALES_OUT_PATH}/{file}", format='wav')
            count += 1
    # mData = np.array(boys)
    # fData = np.array(girls)
    # # Perform PCA
    # pca = PCA(n_components=1)
    # pca.fit(mData)
    # transformed_data_males = pca.transform(mData)
    #
    # pca.fit(fData)
    # transformed_data_females = pca.transform(fData)

    print("m: ", len(male))
    print("f: ", len(female))

    json_obj = {"males": male, "females": female}
    with open('data/m_f_audio.json', 'w') as outfile:
        json.dump(json_obj, outfile)

if __name__ == '__main__':
    add_512_features()
