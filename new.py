import json

import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import os, random

path_of_the_directory = 'data/males'
print("Files and directories in a specified path:")
dirlist = (os.listdir(path_of_the_directory))
features_list = list()
jsonfilename = 'my_json_audio.json'
one = list()
d = {}
x = ''

#
# for filename in dirlist:
#     try:
#         if filename[-5:] == '.json':
#             print(filename)
#             features = json.load(open("data/females/"+filename))['features']
#             # print(features)
#             one.append(features)
#     except:
#         pass
#
#
#
# print(one)

for filename in dirlist:
    if filename[-4:] != '.wav':
        continue
    print(filename)
    f = os.path.join(path_of_the_directory, filename)
    if os.path.isfile(f):
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                    savedir="pretrained_models/spkrec-xvect-voxceleb")
        print('data/males/' + filename)
        signal, fs = torchaudio.load('data/males/' + filename)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.detach().cpu().numpy()
        embedding = embeddings[0][0]
        # x += embedding
        print("finish embedding ", filename)
        # for j in range(len(dirlist)):
        try:
            file = filename
            if file[-4:] != '.wav':
                continue
            if file[-4:] == '.m4a':
                os.system('ffmpeg -i %s %s' % (file, file[0:-4] + '.wav'))
                os.remove(file)
                file = filename[0:-4] + '.wav'

            # if file[-4:] == '.wav' not in dirlist and os.path.getsize(file) > 500:
            try:
                # get wavefile
                wavfile = file
                # print('%s - featurizing %s' % (name.upper(), wavfile))
                # obtain features
                # features = np.append(featurize(wavfile), audio_time_features(wavfile))
                # print(features)
                # append to list
                # one.append(features.tolist())
                features_list.append(embedding.tolist())
                print("features_list.append ", file)
                print(embedding.tolist())
                # save intermediate .json just in case
                data = {
                    'features': embedding.tolist(),
                }
                jsonfile = open(filename[0:-4] + '.json', 'w')
                json.dump(data, jsonfile)
                jsonfile.close()
            except:
                print('error')
            else:
                pass
        except:
            pass
j = open('jason.json', 'w')
json.dump(d, j)
j.close()
print("start feature_list2")
# feature_list2 = list()
# feature_lengths = list()
# for i in range(len(features_list)):
#     one = features_list[i]
#     random.shuffle(one)
#     feature_list2.append(one)
#     feature_lengths.append(len(one))
#
# min_num = np.amin(feature_lengths)
# # make sure they are the same length (For later) - this avoid errors
# while min_num * len(path_of_the_directory) != np.sum(feature_lengths):
#     for i in range(len(path_of_the_directory)):
#         while len(feature_list2[i]) > min_num:
#             print('%s is %s more than %s, balancing...' % (
#                 path_of_the_directory[i].upper(), str(len(feature_list2[i]) - int(min_num)), 'min value'))
#             feature_list2[i].pop()
#     feature_lengths = list()
#     for i in range(len(feature_list2)):
#         one = feature_list2[i]
#         feature_lengths.append(len(one))
#
# # now write to json
# data = {}
# for i in range(len(path_of_the_directory)):
#     data.update({path_of_the_directory[i]: feature_list2[i]})
#
# jsonfile = open(jsonfilename, 'w')
# json.dump(data, jsonfile)
# jsonfile.close()
# print(embedding)
