'''
================================================ 
##            VOICEBOOK REPOSITORY            ##      
================================================ 

repository name: voicebook 
repository version: 1.0 
repository link: https://github.com/jim-schwoebel/voicebook 
author: Jim Schwoebel 
author contact: js@neurolex.co 
description: a book and repo to get you started programming voice applications in Python - 10 chapters and 200+ scripts. 
license category: opensource 
license: Apache 2.0 license 
organization name: NeuroLex Laboratories, Inc. 
location: Seattle, WA 
website: https://neurolex.ai 
release date: 2018-09-28 

This code (voicebook) is hereby released under a Apache 2.0 license license. 

For more information, check out the license terms below. 

================================================ 
##               LICENSE TERMS                ##      
================================================ 

Copyright 2018 NeuroLex Laboratories, Inc. 

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

     http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

================================================ 
##               SERVICE STATEMENT            ##        
================================================ 

If you are using the code written for a larger project, we are 
happy to consult with you and help you with deployment. Our team 
has >10 world experts in Kafka distributed architectures, microservices 
built on top of Node.js / Python / Docker, and applying machine learning to 
model speech and text data. 

We have helped a wide variety of enterprises - small businesses, 
researchers, enterprises, and/or independent developers. 

If you would like to work with us let us know @ js@neurolex.co. 

================================================ 
##            LOAD_AUDIOCLASSIFY.PY           ##    
================================================ 

Fingerprint audio models in a streaming folder. 
'''
import sys

import librosa, pickle, uuid
from pydub import AudioSegment
import os, json
import numpy as np

from dvir_record_funcs import record_to_file

cur_dir = os.getcwd() + '/load_dir'
model_dir = os.getcwd() + '/models'
load_dir = os.getcwd() + '/load_dir'
final = list()

def featurize2(wavfile):
    # initialize features
    hop_length = 512
    n_fft = 2048
    # load file
    y, sr = librosa.load(wavfile)
    # extract mfcc coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    # extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
    mfcc_features = np.array([np.mean(mfcc[0]), np.std(mfcc[0]), np.amin(mfcc[0]), np.amax(mfcc[0]),
                              np.mean(mfcc[1]), np.std(mfcc[1]), np.amin(mfcc[1]), np.amax(mfcc[1]),
                              np.mean(mfcc[2]), np.std(mfcc[2]), np.amin(mfcc[2]), np.amax(mfcc[2]),
                              np.mean(mfcc[3]), np.std(mfcc[3]), np.amin(mfcc[3]), np.amax(mfcc[3]),
                              np.mean(mfcc[4]), np.std(mfcc[4]), np.amin(mfcc[4]), np.amax(mfcc[4]),
                              np.mean(mfcc[5]), np.std(mfcc[5]), np.amin(mfcc[5]), np.amax(mfcc[5]),
                              np.mean(mfcc[6]), np.std(mfcc[6]), np.amin(mfcc[6]), np.amax(mfcc[6]),
                              np.mean(mfcc[7]), np.std(mfcc[7]), np.amin(mfcc[7]), np.amax(mfcc[7]),
                              np.mean(mfcc[8]), np.std(mfcc[8]), np.amin(mfcc[8]), np.amax(mfcc[8]),
                              np.mean(mfcc[9]), np.std(mfcc[9]), np.amin(mfcc[9]), np.amax(mfcc[9]),
                              np.mean(mfcc[10]), np.std(mfcc[10]), np.amin(mfcc[10]), np.amax(mfcc[10]),
                              np.mean(mfcc[11]), np.std(mfcc[11]), np.amin(mfcc[11]), np.amax(mfcc[11]),
                              np.mean(mfcc[12]), np.std(mfcc[12]), np.amin(mfcc[12]), np.amax(mfcc[12]),
                              np.mean(mfcc_delta[0]), np.std(mfcc_delta[0]), np.amin(mfcc_delta[0]),
                              np.amax(mfcc_delta[0]),
                              np.mean(mfcc_delta[1]), np.std(mfcc_delta[1]), np.amin(mfcc_delta[1]),
                              np.amax(mfcc_delta[1]),
                              np.mean(mfcc_delta[2]), np.std(mfcc_delta[2]), np.amin(mfcc_delta[2]),
                              np.amax(mfcc_delta[2]),
                              np.mean(mfcc_delta[3]), np.std(mfcc_delta[3]), np.amin(mfcc_delta[3]),
                              np.amax(mfcc_delta[3]),
                              np.mean(mfcc_delta[4]), np.std(mfcc_delta[4]), np.amin(mfcc_delta[4]),
                              np.amax(mfcc_delta[4]),
                              np.mean(mfcc_delta[5]), np.std(mfcc_delta[5]), np.amin(mfcc_delta[5]),
                              np.amax(mfcc_delta[5]),
                              np.mean(mfcc_delta[6]), np.std(mfcc_delta[6]), np.amin(mfcc_delta[6]),
                              np.amax(mfcc_delta[6]),
                              np.mean(mfcc_delta[7]), np.std(mfcc_delta[7]), np.amin(mfcc_delta[7]),
                              np.amax(mfcc_delta[7]),
                              np.mean(mfcc_delta[8]), np.std(mfcc_delta[8]), np.amin(mfcc_delta[8]),
                              np.amax(mfcc_delta[8]),
                              np.mean(mfcc_delta[9]), np.std(mfcc_delta[9]), np.amin(mfcc_delta[9]),
                              np.amax(mfcc_delta[9]),
                              np.mean(mfcc_delta[10]), np.std(mfcc_delta[10]), np.amin(mfcc_delta[10]),
                              np.amax(mfcc_delta[10]),
                              np.mean(mfcc_delta[11]), np.std(mfcc_delta[11]), np.amin(mfcc_delta[11]),
                              np.amax(mfcc_delta[11]),
                              np.mean(mfcc_delta[12]), np.std(mfcc_delta[12]), np.amin(mfcc_delta[12]),
                              np.amax(mfcc_delta[12])])

    return mfcc_features


def exportfile(newAudio, time1, time2, filename, i):
    # Exports to a wav file in the current path.
    newAudio2 = newAudio[time1:time2]
    g = os.listdir()
    if filename[0:-4] + '_' + str(i) + '.wav' in g:
        filename2 = str(uuid.uuid4()) + '_segment' + '.wav'
        print('making %s' % (filename2))
        newAudio2.export(filename2, format="wav")
    else:
        filename2 = str(uuid.uuid4()) + '.wav'
        print('making %s' % (filename2))
        newAudio2.export(filename2, format="wav")

    return filename2


def audio_time_features(filename):
    # recommend >0.50 seconds for timesplit
    timesplit = 0.50
    hop_length = 512
    n_fft = 2048

    y, sr = librosa.load(filename)
    duration = float(librosa.core.get_duration(y=y))

    # Now splice an audio signal into individual elements of 100 ms and extract
    # all these features per 100 ms
    segnum = round(duration / timesplit)
    deltat = duration / segnum
    timesegment = list()
    time = 0

    for i in range(segnum):
        # milliseconds
        timesegment.append(time)
        time = time + deltat * 1000

    newAudio = AudioSegment.from_wav(filename)
    filelist = list()

    for i in range(len(timesegment) - 1):
        filename = exportfile(newAudio, timesegment[i], timesegment[i + 1], filename, i)
        filelist.append(filename)

        featureslist = np.array([0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0])

    # save 100 ms segments in current folder (delete them after)
    for j in range(len(filelist)):
        try:
            features = featurize2(filelist[i])
            featureslist = featureslist + features
            os.remove(filelist[j])
        except:
            print('error splicing')
            os.remove(filelist[j])

    # now scale the featureslist array by the length to get mean in each category
    featureslist = featureslist / segnum

    return featureslist


def featurize(wavfile):
    features = np.append(featurize2(wavfile), audio_time_features(wavfile))
    return features


def check_name(name):
    for c in name:
        if not c.isalpha() and not c.isalnum():
            return False
    return True


def convert(file):
    # print(file[-4:])
    if file[-4:] != '.wav':
        filename = file[0:-4] + '.wav'
        os.system('ffmpeg -i %s %s' % (file, filename))
        os.remove(file)
    elif file[-4:] == '.wav':
        filename = file

    return filename


model_list = list()
os.chdir(model_dir)
listdir = os.listdir()

for i in range(len(listdir)):
    if listdir[i][-12:] == 'audio.pickle':
        model_list.append(listdir[i])

count = 0
errorcount = 0

try:
    os.chdir(load_dir)
except:
    os.mkdir(load_dir)
    os.chdir(load_dir)

listdir = os.listdir()

ans = input('Do you want to add a record to load_dir? (yes/no)')
while ans != 'no':
    filename = input("Input name for the record (letters and numbers only)")
    while not check_name(filename):
        filename = input("Invalid name!\nInput name again (letters and numbers only)")
    while filename + '.wav' in listdir:
        filename = input("Name exist in load_dir!\nInput name again (letters and numbers only)")
    filename += '.wav'

    print(f"Recording to {filename}:")
    # record the file (start talking)
    if record_to_file(filename):
        listdir = os.listdir()
        print(f'{filename} saved.')
    ans = input('do you want to add more record to load_dir? (yes/no)')

listdir = os.listdir()
for i in range(len(listdir)):
    try:
        if listdir[i][-5:] not in ['Store', '.json']:
            if listdir[i][-4:] != '.wav':
                if listdir[i][-5:] != '.json':
                    os.chmod(listdir[i], 0o644)
                    filename = convert(listdir[i])
            else:
                filename = listdir[i]

            if filename[0:-4] + '.json' not in listdir:
                print(f"\nResults for {filename}:")
                features = featurize(filename)
                features = features.reshape(1, -1)

                os.chdir(model_dir)

                class_list = list()
                model_acc = list()
                deviations = list()
                modeltypes = list()

                for j in range(len(model_list)):
                    modelname = model_list[j]
                    i1 = modelname.find('_')
                    name1 = modelname[0:i1]
                    i2 = modelname[i1:]
                    i3 = i2.find('_')
                    name2 = i2[0:i3]

                    loadmodel = open(modelname, 'rb')
                    model = pickle.load(loadmodel)
                    loadmodel.close()
                    output = str(model.predict(features)[0])
                    print(f'Gender detect: {output}')
                    final.append(output)
                    classname = output
                    class_list.append(classname)

                    g = json.load(open(modelname[0:-7] + '.json'))
                    model_acc.append(g['accuracy'])
                    deviations.append(g['deviation'])
                    modeltypes.append(g['modeltype'])

                os.chdir(load_dir)

                jsonfilename = filename[0:-4] + '.json'
                jsonfile = open(jsonfilename, 'w')
                data = {
                    'filename': filename,
                    'filetype': 'audio file',
                    'class': class_list,
                    'model': model_list,
                    'model accuracies': model_acc,
                    'model deviations': deviations,
                    'model types': modeltypes,
                    'features': features.tolist(),
                    'count': count,
                    'errorcount': errorcount,
                }
                json.dump(data, jsonfile)
                jsonfile.close()
            else:
                print(f"{filename} has already check.")
            count = count + 1
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        errorcount = errorcount + 1
        count = count + 1
print("Gender detect list:")
print(final)
