import json
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class myPCA:
    def __init__(self, df, n):
        self.pca = PCA(n_components=n)
        self.df = df

    def plot_data(self, num, acc):
        plt.scatter(num, acc)
        plt.xlabel('Features Number')
        plt.ylabel('Accuracy %')
        plt.show()

    def normalize_data(self, features):
        # Separating out the features
        x = self.df.loc[:, features].values
        print(f"The number of features before Normalize is: {len(x[0])}")
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        principalComponents = self.pca.fit_transform(x)
        self.df = pd.DataFrame(data=principalComponents, columns=[features])
        return self.df.loc[:, features].values

    def decide_args(self, accuracy):
        print("Entered decide_args")
        covMat = np.cov(self.df)
        feat_lst = self.pca.explained_variance_ratio_
        eigenvalues, eigenvectors = np.linalg.eig(covMat)
        eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        full_dict = {}
        for i in eigen_pairs:
            full_dict.update({i[0]: i[1]})
        x = []
        y = []
        counter = 0
        sum = 0
        for f in feat_lst:
            sum = sum + f
            counter = counter + 1
            x.append(counter)
            y.append(sum)
            if sum > accuracy:
                break
        print(f"In order to get {accuracy * 100}%, you need {counter} features\n")
        return x, y

if __name__ == '__main__':
    precent = 0.01*float(input('What percentage of accuracy would you like to get?'))
    while precent<0 or precent>1:
        precent = 0.01 * float(input('input a number between 0 to 100!'))
    # Opening JSON file
    with open('m_f_toPCA.json', 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)

    males= json_object['males'];
    females= json_object['females'];
    f = []
    for i in range(616):
        f.append(f"feacher {i + 1}")
    print("....................................................")

    # pca algo for males
    print("PCA resuts for males:")
    data = pd.DataFrame(males, columns=f)
    print(f'Number of sampels for males: {len(data)}')
    malesppca = myPCA(data, 616)
    df = malesppca.normalize_data(f)
    x, y = malesppca.decide_args(precent)
    print("....................................................")
    malesppca.plot_data(x,y)
    fnum = len(x)

    # pca algo for females
    print("PCA resuts for females:")
    data = pd.DataFrame(females, columns=f)
    print(f'Number of sampels for females: {len(data)}')
    femalesppca = myPCA(data, 616)
    df = femalesppca.normalize_data(f)
    x, y = femalesppca.decide_args(precent)
    print("....................................................")
    femalesppca.plot_data(x, y)

    if len (x)>fnum:
        fnum = len(x)
    print(f"Number of features after Normalize: {fnum}")

   #create lists for the features after running the PCA algorithm.
    normalmales = []
    for i in df:
        normalmales.append(i[:fnum].tolist())
    normalfemales = []
    for i in df:
        normalfemales.append(i[:fnum].tolist())

    json_obj = {"features":fnum, "males": normalmales, "females": normalfemales}
    with open('m_f_fromPCA.json', 'w') as outfile:
        json.dump(json_obj, outfile)