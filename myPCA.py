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


    def normalize_data(self, features):
        # Separating out the features
        x = self.df.loc[:, features].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        principalComponents = self.pca.fit_transform(x)
        self.df = pd.DataFrame(data=principalComponents, columns=[features])
        print(f"The dataset after Normalize is:\n {len(self.df)}")
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
        pprint.pprint(f"The full dict: {full_dict}\n")
        print(f"Features priority list: {feat_lst}\n")
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
