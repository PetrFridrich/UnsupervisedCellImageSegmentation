import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.metrics import jaccard_score

from sklearn.cluster import MiniBatchKMeans, KMeans, Birch, BisectingKMeans
from sklearn.mixture import GaussianMixture


class Clusterer():

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

        self.algorithms = [KMeans(n_clusters=self.n_clusters), 
                           MiniBatchKMeans(n_clusters=self.n_clusters),
                           BisectingKMeans(n_clusters=self.n_clusters),
                           Birch(n_clusters=self.n_clusters),
                           GaussianMixture(n_components=self.n_clusters)]

        return None
    

    def fit_predict(self, X):

        y = self.apply_algorithms(X)
        y = self.relabel(X,y)
        y = self.majority_voting(y)

        return y
    

    def apply_algorithms(self, X):
        
        y = np.zeros((len(X), len(self.algorithms)))

        for i, model in enumerate(self.algorithms):

            y[:, i] = model.fit_predict(X)

        return y
    

    def relabel(self, X, y):

        relabeled_y = np.zeros_like(y)
        

        for i in range(len(self.algorithms)):

            means = self.calculate_means(X,y[:, i])
            ordered_means = np.argsort(means)[::-1]

            for index, value in enumerate(ordered_means):

                relabeled_y[:, i] = np.where(y[:, i] == value, index, relabeled_y[:, i])

        return relabeled_y


    def calculate_means(self, X, y):

        means = list()

        for i, lbl in enumerate(range(self.n_clusters)): 

            mean = np.mean(X.values[y == lbl])
            means.append(mean)

        return np.array(means)
    

    def majority_voting(self, y):

        y = mode(y, axis=1).mode

        return y


    def eval(self, X, y_target):
        
        y = self.apply_algorithms(X)
        y = self.relabel(X,y)

        data = dict()

        for i, model in enumerate(self.algorithms):

            data[model.__class__.__name__] = jaccard_score(y_target, y[:, i], average='weighted')

        y = self.majority_voting(y)

        data[self.__class__.__name__] = jaccard_score(y_target, y, average='weighted')

        return pd.DataFrame([data])


if __name__ == '__main__':

    print('Hello, home!')