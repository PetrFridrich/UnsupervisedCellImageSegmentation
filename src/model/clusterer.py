import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.metrics import jaccard_score

from sklearn.cluster import MiniBatchKMeans, KMeans, Birch, BisectingKMeans
from sklearn.mixture import GaussianMixture


class Clusterer():
    """
    A class for clustering data using multiple clustering algorithms and combining their results.

    This class provides functionality for applying multiple clustering algorithms, relabeling
    clusters to maintain consistency, performing majority voting across algorithms, and evaluating
    clustering performance using metrics like Jaccard score.

    Attributes:
        n_clusters (int): Number of clusters to form.
        algorithms (list): List of clustering algorithms used for predictions.
    """

    def __init__(self, n_clusters):
        """
        Initializes the Clusterer with a specified number of clusters and sets up algorithms.

        Parameters:
            n_clusters (int): Number of clusters to form.

        Returns:
            None
        """

        self.n_clusters = n_clusters

        self.algorithms = [KMeans(n_clusters=self.n_clusters), 
                           MiniBatchKMeans(n_clusters=self.n_clusters),
                           BisectingKMeans(n_clusters=self.n_clusters),
                           Birch(n_clusters=self.n_clusters),
                           GaussianMixture(n_components=self.n_clusters)]

        return None
    

    def fit_predict(self, X):
        """
        Fits the clustering algorithms to the input data and predicts cluster labels.

        Parameters:
            X (array-like): Input data for clustering.

        Returns:
            np.ndarray: Cluster labels after majority voting.
        """

        y = self.apply_algorithms(X)
        y = self.relabel(X,y)
        y = self.majority_voting(y)

        return y
    

    def apply_algorithms(self, X):
        """
        Applies all clustering algorithms to the input data.

        Parameters:
            X (array-like): Input data for clustering.

        Returns:
            np.ndarray: Cluster labels for each algorithm, where each column corresponds to an algorithm.
        """
        
        y = np.zeros((len(X), len(self.algorithms)))

        for i, model in enumerate(self.algorithms):

            y[:, i] = model.fit_predict(X)

        return y
    

    def relabel(self, X, y):
        """
        Relabels clusters to ensure consistency based on the order of cluster means.

        Parameters:
            X (array-like): Input data used for clustering.
            y (np.ndarray): Cluster labels for each algorithm.

        Returns:
            np.ndarray: Relabeled cluster labels.
        """

        relabeled_y = np.zeros_like(y)
        

        for i in range(len(self.algorithms)):

            means = self.calculate_means(X,y[:, i])
            ordered_means = np.Parametersort(means)[::-1]

            for index, value in enumerate(ordered_means):

                relabeled_y[:, i] = np.where(y[:, i] == value, index, relabeled_y[:, i])

        return relabeled_y


    def calculate_means(self, X, y):
        """
        Calculates the mean value of each cluster.

        Parameters:
            X (array-like): Input data used for clustering.
            y (np.ndarray): Cluster labels.

        Returns:
            np.ndarray: Array of mean values for each cluster.
        """

        means = list()

        for i, lbl in enumerate(range(self.n_clusters)): 

            mean = np.mean(X.values[y == lbl])
            means.append(mean)

        return np.array(means)
    

    def majority_voting(self, y):
        """
        Performs majority voting to determine the final cluster labels.

        Parameters:
            y (np.ndarray): Cluster labels for each algorithm.

        Returns:
            np.ndarray: Final cluster labels after majority voting.
        """

        y = mode(y, axis=1).mode

        return y


    def eval(self, X, y_target):
        """
        Evaluates clustering performance using Jaccard score.

        Parameters:
            X (array-like): Input data for clustering.
            y_target (array-like): Ground truth cluster labels.

        Returns:
            pd.DataFrame: DataFrame containing Jaccard scores for each algorithm and the combined results.
        """
        
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