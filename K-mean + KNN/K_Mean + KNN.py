import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns;
from time import time
import os


# insert at 1, 0 is the script path (or '' in REPL)
data_path = os.path.join(os.path.dirname(__file__), '../Data')
KNN_Path = os.path.join(os.path.dirname(__file__), '../KNN')
root_path = os.path.join(os.path.dirname(__file__), '..')

sys.path.append(KNN_Path)
sys.path.append(data_path)
sys.path.append(root_path)


import ReadFile as rf
from Func import Euclidean_distance
from KNN import KNN

sns.set()  # for plot styling
style.use('ggplot')


class Image:
    Name = ""
    Data = []


class K_Means:
    def __init__(self, k=3, tolerance=0.01, max_iterations=5000):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def fit(self, data):
        t0 = time()
        # if the initial centroids are empty initialises based on the first items of the training data
        if len(self.centroids) == 0:
            for i in range(self.k):
                self.centroids[i] = data[i].Data

        # begin iterations
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            #  re-calculate the population of each cluster based on the distance to the nearest centroid
            for features in data:
                # distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                Distances = [Euclidean_distance(features.Data, self.centroids[centroid]) for centroid in self.centroids]
                # distances = [ cdist(features, self.centroids[centroid], metric='cityblock') for centroid in self.centroids]  #Manhattan distance
                # distances = [ distance.minkowski(features, self.centroids[centroid], 3)for centroid in self.centroids]  #Manhattan distance
                Classification = Distances.index(min(Distances))
                self.classes[Classification].append(features)

            previous = dict(self.centroids)

            # re-calculate the new centroid based no the average of the cluster population
            for Classification in self.classes:
                self.centroids[Classification] = np.average([i.Data for i in self.classes[Classification]], axis=0)

            isOptimal = False
            for centroid in self.centroids:
                isOptimal = False
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if Euclidean_distance(curr,
                                      original_centroid) < self.tolerance:  # when a centroid displacement is
                    # inferior to the tolerance
                    print("IsOptimal centroid:" + str(centroid))
                    isOptimal = True

            # To break the loop when optimal for all centroids
            if isOptimal:
                print(time() - t0)
                break

    def Predict(self, data):  # Used to classify the test data
        distances = [Euclidean_distance(data, self.centroids[centroid]) for centroid in self.centroids]  # find the
        # nearest centroid to a given data
        classification = distances.index(min(distances))
        return classification


def main():
    trainingData = rf.ReadFile(os.path.join(data_path ,"optdigits.tra"))  # Extract Training Data
    testingData = rf.ReadFile(os.path.join(data_path , "optdigits.tes"))    # Extract Testing Data

    # DataSet = [item.Data for item in trainingData]
    labels = [item.Name for item in trainingData]

    # same data but in a numpy array
    # DataSetNP = np.array(DataSet)

    # locate the unique number of diferent numbers
    n_digits = len(np.unique(labels))
    initialCentroids = {}
    # n_samples, n_features = DataSetNP.shape

    # to search the firsts centroids for each cluster number
    for i in range(n_digits):
        for x in range(len(trainingData)):
            if trainingData[x].Name == str(i) + "\n":
                initialCentroids[i] = np.array(trainingData[x].Data)
                break
            x += 1

    km = K_Means(n_digits)
    km.centroids = initialCentroids
    # km.fit(DataSetNP)

    km.fit(trainingData)
    # Plotting starts here
    # colors = 10 * ["r", "g", "c", "b", "k", 'm', 'y', 'k', 'w']
    # if n_features < 3:
    #    for centroid in km.centroids:
    #        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

    #    for classification in km.classes:
    #        color = colors[classification]
    #        for features in km.classes[classification]:
    #            plt.scatter(features[0], features[1], color=color, s=30)

    # plt.show()

    Labeling = []

    # to label each centroid, based on a KNN analise between the training data and centroids
    for i in km.centroids:
        centroid = km.centroids.get(i)
        distances = [Euclidean_distance(data.Data, centroid) for data in trainingData]
        NLabel = distances.index(min(distances))
        Labeling.append(trainingData[NLabel].Name)
        print(str(trainingData[NLabel].Name) + str(centroid))

    # labeling the entire cluster --very slow --my clusters suck
    #    for Cluster in km.classes:
    #        Cluster_Data = km.classes[Cluster]
    #        Values = []
    #        for Data in Cluster_Data:
    #            Values.append(KNN(trainingData, Data))
    #        Labeling.append(max(set(Values),key=Values.count))

    # forcing the labels to improve
    # Labeling = ['0\n', '1\n', '2\n', '3\n', '4\n', '5\n', '6\n', '7\n', '8\n', '9\n']
    # print(Labeling)

    # input()

    # plots the shape of each centroid
    aux = [km.centroids[centroid] for centroid in km.centroids]
    centers = np.array(aux).reshape(10, 8, 8)

    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.2);
    # plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(8, 3))

    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

    plt.show()

    print(Labeling)
    error = 0

    # Classification and test
    Errors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Errors per number
    Occurs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Number of occurs per number
    Bar = np.arange(11)
    X = np.arange(11)
    t0 = time()
    for x in testingData:

        cluster = km.Predict(np.array(x.Data))
        var = KNN(km.classes[cluster], x.Data)
        Occurs[int(x.Name)] += 1

        if x.Name != Labeling[int(var)]:
            error += 1
            Errors[int(x.Name)] += 1
            # print(x.Name + "/_" + Labeling[var])
    print(time() - t0)
    print("Accuracy on the testing set with k-mean :")
    Accuracy = (100 - (error / len(testingData) * 100))
    print(Accuracy)

    X[len(X) - 1] = Accuracy
    for i in range(len(Errors)):
        print("Accuracy on the testing " + str(i))
        x = 100 - (Errors[i] / Occurs[i] * 100)
        X[i] = x
        print(x)

    # Show Data Graph
    plt.bar(Bar, X)

    plt.xticks(Bar, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Total'])
    plt.show()


if __name__ == "__main__":
    main()
