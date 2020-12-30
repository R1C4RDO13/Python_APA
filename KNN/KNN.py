from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
# some_file.py
import sys
import os


# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(0, "E:\Python\Data")
data_path = os.path.join(os.path.dirname(__file__), '../Data')
sys.path.append(data_path)

import ReadFile as rf

# region Parsing of the file
from scipy.spatial import distance
def KNN(trainingData, image):
    aux = 10000.0
    valueTest = ""
    for i in trainingData:
        tmp = distance.euclidean(i.Data, image)
        if tmp < aux:
            aux = tmp
            valueTest = i.Name

    return valueTest




def main():
    trainingData = rf.ReadFile(os.path.join(data_path ,"optdigits.tra"))  # Extract Training Data
    testingData = rf.ReadFile(os.path.join(data_path , "optdigits.tes"))    # Extract Testing Data
    Errors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Errors per number
    Occurs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # Number of occurs per number
    MaxTotalError = 0

    Bar = np.arange(11)
    X = np.arange(11)

    for i in testingData:
        value = i.Name
        newValue = KNN(trainingData, i.Data)
        Occurs[int(i.Name)] += 1
        if value != newValue:
            MaxTotalError += 1
            Errors[int(i.Name)] += 1
            print(value + "//" + newValue)

    print("Accuracy on the testing set with k-nn :")
    Accuracy = 100 - ((MaxTotalError / len(testingData)) * 100)
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
