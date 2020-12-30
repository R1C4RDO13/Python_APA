import math


def Euclidean_distance(feat_one, feat_two):
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(feat_one)):
        squared_distance += (feat_one[i] - feat_two[i]) ** 2

    ed = math.sqrt(squared_distance)

    return ed

def moda(L_array):
    dic = {}
    for i in L_array:
        dic[i] += 1

    maxoccurs  = 0
    label = 0
    for i in dic:
        if dic[i] > maxoccurs:
            maxoccurs = dic[i]
            label = i
    return label