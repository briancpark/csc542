import numpy as np
import importlib
import src.fncs as fncs
import matplotlib.pyplot as plt
import random


# It loads the data and extracts the features
def loadFeatures(dataFolder, winSz, timeStep, idList):
    for k, id in enumerate(idList):
        # Loading the raw data
        xt, xv, yt, yv = fncs.loadTrial(dataFolder, id=id)

        # Extracting the time window for which we have values for the measurements and the response
        timeStart = np.max((np.min(xt), np.min(yt)))
        timeEnd = np.min((np.max(xt), np.max(yt)))

        # Extracting the features
        _, feat = fncs.extractFeat(xt, xv, winSz, timeStart, timeEnd, timeStep)
        _, lab = fncs.extractLabel(yt, yv, winSz, timeStart, timeEnd, timeStep)

        # Storing the features
        if k == 0:
            featList = feat
            labList = lab
        else:
            featList = np.concatenate((featList, feat), axis=0)
            labList = np.concatenate((labList, lab), axis=0)

    return featList, labList
