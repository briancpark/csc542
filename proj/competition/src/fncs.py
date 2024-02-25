import numpy as np


# This function loads the signal measusrementa and labels, and splits it into time and values.
def loadTrial(dataFolder, id):
    x = np.genfromtxt("{}Trial{:02d}_x.csv".format(dataFolder, id), delimiter=",")
    xt = x[:, 0]
    xv = x[:, 1:]
    y = np.genfromtxt("{}Trial{:02d}_y.csv".format(dataFolder, id), delimiter=",")
    yt = y[:, 0]
    yv = y[:, 1].astype(int)

    # Returning x measurements and y labels
    return xt, xv, yt, yv
