import numpy as np
import importlib
import src.fncs as fncs
import random


def check_submission_format():
    idTest = [1, 2, 3, 4]
    dataFolder = "data/test/"

    yPred = []
    for k, id in enumerate(idTest):
        xt, xv, yt, yv = fncs.loadTrial(dataFolder, id=id)
        yPred.append({"t": yt, "v": yv})

    y_t_start = [0.02, 0.02, 0.02, 0.02]
    y_t_end = [857.62, 861.82, 1203.42, 949.72]
    y_len = [8577, 8619, 12035, 9498]

    # First checking that the 't' values looks good. If this fails then either the data was not loaded
    # correctly or they were overwritten to be the incorrect values.
    for k in range(len(yPred)):
        assert yPred[k]["t"][0] == y_t_start[k]
        assert yPred[k]["t"][-1] == y_t_end[k]
        assert len(yPred[k]["t"]) == y_len[k]

    y_t_start = [0.02, 0.02, 0.02, 0.02]
    y_t_end = [857.62, 861.82, 1203.42, 949.72]
    y_len = [8577, 8619, 12035, 9498]

    # Checking the 'v' values.
    for k in range(len(yPred)):
        assert len(yPred[k]["v"]) == y_len[k]
        
        # Extracting the proportions of your predicitions
        n0 = np.sum(yPred[k]["v"] == 0)
        n1 = np.sum(yPred[k]["v"] == 1)
        n2 = np.sum(yPred[k]["v"] == 2)
        n3 = np.sum(yPred[k]["v"] == 3)
        print(
            "Trial{:02d}: n0={:4.2f} n1={:4.2f} n2={:4.2f} n3={:4.2f}".format(
                k, n0 / y_len[k], n1 / y_len[k], n2 / y_len[k], n3 / y_len[k]
            )
        )

        # Checking that things add up to 1
        assert (n0 + n1 + n2 + n3) == y_len[k]
