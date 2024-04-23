"""Utility functions for training and evaluation"""

import subprocess
import torch
import numpy as np
from scipy.fftpack import fft
import pandas as pd
from tqdm import tqdm
import src.fncs as fncs
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# pylint: disable=invalid-name

### Always import device to register correct backend
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# README: you should adjust based on your hardware
# NVIDIA GPUs Ampere uarch and after support BF16 (better precision than IEEE FP16)
# M2 Apple Silicon and after also support BF16 (CPU and GPU)
# Don't attempt to use FP16 on CPU, as it's not supported for GEMM
if device.type == "cuda":
    # detect if GPU is capable of BF16
    if torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    dtype = torch.float32
elif device.type == "mps":
    command = 'sysctl -a | grep "hw.optional.arm.FEAT_BF16"'
    with subprocess.Popen(command, stdout=subprocess.PIPE, shell=True) as process:
        output, error = process.communicate()

    if output.decode("utf-8").strip().endswith("1"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    # default to FP32 on CPU, because PyTorch doesn't support HGEMM on any CPU architecture
    dtype = torch.float32


def save_predictions(test_preds, id):
    """Save predictions to the correct file"""
    label_df = pd.read_csv(
        f"data/test/Trial{id:02d}_y.csv", names=["timestamp", "label"]
    )
    total_samples = label_df.shape[0]
    # divide test_preds into slices of 4
    # then take the mode of the slices
    n_samples = test_preds.shape[0]

    labels = []

    for i in range(0, n_samples, 4):
        labels.append(np.argmax(np.bincount(test_preds[i : i + 4])))

    # if there are remaining, fill the rest with the last label
    if len(labels) < total_samples:
        labels += [labels[-1]] * (total_samples - len(labels))

    label_df["label"] = labels

    # Save predictions
    label_df.to_csv(f"data/test/Trial{id:02d}_y.csv", index=False, header=False)


def load_data(dir, ids, window_size=32, testing=False):
    """Load data from the given directory and ids"""
    data_dfs = []
    value_counts = np.empty(4)
    for id in tqdm(ids):
        data_df = pd.read_csv(
            f"{dir}/Trial{id:02d}_x.csv".format(id),
            names=["timestamp", "acc_x", "acc_y", "acc_z", "pos_x", "pos_y", "pos_z"],
        )
        label_df = pd.read_csv(
            f"{dir}/Trial{id:02d}_y.csv", names=["timestamp", "label"]
        )

        # append value counts (in order of labels) to value_counts
        val_cnt = label_df["label"].value_counts()
        for i in range(4):
            if i not in val_cnt.index:
                val_cnt[i] = 0

        # sort by label
        value_counts = np.vstack(
            (value_counts, val_cnt.sort_index().values / val_cnt.sum())
        )
        print(np.around(value_counts[-1], 2))

        # correct the timestamp
        if not testing:
            label_df["timestamp"] = label_df["timestamp"] - 0.02

        # add a column for directional acc and pos, the vector magnitude
        data_df["acc_mag"] = np.linalg.norm(
            data_df[["acc_x", "acc_y", "acc_z"]], axis=1
        )
        data_df["pos_mag"] = np.linalg.norm(
            data_df[["pos_x", "pos_y", "pos_z"]], axis=1
        )

        # now merge based on timestamp (snap to nearest timestamp)
        # pad labels, as data is sampled higher
        data_df = pd.merge_asof(data_df, label_df, on="timestamp")
        data_dfs.append(data_df)

    data_dfs = pd.concat(data_dfs)

    # we want to transform this into slices of window_size
    # such that the data shape eventually becomes (n_samples, n_features, window_size)

    # data_dfs = data_dfs.drop(columns=["timestamp"])

    # if not testing:
    #     data_dfs = data_dfs.dropna()

    # data = data_dfs.drop(columns=["label"]).values
    # labels = data_dfs["label"].values
    # n_samples = data.shape[0]
    # n_features = 6

    # # reshape data
    # window_size += 1
    # n_slices = n_samples - window_size
    # data_slices = np.zeros((n_slices, n_features, window_size))
    # for i in range(n_slices):
    #     data_slices[i] = data[i : i + window_size].T

    # # remove NaN
    # data_slices = data_slices[~np.isnan(data_slices).any(axis=(1, 2))]
    # data = data_slices[:, :, :-1]
    # labels = labels[window_size:]
    # return data, labels

    data_dfs = data_dfs.drop(columns=["timestamp"])

    if not testing:
        data_dfs = data_dfs.dropna()

    data = data_dfs.drop(columns=["label"]).values
    labels = data_dfs["label"].values
    n_samples = data.shape[0]
    n_features = 8

    # reshape data
    window_size += 1
    n_slices = n_samples - window_size
    data_slices = np.zeros((n_slices, n_features, window_size))
    fft_slices = np.zeros((n_slices, n_features, window_size))
    for i in range(n_slices):
        data_slices[i] = data[i : i + window_size].T
        fft_slices[i] = np.abs(fft(data[i : i + window_size].T, axis=1))

    # remove NaN
    data_slices = data_slices[~np.isnan(data_slices).any(axis=(1, 2))]
    fft_slices = fft_slices[~np.isnan(fft_slices).any(axis=(1, 2))]
    data = np.concatenate((data_slices[:, :, :-1], fft_slices[:, :, :-1]), axis=1)
    # labels = labels[window_size:]
    # lables shoudl be centgered
    labels = labels[window_size // 2 : -window_size // 2]
    return data, labels, value_counts[1:]


def evaluate(model, data_loader):
    """Evaluate the model on the given data_loader"""
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy = correct / total
    return accuracy


def get_labels(model, data_loader):
    """return the predicted labels and the true labels"""
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y.tolist())
            y_pred.extend(predicted.tolist())
    return y_true, y_pred


# It loads the data and extracts the features
def loadFeatures(dataFolder, winSz, timeStep, idList):
    """Load the features for the given id"""
    for k, id in tqdm(enumerate(idList)):
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


def allocated_memory():
    """Print the allocated memory in GB"""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_reserved() / 1e9
    return float("nan")


def summary_perf(yTrain, yTrainHat, y, yHat):
    """This function produces a summary of performance metrics including a confusion matrix"""
    # Plotting confusion matrix for the non-training set:
    cm = metrics.confusion_matrix(y, yHat, normalize="true")
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Walk Hard", "Down Stairs", "Up Stairs", "Walk Soft"],
    )
    cm_diag = np.diag(cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")

    train_bal_score = metrics.balanced_accuracy_score(yTrain, yTrainHat)
    val_bal_score = metrics.balanced_accuracy_score(y, yHat)

    return train_bal_score, val_bal_score, cm_diag
