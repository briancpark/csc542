import numpy as np
from src.utils import loadFeatures
from src.model import ConvNet
import subprocess
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def training(epochs=10, batch_size=32, lr=1e-5):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda:1" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        # detect if GPU is capable of BF16
        if torch.cuda.get_device_capability(0)[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif device.type == "mps":
        command = 'sysctl -a | grep "hw.optional.arm.FEAT_BF16"'
        process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        if output.decode("utf-8").strip().endswith("1"):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        dtype = torch.float32
    else:
        # default to FP32 on CPU, because PyTorch doesn't support HGEMM on any CPU architecture
        dtype = torch.float32

    # Specifying the data directory
    dirTrain = "data/train/"

    # Specifying some parameters for the feature extraction
    timeStep = 1
    winSz = 50

    # Specifying IDs for training and validation sets
    valIDs = [2, 11, 25]
    trainIDs = list(set(np.array(range(29)) + 1).difference(valIDs))
    print(trainIDs)

    # Recovering the features and labels
    xTrain, yTrain = loadFeatures(dirTrain, winSz, timeStep, trainIDs)
    xVal, yVal = loadFeatures(dirTrain, winSz, timeStep, valIDs)

    model = ConvNet().to(device).to(dtype)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adadelta(
    # model.parameters(), lr=0.1, rho=0.9, eps=1e-3, weight_decay=0.001
    # )
    # every 10000 epochs
    milestones = [x for x in range(0, epochs, 10000)]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # Define the scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Convert numpy arrays to PyTorch tensors
    xTrain, yTrain = torch.from_numpy(xTrain).to(dtype).to(device), torch.from_numpy(
        yTrain
    ).long().to(device)
    xVal, yVal = torch.from_numpy(xVal).to(dtype).to(device), torch.from_numpy(
        yVal
    ).long().to(device)

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(xTrain, yTrain)
    val_dataset = TensorDataset(xVal, yVal)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(xTrain.shape, yTrain.shape)
    print(xVal.shape, yVal.shape)

    pbar_epochs = tqdm(range(epochs), desc="Epochs")
    pbar_accuracy = tqdm(desc="Accuracy", leave=False)
    # Training loop
    for epoch in pbar_epochs:
        model.train()  # Set the model to training mode

        # Forward pass
        outputs = model(xTrain)
        loss = criterion(outputs, yTrain)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        # scheduler.step()

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if epoch % 50 == 0:
                outputs = model(xVal)
                val_loss = criterion(outputs, yVal)
                _, predicted = torch.max(outputs.data, 1)
                total = yVal.size(0)
                correct = (predicted == yVal).sum().item()
                val_accuracy = 100 * correct / total

                # also validate training set
                outputs = model(xTrain)
                _, predicted = torch.max(outputs.data, 1)
                total = yTrain.size(0)
                correct = (predicted == yTrain).sum().item()
                train_accuracy = 100 * correct / total

                pbar_accuracy.set_postfix(
                    {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
                )

        # Print loss for this epoch
        pbar_epochs.set_postfix({"loss": loss.item(), "val_loss": val_loss.item()})

    # validate
    model.eval()
    with torch.no_grad():
        outputs = model(xVal)
        _, predicted = torch.max(outputs.data, 1)
        total = yVal.size(0)
        correct = (predicted == yVal).sum().item()
        print(f"Accuracy: {100 * correct / total}%")
