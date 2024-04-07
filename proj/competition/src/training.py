import numpy as np
from src.model import ConvNet
import subprocess
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from src.utils import (
    device,
    dtype,
    loadFeatures,
    allocated_memory,
    load_data,
    evaluate,
    save_predictions,
)
import os


def training(epochs=10, batch_size=32, lr=1e-5, seed=42, window_size=120):
    # Specifying the data directory
    train_dir = "data/train/"
    test_dir = "data/test/"

    os.makedirs("models", exist_ok=True)

    test_size = 0.2
    ids = list(range(1, 30))
    test_ids = list(range(1, 5))

    # shuffle ids with seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.shuffle(ids)

    train_ids = ids[: int((1 - test_size) * len(ids))]
    val_ids = ids[int((1 - test_size) * len(ids)) :]

    train_data, train_labels = load_data(train_dir, train_ids, window_size=window_size)
    val_data, val_labels = load_data(train_dir, val_ids, window_size=window_size)

    # Specifying some parameters for the feature extraction
    timeStep = 1
    winSz = 2
    # train_data, train_labels = loadFeatures(train_dir, winSz, timeStep, train_ids)
    # val_data, val_labels = loadFeatures(train_dir, winSz, timeStep, val_ids)
    # import matplotlib.pyplot as plt
    # # 3d scatter plot of 4, 5, 6 columns
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_data[:, 3], train_data[:, 4], train_data[:, 5], c=train_labels, alpha=0.2)
    # plt.savefig("scatter.png")

    # convert numpy tensors to torch tensors and load to device
    xTrain, yTrain = torch.from_numpy(train_data).to(dtype).to(
        device
    ), torch.from_numpy(train_labels).long().to(device)
    xVal, yVal = torch.from_numpy(val_data).to(dtype).to(device), torch.from_numpy(
        val_labels
    ).long().to(device)

    yTrain = yTrain.view(-1)
    yVal = yVal.view(-1)

    # load model
    model = ConvNet().to(device).to(dtype)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adadelta(
    #     model.parameters(), lr=0.1, rho=0.9, eps=1e-3, weight_decay=0.001
    # )
    # every 10000 epochs
    milestones = [x for x in range(0, epochs, 100)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # Define the scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(xTrain, yTrain)
    val_dataset = TensorDataset(xVal, yVal)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    train_accuracy = 0
    val_accuracy = 0

    pbar_epochs = tqdm(range(epochs), desc="Epochs")

    # Training loop
    for epoch in pbar_epochs:
        model.train()
        pbar_batches = tqdm(train_loader, desc="Batches", leave=False)
        for x, y in pbar_batches:
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Compute the gradients
            optimizer.step()  # Update the weights
            # scheduler.step()
            pbar_epochs.set_description(
                f"Epoch {epoch}, Train acc: {train_accuracy}, Val acc: {val_accuracy}"
            )
            pbar_batches.set_postfix(
                {"Loss": loss.item(), "Memory (GB)": allocated_memory()}
            )

        # Validation loop
        model.eval()

        train_accuracy = evaluate(model, train_loader)
        val_accuracy = evaluate(model, val_loader)

    # save model
    torch.save(model.state_dict(), "models/model.pth")

    for i in tqdm(test_ids):
        test_data, _ = load_data(test_dir, [i], window_size=window_size, testing=True)
        xTest = torch.from_numpy(test_data).to(dtype).to(device)

        test_loader = DataLoader(xTest, batch_size=batch_size, shuffle=False)

        # perfrom inference
        model.eval()
        test_preds = []
        for x in test_loader:
            # print the shape of x
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            test_preds.append(predicted.cpu().numpy())

        test_preds = np.concatenate(test_preds)
        # flatten
        test_preds = test_preds.flatten()
        # save predictions
        save_predictions(test_preds, i)
