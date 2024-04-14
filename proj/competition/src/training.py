"""Training script"""

import os
import json
import uuid
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.model import ConvNet
from src.utils import (
    device,
    dtype,
    loadFeatures,
    allocated_memory,
    load_data,
    evaluate,
    save_predictions,
    summary_perf,
    get_labels,
)
from src.check import check_submission_format
from scipy.stats import gmean

# pylint: disable=invalid-name


def training(epochs=10, batch_size=32, lr=1e-5, seed=42, window_size=40):
    """Training loop"""
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

    # val_ids = [2, 11, 25]
    # train_ids = list(set(np.array(range(29)) + 1).difference(val_ids))
    # train_ids = ids

    train_data, train_labels, train_dist = load_data(
        train_dir, train_ids, window_size=window_size
    )
    val_data, val_labels, val_dist = load_data(
        train_dir, val_ids, window_size=window_size
    )

    train_dist_gmean = gmean(train_dist, axis=0)
    val_dist_gmean = gmean(val_dist, axis=0)

    # take the average of train_dist and val_dist
    train_dist = np.mean(train_dist, axis=0)
    val_dist = np.mean(val_dist, axis=0)

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
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(train_dist).to(device).to(dtype)
    )
    # criterion = nn.MSELoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    # optimizer = torch.optim.Adadelta(
    #     model.parameters(), lr=0.1, rho=0.9, eps=1e-3, weight_decay=0.001
    # )
    # every 10000 epochs
    milestones = list(range(0, epochs, 10))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # Define the scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(xTrain, yTrain)
    val_dataset = TensorDataset(xVal, yVal)

    # Calculate weights for training set
    class_counts_train = np.bincount(yTrain.cpu().numpy())
    class_weights_train = 1.0 / class_counts_train
    weights_train = class_weights_train[yTrain.cpu().numpy()]

    # Calculate weights for validation set
    class_counts_val = np.bincount(yVal.cpu().numpy())
    class_weights_val = 1.0 / class_counts_val
    weights_val = class_weights_val[yVal.cpu().numpy()]

    # Create a WeightedRandomSampler for training set
    sampler_train = WeightedRandomSampler(
        torch.tensor(weights_train), len(weights_train)
    )

    # Create a WeightedRandomSampler for validation set
    sampler_val = WeightedRandomSampler(torch.tensor(weights_val), len(weights_val))

    # Create DataLoader with the sampler for training set
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler_train, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=sampler_val, drop_last=True
    )

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    unsampled_train_loader = DataLoader(train_dataset, batch_size=batch_size)
    unsampled_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_accuracy = 0
    val_accuracy = 0
    train_bal_acc = 0
    val_bal_acc = 0
    cm = np.zeros(4)
    losses = []
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
                f"Epoch {epoch}, t-acc: {train_accuracy:.2f}, v-acc: {val_accuracy:.2f}, loss: {loss.item():.2f}, t-bal-acc: {train_bal_acc:.2f}, v-bal-acc: {val_bal_acc:.2f}, cm: {np.around(cm, 2)}",
                refresh=True,
            )
            pbar_batches.set_postfix({"Memory (GB)": allocated_memory()})

        # Validation loop
        model.eval()

        train_accuracy = evaluate(model, unsampled_train_loader)
        val_accuracy = evaluate(model, unsampled_val_loader)

        losses.append(loss.item())

        yTrain, yTrainHat = get_labels(model, unsampled_train_loader)
        yVal, yValHat = get_labels(model, unsampled_val_loader)

        train_bal_acc, val_bal_acc, cm = summary_perf(yTrain, yTrainHat, yVal, yValHat)

    # save model
    torch.save(model.state_dict(), "models/model.pth")

    print(f"Train acc: {train_accuracy}, Val acc: {val_accuracy}")
    # package and save the results to a JSON file

    # Perform inference on the test dataset
    for i in tqdm(test_ids):
        test_data, _, _ = load_data(
            test_dir, [i], window_size=window_size, testing=True
        )
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

    training_information = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "window_size": window_size,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "loss": losses,
        "final_loss": losses[-1],
    }

    # create a uuid
    training_id = str(uuid.uuid4())

    # create a new directory with the uuid
    # copy the data/test directory to the new directory
    # also copy the models directory to the new directory
    # and store the training_information in a json file
    os.makedirs(f"results/{training_id}")
    os.system(f"cp -r data/test results/{training_id}")
    os.system(f"cp -r models results/{training_id}")

    # dump json to info.json

    with open(f"results/{training_id}/info.json", "w") as f:
        json.dump(training_information, f)

    print(f"Training information saved to results/{training_id}/info.json")

    check_submission_format()
