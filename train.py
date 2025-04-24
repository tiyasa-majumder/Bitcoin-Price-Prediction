# set the matplotlib backend so figures can be saved in the background
# pip install torch scikit-learn matplotlib pandas
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import classification_report
from CNNmodel import TwitterCNNModel
import matplotlib
import pandas as pd
matplotlib.use("Agg")
# import the necessary packages

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
INIT_LR = 1e-3
BATCH_SIZE = 8
EPOCHS = 10
# All these hp??
# define the train and val splits
trainData = pd.read_pickle('trainData_2.pkl')
testData = pd.read_pickle('testData_2.pkl')
trainData.reset_index(inplace=True)
testData.reset_index(inplace=True)
trainData.drop(columns=["index"], inplace=True)
testData.drop(columns=["index"], inplace=True)

# trainData = trainData.iloc[:, 1:]
# testData = testData.iloc[:, 1:]
# print(len(trainData))
threshold = 0.5

TRAIN_SPLIT = 0.8
VAL_SPLIT = 1 - TRAIN_SPLIT
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # calculate the train/validation split
# print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = len(trainData) - (numTrainSamples)
print(len(trainData))
# (trainData1, valData1) = random_split(trainData,
#                                       [numTrainSamples, numValSamples],
#                                       generator=torch.Generator().manual_seed(42))
# (testData1, testData2) = random_split(testData,
#                                       [len(testData), 0],
#                                       generator=torch.Generator().manual_seed(42))

#!!! Not doing a random split here in order to capture time-series dependence
# initialize the train, validation, and test data loaders
# for i in range(len(testData)):
# print(testData)
# for i in trainData1.indices:
#     print(trainData["embeddings"][i])

# print(testData.index)
# for i in testData.index:
#     print(testData["embeddings"][i])

# calculate steps per epoch for training and validation set
trainSteps = (numTrainSamples) // BATCH_SIZE
valSteps = (numValSamples) // BATCH_SIZE
# print(trainData.head())
# print(trainDataLoader)
# initialize the LeNet model
# testData.reset_index(inplace = True)
# testData.drop(columns = ["index"],inplace = True)
# print(testData.columns)
print("[INFO] initializing the TwitterCNN model...")
model = TwitterCNNModel(
    d=768,
    nd=128,
    kernels=[3, 4, 5]).to(device)
# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.BCELoss()
# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
# print(testData.head())
startTime = time.time()
# print(type(trainData))
# print(trainData.dataset)
# loop over our epochs
# print(trainData.columns)
for e in range(0, EPOCHS):
    print("Starting epoch ", e)
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # Not shuffling since this seems like a time series data.
    trainCorrect = 0
    valCorrect = 0
    idx = 0
    # print(idx)
    # loop over the training set
    for i in range(0, numTrainSamples, BATCH_SIZE):
        # send the input to the device
        idx += 1
        x = (torch.tensor(np.array(trainData["embeddings"][i]))).permute(
            0, 2, 1)
        y = torch.rand(1, 2)  # Prob. of getting 0 and 1
        # print(y.shape)
        if (trainData.iloc[i]["final_label_2"] == 0):
            y[0,  0] = 1.0
            y[0,  1] = 0.0
        else:
            y[0,  0] = 0.0
            y[0,  1] = 1.0
        for j in range(i+1, i+BATCH_SIZE):
            if (j >= numTrainSamples):
                break

            x1 = (torch.tensor(np.array(trainData["embeddings"][j]))).permute(
                0, 2, 1)
            y1 = torch.rand(1, 2)  # Prob. of getting 1 and 0
            if (trainData.iloc[j]["final_label_2"] == 0):
                y1[0,  0] = 1.0
                y1[0,  1] = 0.0
            else:
                y1[0,  0] = 0.0
                y1[0, 1] = 1.0
            x = torch.concat([x, x1], dim=0)
            y = torch.concat([y, y1], dim=0)

        (x, y) = (x.double(), y.double())
        (x, y) = (x.to(device), y.to(device))
        # print(y)
        # print(x.shape)
        # print(y.shape)
        # print(idx)
        # perform a forward pass and calculate the training loss
        pred = model(x)
        # print(y.shape)
        # print(pred.shape)
        # print(pred)
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        predicted_classes = (pred > threshold).float()
        # print(predicted_classes, y)
        # print((predicted_classes == y))
        trainCorrect += ((predicted_classes == y).sum().item())/2 ## Here we have to divide bt 2 since it is counting both true twice whereas we want it once
        # print(trainCorrect)


# # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for i in range(numTrainSamples, len(trainData), BATCH_SIZE):

            x = (torch.tensor(np.array(trainData["embeddings"][i]))).permute(
                0, 2, 1)
            y = torch.rand(1, 2)  # Prob. of getting 0 and 1
            # print(y.shape)
            if (trainData.iloc[i]["final_label_2"] == 0):
                y[0,  0] = 1.0
                y[0,  1] = 0.0
            else:
                y[0,  0] = 0.0
                y[0,  1] = 1.0
            for j in range(i+1, i+BATCH_SIZE):
                if (j >= numTrainSamples):
                    break

                x1 = (torch.tensor(np.array(trainData["embeddings"][j]))).permute(
                    0, 2, 1)
                y1 = torch.rand(1, 2)  # Prob. of getting 1 and 0
                if (trainData.iloc[j]["final_label_2"] == 0):
                    y1[0,  0] = 1.0
                    y1[0,  1] = 0.0
                else:
                    y1[0,  0] = 0.0
                    y1[0, 1] = 1.0
                x = torch.concat([x, x1], dim=0)
                y = torch.concat([y, y1], dim=0)

            # send the input to the device

            (x, y) = (x.double(), y.double())
            # make the predictions and calculate the validation loss
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            # print(pred,y)
            # print(lossFn(pred, y))
            totalValLoss += lossFn(pred, y).float()
            # calculate the number of correct predictions
            predicted_classes = (pred > threshold).float()
            valCorrect += ((predicted_classes == y).type(
                torch.float).sum().item())/2
    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / numTrainSamples
    valCorrect = valCorrect / numValSamples
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect))

    # finish measuring how long training took
    endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluatio
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for i in testData.index:
        # send the input to the device
        x = (torch.tensor(np.array(testData["embeddings"][i]))).permute(
            0, 2, 1)
        x = x.double()
        x = x.to(device)

        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# # generate a classification report
# print(classification_report(testData.targets.cpu().numpy(),
#                             np.array(preds), target_names=testData.classes))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
# serialize the model to disk
torch.save(model, args["model"])
