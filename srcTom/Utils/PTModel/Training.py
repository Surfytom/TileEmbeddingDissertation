import torch
import torch.nn as nn

import numpy as np

import Utils.PTModel.Models as Models
import Utils.PTModel.Losses as Losses
import Utils.PTModel.Inference as Inference

device = "cuda" if torch.cuda.is_available() else "cpu"

def TrainModel(dataObj, epochs, batchSize, modelClassToTrain=Models.TileEmbeddingVAE, continueTraining=None, earlyStoppingRange=0, dropLearningRate=False):
    # dataObj = {"trainData": trainingdata, "testData": testData, "weightArray": tfidfWeightArray}

    if continueTraining == None:
        model = modelClassToTrain()
    else:
        model = continueTraining
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    imageCritierion = nn.MSELoss()
    textCritierion = Losses.WeightedBCE(dataObj["weightArray"].to(device))

    imageLossWeight = 0.8
    textLossWeight = 1.0 - imageLossWeight

    model.to(device)
    model.train()

    trainData = dataObj["trainData"]
    losses = []
    batchLosses = []
    valTextLosses = []

    earlyStoppingCount = 0

    for i in range(epochs):

        losses.append([])

        for j in range(0, trainData.shape[0], batchSize):

            xImages = np.array(trainData.iloc[j:j+batchSize]["image"].tolist())
            yImages = xImages[:, 16:32, 16:32, :]
            # print(xImages.shape)
            # print(yImages.shape)

            xImageBatch = torch.tensor(xImages, dtype=torch.float32)
            xImageBatch = xImageBatch.reshape((-1, 3, 48, 48))
            xImageBatch = xImageBatch.to(device)

            yImageBatch = torch.tensor(yImages, dtype=torch.float32)
            yImageBatch = yImageBatch.reshape((-1, 3, 16, 16))
            yImageBatch = yImageBatch.to(device)
            
            xTextbatch = torch.tensor(trainData.iloc[j:j+batchSize]["encodedAffordances"].tolist(), dtype=torch.float32).to(device)

            yPredImages, yPredTexts = model(xImageBatch, xTextbatch)
            # print(yPredImages.shape)
            # print(yImageBatch.shape)

            imageLoss = imageCritierion(yPredImages, yImageBatch)
            textLoss = textCritierion(yPredTexts, xTextbatch)
            # print(imageLoss)
            # print(textLoss)

            loss = torch.add(torch.mul(imageLoss, imageLossWeight), torch.mul(textLoss, textLossWeight))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses[i].append(loss.cpu().detach().item())

        batchLosses.append(sum(losses[i])/len(losses[i]))

        valTrueData = torch.tensor(dataObj["testData"]["encodedAffordances"].to_list(), dtype=torch.float32).to(device)
        valPredData = torch.tensor(Inference.ModelInference(model, dataObj["testData"], clampTextOutput=False)[1], dtype=torch.float32).to(device)
        valTextLoss = textCritierion(valPredData, valTrueData)

        valTextLosses.append(valTextLoss)

        print(f"Epoch {i}: loss {batchLosses[i]} | Val Text Loss: {valTextLoss}")

        if earlyStoppingRange > 0 and i > 0:
            # print("In Early Stopping check")
            # print(batchLosses[-earlyStoppingRange:])
            # print(sum(batchLosses[-earlyStoppingRange:]) / earlyStoppingRange)
            # print(batchLosses[(-earlyStoppingRange)-1])
            if valTextLosses[-1] > valTextLosses[-2]:
                earlyStoppingCount += 1
            else:
                earlyStoppingCount = 0
            
            if earlyStoppingCount >= earlyStoppingRange:
                print("Early Stopping...")
                return model

        
        if dropLearningRate and i > 5 - 1:

            if sum(valTextLosses[-5:]) / 5 > valTextLosses[(-5)-1]:
                print("Dropping Learning Rate...")
                for g in optimizer.param_groups:
                    g['lr'] /= 10
                
                dropLearningRate = False
                    
    
    return model