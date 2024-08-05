import torch
import torch.nn as nn

import numpy as np

import Utils.PTModel.Models as Models
import Utils.PTModel.Losses as Losses

device = "cuda" if torch.cuda.is_available() else "cpu"

def TrainModel(dataObj, epochs, batchSize):
    # dataObj = {"trainData": trainingdata, "testData": testData, "weightArray": tfidfWeightArray}

    model = Models.TileEmbeddingVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    imageCritierion = nn.MSELoss()
    textCritierion = Losses.WeightedBCE(dataObj["weightArray"].to(device))

    imageLossWeight = 0.8
    textLossWeight = 1.0 - imageLossWeight

    model.to(device)
    model.train()

    trainData = dataObj["trainData"]
    losses = []

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

        print(f"Epoch {i}: loss {sum(losses[i])/len(losses[i])}")
    
    return model