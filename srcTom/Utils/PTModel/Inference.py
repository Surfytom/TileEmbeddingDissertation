import torch

import numpy as np
import pandas as pd

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def SaveUnifiedRepresentation(model, data, savePath):
    model.to(device)
    model.eval()

    if not os.path.isdir(savePath):
        print("Does not exists")
        os.mkdir(savePath)
    else:
        print("Does exist")
        raise RuntimeError("Save Path Already Exists. Delete folder to retry...")

    for gamename in data["gamename"].unique():
        gameData = data[data["gamename"] == gamename]
        # Adjust 48 x 48 resize to be dynamic for 5 x 5 or larger kernels
        imageArray = torch.tensor(np.reshape(np.array(gameData["image"].tolist()), (-1, 3, 48, 48)), dtype=torch.float32).to(device)
        affordanceArray = torch.tensor(gameData["encodedAffordances"].tolist(), dtype=torch.float32).to(device)

        embeddingArray = model.encode(imageArray, affordanceArray).detach().cpu().numpy()

        os.mkdir(f"{savePath}/{gamename}")
        np.save(f"{savePath}/{gamename}/unifiedRep.npy", embeddingArray)

@torch.no_grad()
def SaveLevelUnifiedRepresentation(model, data, savePath):
    model.to(device)
    model.eval()

    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    else:
        raise RuntimeError("Save Path Already Exists. Delete folder to retry...")
    
    os.mkdir(f"{savePath}/BubbleBobble")

    tiles = []
    embeddings = []

    for i, level in enumerate(data):

        imageArray = np.array(level)
        affordanceArray = np.zeros(shape=(imageArray.shape[0], 13))

        # Adjust 48 x 48 resize to be dynamic for 5 x 5 or larger kernels
        imageArrayTensor = torch.tensor(np.reshape(imageArray, (-1, 3, 48, 48)), dtype=torch.float32).to(device)
        affordanceArrayTensor = torch.tensor(affordanceArray, dtype=torch.float32).to(device)

        embeddingArray = model.encode(imageArrayTensor, affordanceArrayTensor).detach().cpu().numpy()

        centerTiles = imageArray[:, 16:32, 16:32, :]

        np.save(f"{savePath}/BubbleBobble/level{i}Embedding.npy", embeddingArray)

        tiles.extend(centerTiles)
        embeddings.extend(embeddingArray)

    np.save(f"{savePath}/BubbleBobble/centerTiles.npy", np.array(tiles))
    np.save(f"{savePath}/BubbleBobble/embeddings.npy", np.array(embeddings))

@torch.no_grad()
def ModelInference(model, data):

    model.to(device)
    model.eval()

    xImages = np.array(data["image"].tolist())
    yImages = xImages[:, 16:32, 16:32, :]

    xImageBatch = torch.tensor(xImages, dtype=torch.float32)
    xImageBatch = xImageBatch.reshape((-1, 3, 48, 48))
    xImageBatch = xImageBatch.to(device)

    yImageBatch = torch.tensor(yImages, dtype=torch.float32)
    yImageBatch = yImageBatch.reshape((-1, 3, 16, 16))
    yImageBatch = yImageBatch.to(device)
    
    xTextbatch = torch.tensor(data["encodedAffordances"].tolist(), dtype=torch.float32).to(device)

    yPredImages, yPredText = model(xImageBatch, xTextbatch)

    yPredImages = yPredImages.cpu().numpy()
    yPredText = yPredText.cpu().numpy()

    yPredText = np.where(yPredText > 0.5, 1, 0)

    return yPredImages, yPredText