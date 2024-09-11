import torch

import numpy as np
import pandas as pd

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def SaveUnifiedRepresentation(model, data, modelName):

    if os.path.isdir(f"Models/{modelName}/UnifiedRep"):
        raise RuntimeError(f"Folder Models/{modelName}/UnifiedRep already exists. Delete the folder and retry.")
    
    os.mkdir(f"Models/{modelName}/UnifiedRep")

    if model == None:
        model = torch.load(f"Models/{modelName}/{modelName}")

    model.to(device)
    model.eval()

    for gamename in data["gamename"].unique():
        gameData = data[data["gamename"] == gamename]
        # Adjust 48 x 48 resize to be dynamic for 5 x 5 or larger kernels
        imageArray = torch.tensor(np.reshape(np.array(gameData["image"].tolist()), (-1, 3, 48, 48)), dtype=torch.float32).to(device)
        affordanceArray = torch.tensor(gameData["encodedAffordances"].tolist(), dtype=torch.float32).to(device)

        embeddingArray = model.encode(imageArray, affordanceArray).detach().cpu().numpy()

        os.mkdir(f"Models/{modelName}/UnifiedRep/{gamename}")
        np.save(f"Models/{modelName}/UnifiedRep/{gamename}/unifiedRep.npy", embeddingArray)

@torch.no_grad()
def SaveLevelUnifiedRepresentation(model, levelData, modelName, gameName, affordanceData=None):

    if os.path.isdir(f"Models/{modelName}/LevelUnifiedRep/{gameName}"):
        raise RuntimeError(f"Folder Models/{modelName}/LevelUnifiedRep/{gameName} already exists. Delete the folder and retry.")
    
    if not os.path.isdir(f"Models/{modelName}/LevelUnifiedRep"):
        os.mkdir(f"Models/{modelName}/LevelUnifiedRep")

    os.mkdir(f"Models/{modelName}/LevelUnifiedRep/{gameName}")

    if model == None:
        model = torch.load(f"Models/{modelName}/{modelName}.pt")

    model.to(device)
    model.eval()
    
    tiles = []
    embeddings = []

    for i, level in enumerate(levelData):

        imageArray = np.array(level)
        
        if type(affordanceData) == np.ndarray:
            affordanceArray = affordanceData[i]
        else:
            affordanceArray = np.zeros(shape=(imageArray.shape[0], 13))

        # Adjust 48 x 48 resize to be dynamic for 5 x 5 or larger kernels
        imageArrayTensor = torch.tensor(np.reshape(imageArray, (-1, 3, 48, 48)), dtype=torch.float32).to(device)
        affordanceArrayTensor = torch.tensor(affordanceArray, dtype=torch.float32).to(device)

        embeddingArray = model.encode(imageArrayTensor, affordanceArrayTensor).detach().cpu().numpy()

        centerTiles = imageArray[:, 16:32, 16:32, :]

        np.save(f"Models/{modelName}/LevelUnifiedRep/{gameName}/Level {i}Embedding.npy", embeddingArray)

        tiles.extend(centerTiles)
        embeddings.extend(embeddingArray)

    np.save(f"Models/{modelName}/LevelUnifiedRep/{gameName}/centerTiles.npy", np.array(tiles))
    np.save(f"Models/{modelName}/LevelUnifiedRep/{gameName}/embeddings.npy", np.array(embeddings))

@torch.no_grad()
def ModelInference(model, data, clampTextOutput=True, modelName="", affordanceZeroed=False):

    if model == None and modelName != "":
        model = torch.load(f"Models/{modelName}.pt")

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
    
    if affordanceZeroed:
        xTextbatch = torch.zeros(size=(xImages.size()[0], 13), dtype=torch.float32)
    else:
        xTextbatch = torch.tensor(data["encodedAffordances"].tolist(), dtype=torch.float32).to(device)

    yPredImages, yPredText = model(xImageBatch, xTextbatch)

    yPredImages = yPredImages.cpu().numpy()
    yPredText = yPredText.cpu().numpy()

    if clampTextOutput:
        yPredText = np.where(yPredText > 0.5, 1, 0)

    return yPredImages, yPredText