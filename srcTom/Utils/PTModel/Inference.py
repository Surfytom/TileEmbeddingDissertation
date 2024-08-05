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