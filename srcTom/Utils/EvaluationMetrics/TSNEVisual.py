import glob
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

coloursAndMarkers = {
    "loderunner": {"colour": "#F4A460", "marker": "o"},
    "megaman": {"colour": "#17BECF", "marker": "X"},
    "thelegendofzelda": {"colour": "#C71585", "marker": "s"},
    "supermariobros": {"colour": "#006400", "marker": "P"},
    "kidicarus": {"colour": "#9467BD", "marker": "D"},
}

def TrainTSNEModel(modelName):

    if not os.path.isdir(f"Models/{modelName}/UnifiedRep"):
        raise RuntimeError(f"Folder Models/{modelName}/UnifiedRep does not exist. Run inference on the model to get the unified representation saved.")
    
    gameNames = [path[path.rfind("/")+1:] for path in glob.glob(f"Models/{modelName}/UnifiedRep/*")]

    data = {gameName: np.load(f"Models/{modelName}/UnifiedRep/{gameName}/unifiedRep.npy") for gameName in gameNames}

    flattenedData = np.array([embedding for k, v in data.items() for embedding in v])
    
    tsne = TSNE(n_components=2)
    return data, tsne.fit_transform(flattenedData)

def DisplayTSNEEmbeddings(data, fitEmbeddings, modelName="", save=False):
    plt.figure(figsize=(7, 7))

    previousIndex = 0
    for i, gameName in enumerate(data.keys()):
        indexTo = data[gameName].shape[0]
        plt.scatter(fitEmbeddings[previousIndex:previousIndex+indexTo, 0], fitEmbeddings[previousIndex:previousIndex+indexTo, 1], label=gameName, c=coloursAndMarkers[gameName]["colour"], marker=coloursAndMarkers[gameName]["marker"],edgecolors='white', alpha=0.5, s=35)
        previousIndex = previousIndex+indexTo

    plt.legend(ncols=3, bbox_to_anchor=(1, -0.05))

    if save and modelName != "":
        plt.savefig(f"Models/{modelName}/TSNEFig.png", )

    plt.show()
