import glob
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def TrainTSNEModel(pathToUnifiedData):
    
    gameNames = [path[path.rfind("/")+1:] for path in glob.glob(f"{pathToUnifiedData}/*")]

    data = {gameName: np.load(f"{pathToUnifiedData}/{gameName}/unifiedRep.npy") for gameName in gameNames}

    flattenedData = np.array([embedding for k, v in data.items() for embedding in v])
    
    tsne = TSNE(n_components=2)
    return data, tsne.fit_transform(flattenedData)

def DisplayTSNEEmbeddings(data, fitEmbeddings, colorArray = ["red", "black", "green", "orange", "purple"]):
    plt.figure(figsize=(7, 7))

    previousIndex = 0
    for i, gameName in enumerate(data.keys()):
        indexTo = data[gameName].shape[0]
        plt.scatter(fitEmbeddings[previousIndex:previousIndex+indexTo, 0], fitEmbeddings[previousIndex:previousIndex+indexTo, 1], label=gameName, c=colorArray[i], edgecolors='white', alpha=0.5)
        previousIndex = previousIndex+indexTo

    plt.legend(ncols=3, bbox_to_anchor=(1, -0.05))
    plt.show()
