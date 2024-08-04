import cv2
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import torch

from ast import literal_eval

def TextTileToImage(tileArray, tileSize, spritePath, savePath=None):

    outputImage = np.empty((tileSize*tileArray.shape[0], tileSize*tileArray.shape[1], 3), dtype=np.uint8)

    fileName = ""

    for i, row in enumerate(tileArray):
        for j, tile in enumerate(row):

            tile = '@' if tile == '.' else tile
            fileName += tile

            tileImage = cv2.cvtColor(cv2.imread(f"{spritePath}/{tile}.png"), cv2.COLOR_BGR2RGB)
            outputImage[i*tileSize:(i+1)*tileSize, j*tileSize:(j+1)*tileSize] = tileImage

    if savePath:
        cv2.imwrite(f"{savePath}/{fileName}.png", outputImage)

    return outputImage

def TFIDFWeightVector(data, uniqueClasses):
    # Initialize TfidfVectorizer from sklearn
    vectorizer = TfidfVectorizer(stop_words=None)

    # Fit the TfidfVectorizer to affordance values in the training set
    vectorizer.fit_transform(data["affordances"].apply(lambda x: str(x)))

    # Add the weights created for each affordance class to a easily indexable dictionary
    newDict = {affordanceClass: vectorizer.idf_[vectorizer.vocabulary_[affordanceClass]] if affordanceClass in vectorizer.vocabulary_ else 0.0 for affordanceClass in uniqueClasses}

    # Average each weight and scale by 1000
    # weightFreq = {k: v / sum(newDict.values()) for k, v in newDict.items()}
    # weightVector = [v * 1000 for v in newDict.values()] Forked github code dont know why * 1000??
    weightVector = [v for v in newDict.values()]

    tensorFromList = torch.tensor(weightVector, dtype=torch.float32)
    print(tensorFromList)

    return newDict, tensorFromList

def LoadTrainTestData(pathToDataCsv, testSetSize=0.1, shuffle=False, randomState=1):

    dataFrame = pd.read_csv(pathToDataCsv)

    dataFrame['affordances'] = dataFrame['affordances'].apply(lambda x: literal_eval(str(x)))
    dataFrame['tiles'] = dataFrame['tiles'].apply(lambda x: np.array(literal_eval(str(x))))

    return train_test_split(dataFrame, test_size=testSetSize, random_state=randomState, shuffle=shuffle)

def LoadCrossValTrainTestData(pathToDataCsv, shuffle=False, randomState=1):

    crossValDataDict = {}

    spritePaths = {
        "kidicarus": "../data/tomData/sprites/kidicarus", 
        "loderunner": "../data/tomData/sprites/loderunner",
        "megaman": "../data/tomData/sprites/megaman",
        "supermariobros": "../data/tomData/sprites/supermariobros",
        "thelegendofzelda": "../data/tomData/sprites/thelegendofzelda",
    }

    dataFrame = pd.read_csv(pathToDataCsv)

    dataFrame['affordances'] = dataFrame['affordances'].apply(lambda x: literal_eval(str(x)))
    dataFrame['tiles'] = dataFrame['tiles'].apply(lambda x: np.array(literal_eval(str(x))))

    dataFrame['image'] = [TextTileToImage(row['tiles'], 16, spritePaths[row['gamename']]) for index, row in dataFrame.iterrows()]

    uniqueAffordances = set([affordance for affordanceList in dataFrame['affordances'] for affordance in affordanceList])

    print(uniqueAffordances)

    if shuffle:
        dataFrame = dataFrame.sample(frac=1, random_state=randomState).reset_index()

    for gameName in dataFrame['gamename'].unique():
        testData = dataFrame[dataFrame['gamename'] == gameName]
        trainData = dataFrame.drop(testData.index)

        affordanceDict, tfidfWeightArray = TFIDFWeightVector(trainData, uniqueAffordances)

        trainData["encodedAffordances"] = [np.sum(np.array([np.where(np.array(list(affordanceDict.keys())) == affordance, 1, 0) for affordance in row['affordances']]), axis=0) for index, row in trainData.iterrows()]
        testData["encodedAffordances"] = [np.sum(np.array([np.where(np.array(list(affordanceDict.keys())) == affordance, 1, 0) for affordance in row['affordances']]), axis=0) for index, row in testData.iterrows()]
        
        crossValDataDict[gameName] = {"trainData": trainData, "testData": testData, "weightArray": tfidfWeightArray}
    
    return crossValDataDict