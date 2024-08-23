import numpy as np
import pandas as pd

import cv2
import json

DEBUG = False

def ResizeLevel(levelImage, tileSize, kernelSize, heightOffset=0, widthOffset=0, resizeMultiplier=1, columnWise=False):

    if levelImage.ndim > 3 or levelImage.ndim < 2:
        print(f"Level Image has ({levelImage.ndim}) dimensions which is not valid. Valid dimensions: 2 (text files) or 3 (images)")
        return

    height, width = levelImage.shape[:2]

    if DEBUG:
        print(f"Image Height: {height} | Image Width: {width}")
        print(f"Image Height % 16: {height % 16} | Image Width % 16: {width % 16}")

    startPosition = kernelSize % 2

    boxHeight = tileSize * kernelSize
    boxWidth = tileSize * kernelSize

    if DEBUG:
        print(f"BoxHeigh: {boxHeight} | BoxWidth: {boxWidth}")
    
    originPoint = [heightOffset, widthOffset]

    if DEBUG:
        print(f"Start Position: {startPosition}")

    padding = ((kernelSize // 2) * 2) if kernelSize > 1 else 0
    
    widthWithKernel = (width // tileSize) - padding
    heightWithKernel = (height // tileSize) - padding

    if DEBUG:
        print(f"Kernel Size: {kernelSize}")
        print(f"Kernel Size % 2: {kernelSize % 2}")
        print(f"Kernel Size % 2 * 2: {((kernelSize % 2) * 2)}")

        print(f"Height With Kernel: {heightWithKernel} | Width With Kernel: {widthWithKernel}")

    tileGroupImages = []

    for i in range(heightWithKernel):
        for j in range(widthWithKernel):
            
            if levelImage.ndim == 2:
                cropped = levelImage[originPoint[0]:originPoint[0]+boxHeight, originPoint[1]:originPoint[1]+boxWidth]
            else:
                cropped = cv2.resize(levelImage[originPoint[0]:originPoint[0]+boxHeight, originPoint[1]:originPoint[1]+boxWidth, :], (boxHeight*resizeMultiplier, boxWidth*resizeMultiplier), interpolation=cv2.INTER_NEAREST)

            tileGroupImages.append(cropped)

            originPoint[1] += tileSize
        
        originPoint[0] += tileSize
        originPoint[1] = widthOffset

    # if os.path.isdir("testdata"):
    #     shutil.rmtree("testdata")
    
    # os.mkdir("testdata")

    # for i, tile in enumerate(tileGroupImages):
    #     cv2.imwrite(f"testdata/{i}-{j}.png", tile)

    tileGroupImages = np.array(tileGroupImages)

    if columnWise:
        if tileGroupImages.ndim == 3:
            tileGroupImages = np.reshape(tileGroupImages, (heightWithKernel, widthWithKernel, boxHeight*resizeMultiplier, boxWidth*resizeMultiplier))
            tileGroupImages = np.transpose(tileGroupImages, (1, 0, 2, 3)).reshape((heightWithKernel*widthWithKernel, boxHeight*resizeMultiplier, boxWidth*resizeMultiplier))
        else:
            tileGroupImages = np.reshape(tileGroupImages, (heightWithKernel, widthWithKernel, boxHeight*resizeMultiplier, boxWidth*resizeMultiplier, 3))
            tileGroupImages = np.transpose(tileGroupImages, (1, 0, 2, 3, 4)).reshape((heightWithKernel*widthWithKernel, boxHeight*resizeMultiplier, boxWidth*resizeMultiplier, 3))

    return tileGroupImages

def LoadLevelTextFile(filePath):
    
    tileArray = []

    with open(filePath, "r") as f:
        for line in f:
            tileArray.append(list(line.rstrip("\n")))
    
    tileArray = np.array(tileArray)

    return tileArray

def GetAffordances(tileArray, affordanceDictionary, centerTileOnly=True):

    if centerTileOnly:
        centerX, centerY = tileArray.shape[0] // 2, tileArray.shape[1] // 2
        #print("Tile: ", tileArray[centerX, centerY], " affordance: ", affordanceDictionary[tileArray[centerX, centerY]])
        return affordanceDictionary[tileArray[centerX, centerY]]
    
    outputAffordancs = []
    
    for i, row in enumerate(tileArray):
        outputAffordancs.append([])
        for j, tile in enumerate(row):
            outputAffordancs[i].append(affordanceDictionary[tile])

    return outputAffordancs

def GenerateData(LevelPathSpriteDict, savePath="test.csv", kernelSize=3, centerTileOnly=True):

    totalUniqueCombos = 0
    dataFramesArray = []

    for key, (gameLevelPaths, affordanceJsonPath) in LevelPathSpriteDict.items():

        print(f"Folder Name: {key}")

        affordanceJson = None

        with open(LevelPathSpriteDict[key][affordanceJsonPath], "r") as f:
            affordanceJson = json.load(f)["tiles"]
        
        textLevels = []

        for textLevel in LevelPathSpriteDict[key][gameLevelPaths]:
            
            if DEBUG:
                print(textLevel)

            t = LoadLevelTextFile(textLevel)

            if DEBUG:
                print("Loaded Level Shape: ", t.shape)
                print(f"Possible 3x3 Combinations for level: {(t.shape[0]-2) * (t.shape[1]-2)}")

            t = ResizeLevel(t, 1, kernelSize)

            if DEBUG:
                print("Processed Level Shape: ", t.shape)

            textLevels += t.tolist()
        
        textLevelGroups = np.array(textLevels)

        if DEBUG:
            print(f"Text levels combined shape {key}: {textLevelGroups.shape}")

        textLevelGroups = np.unique(textLevelGroups, axis=0)
        totalUniqueCombos += textLevelGroups.shape[0]

        if DEBUG:
            print(f"Unique Combinations {key}: {textLevelGroups.shape}\n")

        tempDataFrame = pd.DataFrame()

        tileSeries = pd.Series(textLevelGroups.tolist())
        affordanceSeries = pd.Series([GetAffordances(kernal, affordanceJson, centerTileOnly) for kernal in textLevelGroups])

        tempDataFrame["tiles"] = tileSeries
        tempDataFrame["affordances"] = affordanceSeries

        tempDataFrame.insert(0, 'gamename', key)

        dataFramesArray.append(tempDataFrame)

    print(f"Total Unique Combos of Tiles: {totalUniqueCombos}")

    pd.concat(dataFramesArray).to_csv(savePath)