import torch
import torch.nn as nn

class TileEmbeddingVAE(nn.Module):

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.imageEncoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Flatten(),
        )

        self.textEncoder = nn.Sequential(
            nn.Linear(13, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        self.embeddingLayer = nn.Linear(4112, 256)

        self.imageDecoder = nn.Sequential(
            nn.Linear(256, 4096),
            nn.Unflatten(1, (16, 16, 16)),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

        self.textDecoder = nn.Sequential(
            nn.Linear(256, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 13),
            nn.Sigmoid(),
        )

    def encode(self, xImages, xText):
        encodedImage = self.imageEncoder(xImages)

        if self.debug:
            print(f"xText shape: {xText.shape}")

        encodedText = self.textEncoder(xText)

        if self.debug:
            print("EncodedImage shape: ", encodedImage.shape)
            print("encodedText shape: ", encodedText.shape)

        concatenateEmbedding = torch.cat((encodedImage, encodedText), 1)

        embedding = self.embeddingLayer(concatenateEmbedding)
        embedding = nn.functional.tanh(embedding)

        return embedding

    def decode(self, embedding):
        decodedImage = self.imageDecoder(embedding)
        decodedText = self.textDecoder(embedding)
        return decodedImage, decodedText

    def forward(self, xImages, xText):
        # Encoder
        encodedEmbedding = self.encode(xImages, xText)

        # Decoder
        yPredImage, yPredText = self.decode(encodedEmbedding)
        
        return yPredImage, yPredText
    
class TileEmbeddingVAEwMHA(nn.Module):

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self.imgEncConv1 = nn.Conv2d(3, 32, kernel_size=3, stride=3)
        self.imgEncConv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.imgEncConv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.imgEncBn1 = nn.BatchNorm2d(32)
        self.imgEncBn2 = nn.BatchNorm2d(32)
        self.imgEncBn3 = nn.BatchNorm2d(16)

        self.imgEncMha = nn.MultiheadAttention(32, 1, batch_first=True)
        self.imgEncScaler = nn.Parameter(torch.zeros(1))

        self.imageEncoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Flatten(),
        )

        self.textEncoder = nn.Sequential(
            nn.Linear(13, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        self.embeddingLayer = nn.Linear(4112, 256)

        self.imageDecoder = nn.Sequential(
            nn.Linear(256, 4096),
            nn.Unflatten(1, (16, 16, 16)),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

        self.textDecoder = nn.Sequential(
            nn.Linear(256, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 13),
            nn.Sigmoid(),
        )

    def useAttention(self, x):

        bs, c, h, w = x.shape
        x = x.reshape(bs, c, h * w).transpose(1, 2)

        attOut, attMap = self.imgEncMha(x, x, x)

        if self.debug:
            print(f"attOut shape: {attOut.size()}")
            print(f"attMap shape: {attMap.size()}")

        attOut = attOut.transpose(1, 2).reshape(bs, c, h, w)

        if self.debug:
            print(f"attOut reshaped shape: {attOut.size()}")

        return attOut, attMap

    def encodeImage(self, xImages):

        xImages = self.imgEncConv1(xImages)
        xImages = self.imgEncBn1(xImages)
        
        xImages = self.imgEncScaler * self.useAttention(xImages)[0] + xImages

        xImages = nn.functional.tanh(xImages)

        xImages = self.imgEncConv2(xImages)
        xImages = self.imgEncBn2(xImages)
        xImages = nn.functional.tanh(xImages)

        xImages = self.imgEncConv3(xImages)
        xImages = self.imgEncBn3(xImages)
        xImages = nn.functional.tanh(xImages)

        if self.debug:
            print(f"xImages b4 flatten shape: {xImages.size()}")

        xImages = xImages.flatten(start_dim=1)

        if self.debug:
            print(f"xImages flattened shape: {xImages.size()}")

        return xImages

    def encode(self, xImages, xText):

        if self.debug:
            print(f"xImages shape: {xImages.shape}")
            print(f"xText shape: {xText.shape}")

        encodedImage = self.encodeImage(xImages)

        encodedText = self.textEncoder(xText)

        if self.debug:
            print(f"EncodedImage shape: {encodedImage.size()}")
            print(f"encodedText shape: {encodedText.size()}")

        concatenateEmbedding = torch.cat((encodedImage, encodedText), 1)

        embedding = self.embeddingLayer(concatenateEmbedding)
        embedding = nn.functional.tanh(embedding)

        return embedding

    def decode(self, embedding):
        decodedImage = self.imageDecoder(embedding)
        decodedText = self.textDecoder(embedding)
        return decodedImage, decodedText

    def forward(self, xImages, xText):
        # Encoder
        encodedEmbedding = self.encode(xImages, xText)

        # Decoder
        yPredImage, yPredText = self.decode(encodedEmbedding)
        
        return yPredImage, yPredText
    
class LSTMModel(nn.Module):

    def __init__(self, debug=False):
        super().__init__()

        self.histLSTM = nn.LSTM(256, 128, batch_first=True)
        self.colLSTM = nn.LSTM(256, 128, batch_first=True)

        self.textLSTM = nn.LSTM(256, 128, batch_first=True)

        self.outputLayer = nn.Linear(128, 256)

    def forward(self, xHist, xText, xCol):
        
        histOut, (histH, histC) = self.histLSTM(xHist)

        #print(f"hist out shape b4: {histOut.size()}")
        #histOut = histOut[-1, :, :]
        #print(f"hist out shape b4: {histOut.size()}")

        colOut, (colH, colC) = self.colLSTM(xCol)

        #colOut = colOut[:, -1]

        hiddenAdd = torch.add(histH, histC)
        channelAdd = torch.add(colH, colC)

        #textOut, (textH, textC) = self.textLSTM(xText, (hiddenAdd, channelAdd)) if self.training else self.infTextLSTM(xText, (hiddenAdd, channelAdd))
        textOut, (textH, textC) = self.textLSTM(xText, (hiddenAdd, channelAdd))

        # print(f"text out size b4: {textOut.size()}")
        # if textOut.ndim == 2:
        #     textOut = textOut[-1, :]
        # else:
        #     textOut = textOut[:, -1, :]
        # print(f"text out size after: {textOut.size()}")

        output = nn.functional.tanh(self.outputLayer(textOut))
        
        return output
    
class VGLCLSTMModel(nn.Module):

    def __init__(self, debug=False):
        super().__init__()

        self.histLSTM = nn.LSTM(1, 128, batch_first=True)
        self.colLSTM = nn.LSTM(256, 128, batch_first=True)

        self.textLSTM = nn.LSTM(1, 128, batch_first=True)

        self.outputLayer = nn.Linear(128, 9)

    def forward(self, xHist, xText, xCol):
        
        histOut, (histH, histC) = self.histLSTM(xHist)

        #print(f"hist out shape b4: {histOut.size()}")
        #histOut = histOut[-1, :, :]
        #print(f"hist out shape b4: {histOut.size()}")

        colOut, (colH, colC) = self.colLSTM(xCol)

        #colOut = colOut[:, -1]

        hiddenAdd = torch.add(histH, histC)
        channelAdd = torch.add(colH, colC)
        
        textOut, (textH, textC) = self.textLSTM(xText, (hiddenAdd, channelAdd))

        # print(f"text out size b4: {textOut.size()}")
        # if textOut.ndim == 2:
        #     textOut = textOut[-1, :]
        # else:
        #     textOut = textOut[:, -1, :]
        # print(f"text out size after: {textOut.size()}")

        output = nn.functional.softmax(self.outputLayer(textOut), dim=1)
        # print(f"Output Size SoftMax: {output.size()}")
        
        return output

class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=3)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.flattenLayer = nn.Flatten()
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.batchNorm3 = nn.BatchNorm2d(16)
        pass

    def forward(self, x):
        print("In Forward!")
        x = self.layer1(x)
        x = self.batchNorm1(x)
        x = nn.functional.relu(x)
        print(x.shape)

        x = self.layer2(x)
        x = self.batchNorm2(x)
        x = nn.functional.relu(x)
        print(x.shape)

        x = self.layer3(x)
        x = self.batchNorm3(x)
        x = nn.functional.relu(x)
        print(x.shape)

        x = self.flattenLayer(x)
        print(x.shape)

        return x