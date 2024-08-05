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

        concatenateEmbeddding = torch.cat((encodedImage, encodedText), 1)

        embedding = self.embeddingLayer(concatenateEmbeddding)

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