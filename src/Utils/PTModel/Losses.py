import torch
import torch.nn as nn

class WeightedBCE(nn.Module):

    def __init__(self, weightedArray, debug=False):
        super().__init__()

        self.debug = debug

        self.weighedArray = weightedArray

    def forward(self, yPred, yTrue):

        bce_array = nn.functional.binary_cross_entropy(yPred, yTrue, reduction="none")
        weighted_array = torch.mul(bce_array, self.weighedArray)

        if self.debug:
            print(weighted_array.shape)

        bce_sum = torch.sum(weighted_array, axis=1)
        loss = torch.div(bce_sum, 13.0)
        loss = torch.mean(loss)

        return loss