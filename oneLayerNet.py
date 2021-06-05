import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image


######################################################################
# OneLayerNetwork
######################################################################

class OneLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(OneLayerNetwork, self).__init__()
        #implement OneLayerNetwork with torch.nn.Linear
        self.layer = torch.nn.Linear(784, 3)

    def forward(self, x):
        # x.shape = (n_batch, n_features)

        #implement the foward function
        x = x.view(x.size(0), -1)
        L1_out = self.layer(x)
        outputs = L1_out

        return outputs
