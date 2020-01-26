from __future__ import print_function, division
import os
import torch
from torch import nn, optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

e = torch.tensor([])
i1 = torch.randn(3,299,299)
i1 = torch.unsqueeze(i1,dim=0)
print(i1.shape)
r = torch.cat((e,i1,i1,i1,i1),0)
print(r.shape)

#l = [0,1,2,3,4]
#inds = [0,2]
#r = [l[i] for i in inds]
#print(r)
