from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MantaDataset(Dataset):
    """Mantas dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_frame["annotations"][idx]["originalImageFileName"])
        image = io.imread(img_name)
        label = self.labels_frame["annotations"][idx]["individualId"]
        image = transform.resize(image,(299,299))
        #image = transform.resize(image,(20,20))
        image = np.swapaxes(image,1,2)
        image = np.swapaxes(image,0,1)
        image = torch.from_numpy(image)
        image = image.float()
        #image = torch.unsqueeze(image,0)
        sample = {'image': image, 'label': label}

        return sample


#dataset = MantaDataset(json_file = "~/Documents/mastersProject/dataSetOne/mantaAnnotations.json", root_dir = "~/Documents/mastersProject/dataSetOne/")
#myDataset is just a list of dictionaries, with each dictionary having structure  {'image': image, 'label': label}
#the images are just squashed to 299 x 299, this is certainly not a good idea
#maybe can use this as a basis from which to generate triplets for training


#dataset = MantaDataset(json_file = "~/Documents/mastersProject/dataSetOne/mantaAnnotations.json", root_dir = "~/Documents/mastersProject/dataSetOne/")


"""
fig = plt.figure()
for i in range(len(dataset)):
    sample = dataset[i]
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.axis('off')
    plt.imshow(sample["image"])
    print(sample["label"])
    if i == 3:
        plt.show()
        break
"""

