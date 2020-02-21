from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from skimage.util import crop
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from skimage.draw import rectangle_perimeter

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
        image = io.imread(img_name) #at loading: (height,width,channels)
        #image = Image.fromarray(np.uint8(image))
        #draw = ImageDraw.Draw(image)
        
        
        label = self.labels_frame["annotations"][idx]["individualId"]
        # box = self.labels_frame["annotations"][idx]["box_xmin_ymin_xmax_ymax"]
        # x1 = box[0]
        # x2 = box[2]
        # y1 = box[1]
        # y2 = box[3]

        image = transform.resize(image,(299,299))
        image = np.swapaxes(image,1,2)
        image = np.swapaxes(image,0,1)
        image = torch.from_numpy(image)
        image = image.float()

        #image = torch.unsqueeze(image,0)
        sample = {'image': image, 'label': label}

        return sample




#myDataset is just a list of dictionaries, with each dictionary having structure  {'image': image, 'label': label}
#the images are just squashed to 299 x 299, this is certainly not a good idea
#maybe can use this as a basis from which to generate triplets for training


dataset = MantaDataset(json_file = "~/Documents/mastersProject/manta_git/mantaAnnotations.json", root_dir = "~/Documents/mastersProject/manta_git/scratch/small_image_set/")
(transforms.ToPILImage()(dataset[500]["image"])).show()



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

