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
        
        
        label = self.labels_frame["annotations"][idx]["individualId"]

        image = transform.resize(image,(299,299))
        image = np.swapaxes(image,1,2)
        image = np.swapaxes(image,0,1)
        image = torch.from_numpy(image)
        image = image.float()

        #image = torch.unsqueeze(image,0)
        sample = {'image': image, 'label': label}

        return sample




#myDataset is just a list of dictionaries, with each dictionary having structure  {'image': image, 'label': label}

dataset = MantaDataset(json_file = "~/Documents/mastersProject/manta_git/mantaAnnotations.json", root_dir = "~/Documents/mastersProject/manta_git/scratch/small_image_set_crop/")
#(transforms.ToPILImage()(dataset[500]["image"])).show()

image = dataset[0]["image"]

####convert to PIL###
image = (transforms.ToPILImage()(image)) 
image.show()

#rotation
# image = (transforms.RandomRotation(degrees = 180,expand = True))(image)
# image = image = transforms.Resize((299,299))(image)
# image.show()

#perspective
#image = (transforms.RandomPerspective(distortion_scale = 0.85,p = 1))(image)
#image.show()

#affine


#brightness/contrast
image = (transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0))(image)
image.show()

###convert back to Tensor###
image = (transforms.ToTensor())(image) 
#print(image.shape)
#(transforms.ToPILImage()(image)).show()

