import torch
from torch import nn, optim
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dataset import MantaDataset
from torch.utils.data import DataLoader

#returnss a BxB matrix of pairwise euclidean distances
def pairwise_distances(embeddings,b): #b is batch size
    distance_matrix = torch.zeros(b,b)
    for i in range(b):
        for j in range(b):
            distance_matrix[i][j] = torch.dist(embeddings[i],embeddings[j])

    return distance_matrix


#looks like inception needs a batch size of 32

####Define Model###
model = models.inception_v3(pretrained = False, transform_input = False)
model.fc = torch.nn.Linear(2048,128) #our latent space vectors should be 1x128

###Import Dataset###
dataset = MantaDataset(json_file = "~/Documents/mastersProject/dataSetOne/mantaAnnotations.json", root_dir = "~/Documents/mastersProject/dataSetOne/") #dataset is a list of dictionaries, with each dictionary having keys "image" and "label", each corresponding to a single image/label; dataset[0]["image"] is the first image, and dataset[0]["label"] is the first label.


# for each epoch
# batch construction: randomly sample p identities, and k items for each identity giving batch size k (start with p = 8, k = 4, giving batch size 32)
# maybe a first step is to construct a map from identity to list of all images for that identity? Then the random sampling would be easier. 

print(dataset[1]["label"])


#For each epoch
#For each batch
"""
for i,(batch) in enumerate(train_loader):
    if(i == 1): break #laptop is slow
    images = batch["image"] # batch of b images (b * 3 * 299 * 299)
    labels = batch["label"] # batch of b labels (b-long list of strings )

    #Step One: Compute B emmbeddings from a batch of B inputs
    embeddings = model(images)[0] # b * 128 tensor, NOTE:the [0] is because during training, inception gives also an auxiliary output which we do not care about. it does not do this for evaluation

    #Step Two: Compute the pairwise distance matrix of size b x b
    distance_matrix = pairwise_distances(embeddings,b)
    print(distance_matrix.shape)
"""
    
    








#batch sampling strategy https://arxiv.org/pdf/1703.07737.pdf page 3 second column
#randomly sample p identities, and then randomly sample k images of each individual, giving  batch p * k
#dataset is approx 100 individuals with 10 pics each

#to print an image
#(transforms.ToPILImage()(image)).show()

#https://omoindrot.github.io/triplet-loss


