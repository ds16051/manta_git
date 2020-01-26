import torch
from torch import nn, optim
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dataset import MantaDataset
from torch.utils.data import DataLoader
import random

load_manta_dict_pt = True #set to true to load manta_dict from .pt file, set to false to construct manta_dict from manta_dataset, without loading a .pt file

####Define Model###
model = models.inception_v3(pretrained = False, transform_input = False)
model.fc = torch.nn.Linear(2048,128) # 128d latent space

###Import Dataset###
"""
datset is a list of dictionaries
each dictionary corresponds to one image
dataset[i]["image"] gives the ith image as torch.Tensor
dataset[i]["label"] gives the ith image's label as a string
"""
manta_dataset = MantaDataset(json_file = "~/Documents/mastersProject/dataSetOne/mantaAnnotations.json", root_dir = "~/Documents/mastersProject/dataSetOne/")

###Construct/Load Dictionary###
"""
construct a dictionary, mapping from label to a list of all torch.Tensor images for that label
this dictionary is saved to dict
"""
if(not load_manta_dict_pt): #construct manta_dict from manta_dataset
    manta_dict = {}
    print(len(manta_dataset))
    for i in range(len(manta_dataset)):
        print(i)
        label = manta_dataset[i]["label"]
        image = manta_dataset[i]["image"]
        if(label in manta_dict):
            manta_dict[label].append(image)
        else:
            manta_dict[label] = [image]

    torch.save(manta_dict,"manta_dict.pt")
    print("saved")
else: #load manta_dict from pt file
    manta_dict = torch.load("manta_dict.pt")
    print("loaded")


###Batch Selection###
#NOTE: There are 10 images of each id
"""
we construct a batch by selecting p random identities, and then k images for each identity, giving batch size pk
guideline values: p = 8, k = 4 for batch size 32
this is done from manta_dict
"""
#returns(images_batch,labels_batch);batch_labels is a (p*k)-long list;batch_images is a (p*k)* 3 * 299 * 299 tensor
def batch_select(p,k): #select p individuals, k photos of each
    batch_images = torch.tensor([])
    batch_labels = []
    
    all_ids = list(manta_dict.keys()) #list of all ids, i.e a list of the keys of manta_dict
    rands = random.sample(range(len(all_ids)),p) #list of p chosen indices for all_ids
    chosen_ids = [all_ids[indice] for indice in rands] #list of p chosen individuals
    
    #for each chosen individual
    for i in range(len(chosen_ids)):
        _id = chosen_ids[i]
        images_pool = manta_dict[_id] #list of all images of _id
        if(len(images_pool)<k): raise ValueError("k > available images of this individual")
        rands = random.sample(range(len(images_pool)),k) #list of k chosen indices for images_pool
        #for each chosen image of chosen individual
        for j in range(len(rands)):
            im = images_pool[rands[j]]
            im = torch.unsqueeze(im,dim = 0) # change from (3,299,299) to (1,3,299,299)
            batch_images = torch.cat((batch_images,im),dim = 0) #append im to batch_images
            batch_labels.append(_id) # append corresponding id

    return(batch_images,batch_labels) 
    #verified correct (by eye)

###For Each Batch###

###Compute Forward Pass On Batch###
p = 8 #individuals per batch
k = 4 #images per individual
my_batch = batch_select(p,k) #selects a size 32 batch
my_batch_ims = my_batch[0] 
my_batch_ids = my_batch[1]
#embeddings = model(my_batch_ims)[0] #extremely slow, but verified this is 32 * 128, as expected
embeddings = torch.randn(32,128) #use this for now for speed

###Compute Distance Matrix On Embeddings###

#returns a bxb matrix of pairwise euclidean distances, where b is batch size, and diagonal elements are -1
def pairwise_distances(embeddings):
    b = embeddings.shape[0]
    distance_matrix = torch.zeros(b,b)
    for i in range(b):
        for j in range(b):
            distance_matrix[i][j] = torch.dist(embeddings[i],embeddings[j])
            if i == j: distance_matrix[i][j] = -1
    return distance_matrix

#pairwise_distances = pairwise_distances(embeddings)
#print(pairwise_distances)

###Compute Batch-Hard Triplet Loss###
"""
we treat each image in turn as the anchor, and calculate a triplet loss for each anchor.
the final loss is the average of the loss for each anchor
to find the loss for each anchor a:
    1.) find d_pos = max[d(a,p)], where p has same identity as a
    2.) find d_neg = min[d(a,p)], where p has different identity to a
    3.) the loss for this anchor is max(d_pos-d_neg + margin,0)
"""
def batch_hard_triplet_loss(labels,embeddings,margin):
    distance_matrix = pairwise_distances(embeddings) #diagonal elements are -1, use <-0.5 to test validity
    #embeddings never used after this, task is now to derive triplet loss from distance_matrix
    #recall that the batch construction is such that we have k ims of one individual, followed by k ims of next...
    
    
    #for each anchor
    for anc in range(distance_matrix.shape[0]):
        #find largest distance to common identity
        for i in range(distance_matrix.shape[1]):




#loss = batch_hard_triplet_loss(my_batch_ids,embeddings,0.2)
#[row][column]










#to print an image
#(transforms.ToPILImage()(image)).show()
