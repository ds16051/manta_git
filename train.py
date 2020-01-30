import torch
from torch import nn, optim
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from dataset import MantaDataset
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#IN THIS FILE ONLY: ALL ARRAYS TYPE OBJECTS SHOULD BE TENSORS, ASIDE FROM LABELS,WHICH ARE LISTS

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

###Compute Distance Matrix On Embeddings###
#returns a bxb tensor of pairwise euclidean distances, where b is batch size, and diagonal elements are -1
def pairwise_distances(embeddings):
    b = embeddings.shape[0]
    distance_matrix = torch.zeros(b,b)
    for i in range(b):
        for j in range(b):
            distance_matrix[i][j] = torch.dist(embeddings[i],embeddings[j]) #euclidean distance
            if i == j: distance_matrix[i][j] = -1
    return distance_matrix

#pairwise_distances = pairwise_distances(embeddings)
#print(pairwise_distances)

###Compute Batch-Hard Triplet Loss On Embeddings###
"""
we treat each image in turn as the anchor, and calculate a triplet loss for each anchor.
the final loss is the average of the loss for each anchor
to find the loss for each anchor a:
    1.) find d_pos = max[d(a,p)], where p has same identity as a (this is the hardest positive)
    2.) find d_neg = min[d(a,p)], where p has different identity to a (this is the hardest negative)
    3.) the loss for this anchor is max(d_pos - d_neg + margin,0)
"""
def batch_hard_triplet_loss(labels,embeddings,margin):#returns triplet loss for one batch, as 0d tensor
    margin = float(margin)
    distance_matrix = pairwise_distances(embeddings) 
    #embeddings never used after this, derive triplet loss from distance_matrix
    all_triplet_losses = torch.tensor([]) #list of triplet losses for each anchor
    
    #for each anchor
    for anc in range(distance_matrix.shape[0]):
        all_p_dists =torch.tensor([]) #all distances to positives with relation to this anc
        all_n_dists =torch.tensor([]) #all distances to negatives with relation to this anc
        for i in range(distance_matrix.shape[1]):
            if(not (i == anc)): # if i == anc then ignore since this is self comparison
                if(labels[i]==labels[anc]): #positive
                    all_p_dists = torch.cat((all_p_dists,torch.unsqueeze(distance_matrix[anc][i],dim=0)),dim = 0)
                    #distance_matrix[anc][i] is a 0d tensor, so we make it 1d (for concatenation) using unsqueeze
                else: #negative
                    all_n_dists = torch.cat((all_n_dists,torch.unsqueeze(distance_matrix[anc][i],dim=0)),dim = 0)
            
        max_p_dist = torch.max(all_p_dists) #maximum distance to positive with relation to this anc, 0d tensor
        min_n_dist = torch.min(all_n_dists) #minimum distance to negative with relation to this anc, 0d tensor
        triplet_loss_anc = torch.abs(max_p_dist - min_n_dist + torch.tensor(margin)) #batch hard triplet loss for this anchor, 0d tensor
        all_triplet_losses = torch.cat((all_triplet_losses,torch.unsqueeze(triplet_loss_anc,dim=0)),dim=0)    

    average_triplet_loss = torch.mean(all_triplet_losses)
    return average_triplet_loss

###Optimiser###
learning_rate = 0.001
weight_decay = 1e-5
optimiser= optim.Adam(params = model.parameters(),lr = learning_rate,weight_decay = weight_decay)

epochs = 5
train_losses = np.zeros(epochs)
###For Each Batch###
#We treat a batch as an epoch
for epoch in range(0,epochs):
    #Select Batch
    batch = batch_select(p=8,k=4) #randomly selects a size 32 batch
    batch_ims = batch[0] 
    batch_ids = batch[1]

    ###Compute Embeddings On Batch###
    embeddings = model(batch_ims)[0] #extremely slow, but verified this is 32 * 128, as expected
    #embeddings = torch.randn(32,128) #use this for now for speed
    
    ###Compute Loss On Batch ###
    loss = batch_hard_triplet_loss(batch_ids,embeddings,0.2)
    print(loss)
    train_losses[epoch] = loss
    loss.backward()
    print("backprop done")
    optimiser.step()
    optimiser.zero_grad()
    print("optimiser done")
    

print("1 epoch complete")

#plot losses
plt.plot(train_losses)
plt.savefig("figs/train_loss")

    



#to print an image
#(transforms.ToPILImage()(image)).show()
#[row][column]
