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

###Outputting Image###
#pass this a tensor to show as image
def showImage(tensor_to_show):
    (transforms.ToPILImage()(tensor_to_show)).show()

###Import/Generate Dataset###
"""
datset is a list of dictionaries
each dictionary corresponds to one image
dataset[i]["image"] gives the ith image as torch.Tensor
dataset[i]["label"] gives the ith image's label as a string
"""
def generate_datasets(json_file,root_dir):
    manta_dataset = MantaDataset(json_file,image_dir)
    #Train/Test/Unknown Split
    train_length = int(np.floor(0.8 * len(manta_dataset)))
    test_length = int(np.floor(0.1 * len(manta_dataset)))
    unknown_length = len(manta_dataset)-train_length-test_length
    data_split = torch.utils.data.random_split(manta_dataset,[train_length,test_length,unknown_length])
    training_dataset = data_split[0]
    test_dataset = data_split[1]
    unknown_dataset = data_split[2]
    return(training_dataset,test_dataset,unknown_dataset)


###Construct/Load Dictionary###
"""
construct a dictionary from provided dataset,mapping from label to a list of all torch.Tensor images for that label
this dictionary is saved to file_to_save (which must end in ".pt")
"""
def generate_dictionary(dataset,file_to_save):
    dictionary = {}
    #print(len(dataset))
    for i in range(len(dataset)):
        #print(i)
        label = dataset[i]["label"]
        image = dataset[i]["image"]
        if(label in dictionary):
            dictionary[label].append(image)
        else:
            dictionary[label] = [image]
    torch.save(dictionary,file_to_save)
    return dictionary


###Batch Selection###
#NOTE: There are 10 images of each id
"""
we construct a batch by selecting p random identities, and then k images for each identity, giving batch size pk
guideline values: p = 8, k = 4 for batch size 32
this is done from dictionary
"""
#returns(images_batch,labels_batch);batch_labels is a (p*k)-long list;batch_images is a (p*k)* 3 * 299 * 299 tensor
def batch_select(p,k,dictionary): #select p individuals, k photos of each, from the given dictionary
    batch_images = torch.tensor([])
    batch_labels = []
    
    all_ids = list(dictionary.keys()) #list of all ids, i.e a list of the keys of dictionary
    rands = random.sample(range(len(all_ids)),p) #list of p chosen indices for all_ids
    chosen_ids = [all_ids[indice] for indice in rands] #list of p chosen individuals
    
    #for each chosen individual
    for i in range(len(chosen_ids)):
        _id = chosen_ids[i]
        images_pool = dictionary[_id] #list of all images of _id
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

###################################################################################################################
###################################################################################################################
###################################################################################################################

is_generate_dictionaries = False #True to generate dictionaries;;False to load dictionaries from .pt files 
is_train_net = False #True to train network and save weights,False to load network from .pt file
json_file = "mantaAnnotations.json" 
image_dir = "scratch/small_image_set/"

###Generate Datasets### 
if(is_generate_dictionaries):
    (train_dataset,test_dataset,unknown_dataset) = generate_datasets(json_file,image_dir)

    
###Generating Dictionaries###
if(is_generate_dictionaries):
    train_dict = generate_dictionary(train_dataset,"train_dict.pt")
    test_dict = generate_dictionary(test_dataset,"test_dict.pt")
    unknown_dict = generate_dictionary(unknown_dataset,"unknown_dict.pt")

###Loading Dictionaries###
if(not is_generate_dictionaries):
    train_dict = torch.load("train_dict.pt")
    test_dict = torch.load("test_dict.pt")
    unknown_dict = torch.load("unknown_dict.pt")
    print("dictionaries loaded")

####Model###
model = models.inception_v3(pretrained = False, transform_input = False)
model.fc = torch.nn.Linear(2048,128) # 128d latent space

###Optimiser###
learning_rate = 0.001
weight_decay = 1e-5
optimiser= optim.Adam(params = model.parameters(),lr = learning_rate,weight_decay = weight_decay)

###Training Loop OR Loading Weights###
epochs = 5
if(is_train_net):
    train_losses = np.zeros(epochs)
    ###For Each Batch###
    #We treat a batch as an epoch
    for epoch in range(0,epochs):
        #Select Batch
        batch = batch_select(p=8,k=4,dictionary =train_dict) #randomly selects a size 8*4=32 batch
        batch_ims = batch[0] 
        batch_ids = batch[1]

        ###Compute Embeddings On Batch###
        embeddings = model(batch_ims)[0] #32 * 128, the [0] is because inception net also gives an auxiliary output during training, but not during evaluation.
        
        ###Compute Loss On Batch ###
        loss = batch_hard_triplet_loss(batch_ids,embeddings,0.2)
        train_losses[epoch] = loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    
    ###Save Model###
    torch.save(model.state_dict(),"network_weights.pt")
    
    ###Plotting###
    plt.plot(train_losses)
    plt.savefig("train_loss.png")

if(not is_train_net):
    ###Load Saved Weights###
    model.load_state_dict(torch.load("network_weights.pt"))
    print("model loaded")

###Evaluation###
#For evaluation we should be using test_dict and unkown_dict, not the original datasets
model.eval()
test_keys = list(test_dict.keys())
output = model(torch.unsqueeze(test_dict[test_keys[0]][0],dim=0))





    



