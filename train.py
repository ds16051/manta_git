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
import classifier
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import os
import gzip

#empty log file
f = open('log.txt', 'r+')
f.truncate(0)
f.close()

#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#IN THIS FILE ONLY: ALL ARRAYS TYPE OBJECTS SHOULD BE TENSORS, ASIDE FROM LABELS,WHICH ARE LISTS

###Outputting Image###
#pass this a tensor to show as image
def showImage(tensor_to_show):
    (transforms.ToPILImage()(tensor_to_show)).show()

def file_write(out):
    text_file = open("log.txt", "a+")
    text_file.write(str(out))
    text_file.write("\n")
    text_file.close()


###Generate Dataset###
"""
datset is a list of dictionaries
each dictionary corresponds to one image
dataset[i]["image"] gives the ith image as torch.Tensor
dataset[i]["label"] gives the ith image's label as a string
"""
def generate_dataset(json_file,image_dir):
    dataset = MantaDataset(json_file,image_dir)
    return dataset



###Construct/Load Dictionary###
"""
In this function we choose 5% of the identities to act as unknown. All images for these classes are returned in the list unknown_list.
Then for each of the remaining ids, 80% of the images for that id are used in train_dict, and the other 20% are used in test_dict.
The keys for train_dict and test_dict are the same. Keys are ids, and they map to images for that id.
"""
def generate_save_dictionaries(dataset):
    #Construct dictionary of whole dataset, mapping from id to list of images for that id
    whole_dictionary = {}
    #for i in range(len(dataset)):
    for i in range(100):
        label = dataset[i]["label"]
        image = dataset[i]["image"]
        if(label in whole_dictionary):
            whole_dictionary[label].append(image)
        else:
            whole_dictionary[label] = [image]

    all_ids = list(whole_dictionary.keys())
    num_unknown_ids = int(np.ceil(0.05 * len(all_ids))) 
    unknown_ids = all_ids[0:num_unknown_ids] #5% of classes are selected to act as unseen
    known_ids = all_ids[num_unknown_ids:len(all_ids)] 

    ###Make List Of Unknown Images###
    unknown_list = []
    for i in range(len(unknown_ids)):
        current_images = whole_dictionary[unknown_ids[i]]
        for j in range(len(current_images)):
            unknown_list.append(current_images[j])

    ###Make Train and Test Dictionaries###
    train_dict = {}
    test_dict = {}
    for i in range(len(known_ids)):
        current_id = known_ids[i]
        current_images = whole_dictionary[current_id]
        num_train = int(np.floor(0.8 * len(current_images)))
        train_dict[current_id] = current_images[0:num_train]
        test_dict[current_id] = current_images[num_train:len(current_images)]
    
    torch.save(unknown_list,"unknown_list_mini.pt")
    torch.save(train_dict,"train_dict_mini.pt")
    torch.save(test_dict,"test_dict_mini.pt")
    return(train_dict,test_dict,unknown_list)


###Batch Selection###
#Try to do augmentation in here
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
            #im = (transforms.ToPILImage()(im)) #convert from tensor to PIL
            #####AUGMENT DATA HERE#####
            #im is (3,299,299) tensor


            ###########################
            #im = (transforms.ToTensor())(im) #convert from PIL back to tensor


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


###Evaluation###
#For evaluation we should be using train_dict, test_dict, and unknown_list, not the original dataset
#We will calculate the metrics test_accuracy and open_set_accuracy
"""
    1.) Calculate and store embeddings and ids for train set, test set and unknown set
    2.) Classify the test and unknown embeddings using OSNN, with the train embeddings as the training set.
"""
###Calculate embeddings
def calculate_embeddings(train_dict,test_dict,unknown_list,model):
    with torch.no_grad():
        #file_write("started function")
        model.eval()
        train_embeddings = []
        train_ids = []
        test_embeddings = []
        test_ids = []
        unknown_embeddings = []
        train_test_keys = list(train_dict.keys())
        
        
        #calculate train_embeddings
        #for each identity
        for i in range(len(train_test_keys)):
            #file_write(i)
            key = train_test_keys[i]
            images = train_dict[key]
            #for each image of this identity
            for j in range(len(images)):
                image = torch.unsqueeze(images[j],dim=0)
                embedding = model(image)
                train_embeddings.append(embedding)
                train_ids.append(key)
        #file_write("train embeddings done")
        #torch.save(train_embeddings,"train_embeddings_mini.pt")
        #torch.save(train_ids,"train_ids_mini.pt")
        
        # #calculate test_embeddings
        for i in range(len(train_test_keys)):
            #file_write(i)
            key = train_test_keys[i]
            images = test_dict[key]
            for j in range(len(images)):
                image = torch.unsqueeze(images[j],dim=0)
                embedding = model(image)
                test_embeddings.append(embedding)
                test_ids.append(key)
        
        # torch.save(test_embeddings,"test_embeddings_mini.pt")
        # torch.save(test_ids,"test_ids_mini.pt")
        #file_write("test embedding done")

        # #calculate unknown_embeddings
        for i in range(len(unknown_list)):
            #file_write(i)
            image = torch.unsqueeze(unknown_list[i],dim=0)
            embedding = model(image)
            unknown_embeddings.append(embedding)
        
        # torch.save(unknown_embeddings,"unknown_embeddings_mini.pt")
        #file_write("unknown embedding done")
        
        return(train_embeddings,train_ids,test_embeddings,test_ids,unknown_embeddings)

#Calculate test and unknown accuracies, given the dictionaries, and a trained model, using osnn
def accuracy(train_dict,test_dict,unknown_list,model):
    with torch.no_grad():
        model.eval()
        #1) Caclulate embeddings from provided model
        (train_embeddings,train_ids,test_embeddings,test_ids,unknown_embeddings) = calculate_embeddings(train_dict,test_dict,unknown_list,model)
        #file_write("calculated embeddings")
        #2) Calculate OSNN threshold from training embeddings
        threshold = classifier.osnn_threshold(train_embeddings,train_ids)
        #file_write("calculated threshold")
        #3) Calculate accuracy on test set
        correct = float(0)
        total = float(0)
        for i in range(len(test_embeddings)):
            prediction = classifier.osnn_classify(train_embeddings,train_ids,test_embeddings[i],threshold)
            target = test_ids[i]
            if(prediction == target): correct = correct + 1
            total = total + 1
        test_accuracy = float(100) * (correct/total)
        #file_write("test accuracy calculated")
        #file_write(test_accuracy)
        #4) Caclulate accuracy on unknown set
        correct = float(0)
        total = float(0)
        for i in range(len(unknown_embeddings)):
            prediction = classifier.osnn_classify(train_embeddings,train_ids,unknown_embeddings[i],threshold)
            if(prediction == "unknown"): correct = correct + 1
            total = total + 1
        unknown_accuracy = float(100) * (correct/total)
        #file_write("unknown accuracy caclulated")
        #file_write(unknown_accuracy)

        return(test_accuracy,unknown_accuracy)

#accuracy with simple nn
def accuracy_nn(train_dict,test_dict,unknown_list,model):
    with torch.no_grad():
        model.eval()
        (train_embeddings,train_ids,test_embeddings,test_ids,unknown_embeddings) = calculate_embeddings(train_dict,test_dict,unknown_list,model)
        
        correct = float(0)
        total = float(0)
        for i in range(len(test_embeddings)):
            prediction = classifier.nn_classify(train_embeddings,train_ids,test_embeddings[i])
            target = test_ids[i]
            #file_write("prediction")
            #file_write(prediction)
            #file_write("target")
            #file_write(target)
            if(prediction == target): correct = correct + 1
            total = total + 1
        #file_write("correct")
        #file_write(correct)
        test_accuracy = float(100) * (correct/total)
        #file_write("test accuracy calculated")
        #file_write(test_accuracy)
        return test_accuracy

#simple nn accuracy  with topk
def accuracy_nn_topk(train_dict,test_dict,unknown_list,model,k):
    with torch.no_grad():
        model.eval()
        (train_embeddings,train_ids,test_embeddings,test_ids,unknown_embeddings) = calculate_embeddings(train_dict,test_dict,unknown_list,model)
        
        correct = float(0)
        total = float(0)
        #file_write("started topk")
        #file_write(len(test_embeddings))
        for i in range(len(test_embeddings)):
            #file_write(i)
            predictions = classifier.nn_classify_topk(train_embeddings,train_ids,test_embeddings[i],k)
            target = test_ids[i]
            #file_write("prediction")
            #file_write(predictions)
            #file_write("target")
            #file_write(target)
            if(target in predictions): correct = correct + 1
            total = total + 1
        #file_write("correct")
        #file_write(correct)
        test_accuracy = float(100) * (correct/total)
        #file_write("top k test accuracy calculated")
        #file_write(test_accuracy)
        return test_accuracy

#takes a set of embeddings, and performs dimensionality reduction using t-SNE.
#embeddings is a list of torch tensors.
#labels is a list of strings
def plot_tsne(embeddings,labels):
    #convert tensors to np arrays
    embeddings_array = embeddings[0].numpy()
    for i in range(1,len(embeddings)):
        embeddings_array = np.vstack((embeddings_array,embeddings[i].numpy()))

    #use tsne for dimensionality reduction
    tsne = TSNE(n_components=2,verbose=0,perplexity = 40,n_iter = 300)
    tsne_results = tsne.fit_transform(embeddings_array)

    #ensure labels are not just numbers, because searborn doesn't accept number labels
    for i in range(len(labels)):
        labels[i] = labels[i] + "a"

    unique_labels = np.unique(labels)
    
    #seaborn plots pd data frames, so convert to this format
    reduced_data = pd.DataFrame()
    reduced_data["d1"] = tsne_results[:,0]
    reduced_data["d2"] = tsne_results[:,1]
    reduced_data["label"] = labels

    print(reduced_data)

    #Scatterplot using seaborns
    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
        x="d1", y="d2",
        hue="label",
        #palette=sns.color_palette("hls", len(reduced_data["label"])),
        data=reduced_data[(reduced_data["label"]==unique_labels[5]) | (reduced_data["label"]==unique_labels[6]) ],
        legend="full",
        #alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig("scatter.png")


    





###################################################################################################################
#################################################----MAIN----######################################################
###################################################################################################################
#file_write("main")
is_generate_dictionaries = False #True to generate dictionaries;;False to load dictionaries from _mini.pt files 
is_train_net = True #True to train network and save weights,False to load network from _mini.pt file
json_file = "mantaAnnotations.json" 
image_dir = "scratch/small_image_set_crop/"
#image_dir = "scratch/mini/"
#image_dir = "scratch/small_image_set/"

###Generate Dataset### 
if(is_generate_dictionaries):
    dataset = generate_dataset(json_file,image_dir)
    file_write("dataset generated")
    
###Generate and Save Dictionaries###
if(is_generate_dictionaries):
    (train_dict,test_dict,unknown_list) = generate_save_dictionaries(dataset)
    file_write("dictionaries generated")
    

###Load Dictionaries###
if(not is_generate_dictionaries):
    train_dict = torch.load("train_dict_mini.pt")
    test_dict = torch.load("test_dict_mini.pt")
    unknown_list = torch.load("unknown_list_mini.pt")
    file_write("dictionaries loaded")

####Model###
model = models.inception_v3(pretrained = True, transform_input = False)
model.fc = torch.nn.Linear(2048,128) # 128d latent space

###Optimiser###
learning_rate = 0.0005
weight_decay = 1e-5
optimiser= optim.Adam(params = model.parameters(),lr = learning_rate,weight_decay = weight_decay)

###Training Loop OR Loading Weights###
file_write("started training")
epochs = 100
batches_per_test_step = 5 #how many batches to train on before testing
model.train()
if(is_train_net):
    train_losses = np.zeros(epochs)
    test_accuracies = []
    ###For Each Batch###
    #We treat a batch as an epoch
    for epoch in range(0,epochs):
        #Select Batch
        batch = batch_select(p=8,k=4,dictionary = train_dict) #randomly selects a size 8*4=32 batch
        batch_ims = batch[0] 
        batch_ids = batch[1]

        ###Compute Embeddings On Batch###
        embeddings = model(batch_ims)[0] #32 * 128, the [0] is because inception net also gives an auxiliary output during training, but not during evaluation.
        
        ###Compute Accuracy On Batch
        # if((epoch % batches_per_test_step == 0)):
        #     simple_accuracy = accuracy_nn(train_dict,test_dict,unknown_list,model)
        #     test_accuracies.append(simple_accuracy)

        ###Compute Loss On Batch ###
        loss = batch_hard_triplet_loss(batch_ids,embeddings,0.2)
        train_losses[epoch] = loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    
    ###Save Model###
    torch.save(model.state_dict(),"network_weights_mini.pt")
    file_write("model trained")
    
    ###Plotting###
    plt.plot(train_losses)
    plt.savefig("train_loss_mini.png")

    # plt.plot(np.array(test_accuracies))
    # plt.savefig("test_accuracies_mini.png")

if(not is_train_net):
    ###Load Saved Weights###
    model.load_state_dict(torch.load("network_weights_mini.pt"))
    file_write("model loaded")

#Evaluate accuracy of model,
(test_accuracy,unknown_accuracy) = accuracy(train_dict,test_dict,unknown_list,model)
file_write("osnn test_accuracy")
file_write(test_accuracy)
file_write("osnn unknown_accuracy")
file_write(unknown_accuracy)

#Evaluate accuracy with simple nearest neighbour
simple_accuracy = accuracy_nn(train_dict,test_dict,unknown_list,model)
file_write("nn accuracy")
file_write(simple_accuracy)

# #Evaluate accuracy with topk simple nearest neighbour
with torch.no_grad():
    simple_accuracy_topk = accuracy_nn_topk(train_dict,test_dict,unknown_list,model,5)
    file_write("topk nn accuracy")
    file_write(simple_accuracy_topk)

# TSNE Plot
#(train_embeddings,train_ids,test_embeddings,test_ids,unknown_embeddings) = calculate_embeddings(train_dict,test_dict,unknown_list,model) 
#torch.save(train_embeddings,"plotEmbeddings_mini.pt")
#torch.save(train_ids,"plotIDs_mini.pt")
#print("calculated embeddings")
# train_embeddings = torch.load("plotEmbeddings_mini.pt")
# train_ids = torch.load("plotIDs_mini.pt")
# print("loaded embs")
# plot_tsne(train_embeddings,train_ids)
# print("plotted graphs")





















