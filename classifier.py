import pandas as pd
import numpy as np

"""
Once the triplet loss network is trained, it will project images into an embedding space, in which euclidean distance is a measure of similarity: images of the same manta have close embeddings, while images of different mantas have distant embeddings.  
There are then a number of options for actual classification:
    1.) Train an SVM, though this is not really open set
    2.) Use k-nearest-neighbour, which can be slow, but is compatible with open set
Let's think about k-nn approaches. Maybe we use the training set to compute some embeddings, and store these in a database along with the labels. Then, for classification, we pass an image through the net, and compare it's embedding to those in the database using k-nn.
Maybe, if we have multiple embeddings for an individual in teh database, we can store a single average point, and compare to this. Then when a new individual is passed through, it is classified as the nearest average point, or as a new individual if it is further than some threshold. 
"""

"""
The open set scenario is one in which there are no a priori training samples for some classes that might appear during testing. For an example, we must return:
    1.) "this sample us unknown" or
    2.) "this sample is known and has identity x"
"""


"""
Open-Set-Nearest-Neighbour
#https://link.springer.com/content/pdf/10.1007/s10994-016-5610-8.pdf

we have stored the training data ( a large number of labelled embeddings) without any clustering/averaging
we need to classify  the embedding s of a test sample
1.) find the nearest neighbour t of s
2.) find the nearest neighbour u of s, where u has different label to t
3.) calculate ratio R = d(s,t)/d(s,u), where d is euclidean
4.) If R is less than or equal to some threshold T, then s is classified with the label of t. Otherwise it is classified as unknown

The threshold T is between 0 and 1.
An optimisation procedure can be used to select T
We have a training set of labelled embeddings. 
Among the _classes_ that occur in the training set, half are chosen to act as “known” classes in, the other half as “unknown”.
The training set is divided into a fitting set F that contains half for the instances of the “known” classes, and a validation set V that contains the other half of the instances of the “known” classes, and all instances of the “unknown” classes.
Once the sets F and V are defined as described, we try out values of T, and choose the best.
"""

"""
Simpler starting point (and faster) solutii=in that osnn

Average all stored embeddings for each class, and assign a test sample to the closest, but with a threshold for assigning unknown
"""

### OSNN Test using Iris dataset

