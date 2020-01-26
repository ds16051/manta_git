import pandas as pd

#labels_frame = pd.read_json("mantaAnnotations.json")
labels_frame = pd.read_json("~/Documents/mastersProject/dataSetOne/mantaAnnotations.json")
print(labels_frame["annotations"][1]["individualId"])