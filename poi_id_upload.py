#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer, recall_score, accuracy_score, precision_score, roc_auc_score

sys.path.append("../tools/")

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Store to my_dataset for easy export below.
my_dataset = data_dict
del(my_dataset["TOTAL"]) #Delete the outlier

interim_features_list = ["poi"] 

count = 0
for i in data_dict:
    while count == 0:
        for j in data_dict[i]:
            if j in ['email_address', 'poi']: #exclude email, and skip poi
                pass
            else:
                interim_features_list.append(j)
        count +=1

features_to_keep = [5,6,10,13] #A result of looping through the importances and eliminating 
                                #features that weren't adding value

new_list = ["poi"]

for i,j in enumerate(interim_features_list):
    if i in features_to_keep:
        new_list.append(j)
    else:
        pass   
     
features_list = new_list #Incorporating the result of the feature selection into the final list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42).fit(features_train,labels_train)

##Figure out which features have zero importance. Used for initial feature selection.
count = 0
importance_list = []
for i in clf.feature_importances_:
    importance_list.append([features_to_keep[count],interim_features_list[features_to_keep[count]],i])
    count +=1
importance_df = pd.DataFrame(importance_list,columns=['original_index','feature','feature_importance'])
print(importance_df)

dump_classifier_and_data(clf, my_dataset, features_list)