import numpy as np
import csv
from unidecode import unidecode
import re
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


eigenfaces = np.load("pca.npz")

#map vector to  a name
#name->(filename/index,num_of_degrees)
###
deg_map = dict()
with open("./scraping/fakulty.csv","r",encoding="utf8") as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row!=[]:
            name_degrees = row[0]
            degrees = name_degrees.count(".")
            name_degrees = name_degrees.split(" ")
            name = row[1]
            deg_map[name] = degrees

#add index of filenme so it can be used for training
faces_filenames = eigenfaces["faces_names"]
for filename in faces_filenames:
    if filename in deg_map:
        #only looking for the first instance of one filename
        if isinstance(deg_map[filename],int):
            deg_map[filename] = (deg_map[filename],np.where(faces_filenames==filename)[0][0])
face_vectors = eigenfaces["weights"]


usable_faces = np.empty((0,face_vectors.shape[1]+1))

#stack face vectors on top of each other
for name in deg_map:
    if isinstance(deg_map[name],tuple):
        degrees = deg_map[name][0]
        ind = deg_map[name][1]
        face_degrees = np.append(face_vectors[ind],degrees)
        usable_faces = np.vstack((usable_faces,face_degrees))






#use random forest to train data
X_train, X_test, Y_train, Y_test = train_test_split(usable_faces[:,:-1],usable_faces[:,-1],random_state=666,train_size=0.75)


#Use oversampling
ros = RandomOverSampler(sampling_strategy={0:42,1:42})
X_train, Y_train = ros.fit_resample(X_train,Y_train)
print(Counter(Y_train))

#Hyperparameter search

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}



# rf = RandomForestClassifier(random_state=4646)
# rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,n_jobs = -1)
# rf_random.fit(X_train, Y_train)
# print(rf_random.best_params_)

#{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, '
# max_features': 'sqrt', 'max_depth': 80, 'bootstrap': False}


r_forest = RandomForestClassifier(random_state=4646,n_estimators= 200, min_samples_split= 2, min_samples_leaf= 4, max_features= 'sqrt', max_depth= 80, bootstrap= False)
r_forest.fit(X_train,Y_train)
ypred = r_forest.predict(X_test)
cr = classification_report(Y_test,ypred)
print(cr)
# ##
# accuracy: 0.49 