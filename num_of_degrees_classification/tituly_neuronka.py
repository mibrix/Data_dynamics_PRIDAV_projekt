import numpy as np
import csv
from unidecode import unidecode
import re
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
# TensorFlow and tf.keras
import tensorflow as tf


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






#use neural network to train data
X_train, X_test, Y_train, Y_test = train_test_split(usable_faces[:,:-1],usable_faces[:,-1],random_state=666,train_size=0.75)

#napr pouzijeme sequential
#flattenovat nemusime to za nas robi PCA 
print(X_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(165, activation='relu'),
    tf.keras.layers.Dense(41, activation='sigmoid'),
    tf.keras.layers.Dense(4)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=100)
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print(test_acc)
