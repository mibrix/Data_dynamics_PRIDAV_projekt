import cv2
import sqlite3
import numpy as np
import numpy.random
import tensorflow as tf





con = sqlite3.connect('../scraping/database.db')
data = con.execute('SELECT * FROM important').fetchall()
labels = con.execute('SELECT * FROM labels').fetchall()

con.close()




numpy.random.shuffle(data)
t_labels = []
t_photos = []
for per in data:
    for label in labels:
        if per[0] == label[1]:
            t_labels.append(label[0] - 1)
            t_photos.append(cv2.resize(cv2.imread("../reconstructed/"  + per[1],cv2.IMREAD_GRAYSCALE),(50,50), interpolation = cv2.INTER_AREA))



print(len(t_photos),len(t_photos[0]))
t_labels = np.array(t_labels)
t_photos = np.array(t_photos)
t_photos = t_photos / 255
train_photos = t_photos[:250]
train_labels = t_labels[:250]
test_photos = t_photos[250:]
test_labels = t_labels[250:]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(50, 50)),
    tf.keras.layers.Dense(1000, activation='sigmoid'),
    tf.keras.layers.Dense(500, activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


model.fit(train_photos, train_labels, epochs=150)

test_loss, test_acc = model.evaluate(test_photos,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])