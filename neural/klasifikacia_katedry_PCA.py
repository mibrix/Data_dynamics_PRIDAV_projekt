import sqlite3
import numpy as np
import numpy.random
import tensorflow as tf





con = sqlite3.connect('../scraping/database.db')
data = con.execute('SELECT * FROM important').fetchall()
labels = con.execute('SELECT * FROM labels').fetchall()

con.close()

pca = np.load('../PCA.npz')

print(pca['weights'].flatten().min(),pca['weights'].flatten().max())

numpy.random.shuffle(data)
t_labels = []
t_photos = []
for per in data:
    for label in labels:
        if per[0] == label[1]:
            t_labels.append(label[0] - 1)
            index = np.where(pca['faces_names'] == per[1])
            index = index[0][0]
            t_photos.append(list(pca['weights'][index]))





t_labels = np.array(t_labels)
t_photos = np.array(t_photos)
train_photos = t_photos[:250]
train_labels = t_labels[:250]
test_photos = t_photos[250:]
test_labels = t_labels[250:]

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[25]),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(60,activation='sigmoid'),
    tf.keras.layers.Dense(50,activation='sigmoid'),
    tf.keras.layers.Dense(20,activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


model.fit(train_photos, train_labels, epochs=600)

test_loss, test_acc = model.evaluate(test_photos,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])