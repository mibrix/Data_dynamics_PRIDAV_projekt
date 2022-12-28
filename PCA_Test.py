import numpy as np
import cv2
import pathlib

from Preprocessing import process_image, download_haarcascade

basepath = pathlib.Path(__file__).parent.absolute()
# check if the haarcascade is downloaded, since this is a repo it should be
if not basepath.joinpath("haarcascade_frontalface_default.xml").exists():
    download_haarcascade()
# Create the face cascade classifier
face_cascade = cv2.CascadeClassifier(str(basepath.joinpath("haarcascade_frontalface_default.xml")))

data = np.load("PCA.npz")
weights = data["weights"]
avg_v = data["avg_v"]
keigenvectors = data["keigenvectors"]
faces_names = data["faces_names"]

# Load the image to test
test_image = "vinar1.jpg"
# grayscales the image, detects the face and centers it and resizes it to 100x100 pixels
processed_image = process_image(str(basepath.joinpath("test_photos", test_image)), face_cascade, basepath)
base_processed_image = processed_image.copy()

# Flatten the image
processed_image = processed_image[0].flatten()

# Subtract the average face
processed_image_avg_v = processed_image - avg_v

# project the image into the eigen space
test_weights = np.dot(processed_image_avg_v.T, keigenvectors.real)
# convert to real because the imaginary part is just 0 anyway

# Compute the distance between the test image and the training images
distances = np.linalg.norm(weights - test_weights, axis=1, ord=np.inf)

# Get the index of the 5 minimum distances
min_indexes = np.argsort(distances)[:5]

# Get the names of the 5 closest faces
closest_faces = [faces_names[i] for i in min_indexes]

# Display the 5 closest faces with distances
for i, face in enumerate(closest_faces):
    print(f"{face} - {distances[min_indexes[i]]}")

# find the index of the real face
try:
    real_face_index = np.where(faces_names == test_image)[0][0]
except IndexError:
    print("The real face is not in the training set or something went wrong")
    real_face_index = None

print(f"The real face is {test_image}")
print(f"The real face is at predicted index {real_face_index}")
# print distances of the real face
if real_face_index is not None:
    print(f"The distance of the real face is {distances[real_face_index]}")

import matplotlib.pyplot as plt
realone = cv2.imread(str(basepath.joinpath("reconstructed", test_image)))

# display also the reconstructed images
# 2x2 grid, 1st subplot
fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout="tight")
# reconstruct the image that has been predicted
reconstructed_image = np.dot(test_weights, keigenvectors.real.T) + avg_v

ax[0, 0].imshow(reconstructed_image.astype(np.uint8).reshape(100, 100), cmap="gray")
ax[0, 0].set_title("Test image - {}".format(test_image))
ax[0, 1].imshow(cv2.imread(str(basepath.joinpath("reconstructed", faces_names[min_indexes[0]]))), cmap="gray")
ax[0, 1].set_title("Predicted image - {}".format(faces_names[min_indexes[0]]))
ax[1, 0].imshow(realone, cmap="gray")
ax[1, 0].set_title("Training image for that person - {}".format(test_image))
ax[1, 1].imshow(cv2.imread(str(basepath.joinpath("reconstructed", faces_names[min_indexes[2]]))), cmap="gray")
ax[1, 1].set_title("2rd predicted image - {}".format(faces_names[min_indexes[1]]))

plt.suptitle("Reconstructed images")
plt.show()
