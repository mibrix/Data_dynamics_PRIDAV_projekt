import pathlib
import os
import cv2
import numpy as np

basepath = pathlib.Path(__file__).parent.absolute()


def list_faces():
    return list(os.listdir(basepath.joinpath("faces")))


def load_face(filename):
    face = cv2.imread(str(basepath.joinpath("faces", filename)),0)
    # flatten the image
    face = face.flatten()
    return face


def average_face(faces_v, size):
    # initialize the average face vector
    zero_v = np.zeros(size)
    # sum all the face vectors
    for face_v in faces_v:
        zero_v += face_v
    # divide by the number of faces to get the average
    avg_v = zero_v / len(faces_v)
    # initialize the list of normalized face vectors
    norm_faces_v = []

    for face_v in faces_v:
        # subtract the average from each face vector to get the normalized face vectors
        norm_face_v = face_v - avg_v
        # and add them to the list
        norm_faces_v.append(norm_face_v)

    # Return the list of normalized face vectors, and the average face vector
    return norm_faces_v, avg_v


if __name__ == "__main__":
    # List the directory with faces, store the names of the files
    faces_names = list_faces()
    # Load the faces, and flatten them  store them in a list
    faces_v = [load_face(face) for face in faces_names]
    # If you want to split the faces into training and testing, do it here
    # Compute the average face, and the list of normalized faces
    # They are in the same order as the faces_names list
    norm_faces_v, avg_v = average_face(faces_v, faces_v[0].shape)
    # stack them in a matrix
    faces_m = np.stack(norm_faces_v)
    # Compute the covariance matrix
    #cov_m = np.cov(faces_m)

    cov_m = np.dot(faces_m.T, faces_m)

    # Compute the eigenvalues and eigenvectors
    compute_eig = False  # it takes like 5 min on 8+8 cores don't do it if you don't have to
    if compute_eig:
        eigenvalues, eigenvectors = np.linalg.eig(cov_m)
        # save the eigenvalues and eigenvectors
        np.savez_compressed("eigen.npz", eigenvalues=eigenvalues.real, eigenvectors=eigenvectors.real)
    else:
        eigenvalues, eigenvectors = np.load("eigen.npz")["eigenvalues"], np.load("eigen.npz")["eigenvectors"]

    # Sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    # select the first 10 eigenvalues and eigenvectors
    k = 25
    keigenvalues = eigenvalues[idx][:k]
    keigenvectors = eigenvectors[:, idx][:, :k]
    # Stack the eigenvectors into a matrix instead of a list of arrays
    keigenvectors = np.stack(keigenvectors).real
    # normalize the eigenvectors
    keigenvectors = keigenvectors / np.linalg.norm(keigenvectors, axis=0)
    # Should be representation of the faces with linear combination of 10 eigen vectors
    weights = np.dot(faces_m, keigenvectors)
    # Save the weights, the average face, and the eigenvalues and eigenvectors
    np.savez("PCA.npz",
             weights=weights,
             avg_v=avg_v,
             keigenvectors=keigenvectors,
             faces_names=faces_names)

    # reconstructing the faces from the weights and the eigen vectors
    # multiply the weights with the eigen vectors
    reconstructed_faces = np.dot(weights, keigenvectors.real.T)
    # add the average face
    reconstructed_faces = reconstructed_faces + avg_v
    # reshape the faces to 2D 100x100
    reconstructed_faces = reconstructed_faces.reshape(-1, 100, 100)
    # convert the faces to uint8, so they can be saved as images
    reconstructed_faces = reconstructed_faces.astype(np.uint8)
    # save the reconstructed faces into the reconstructed folder
    for i, face in enumerate(reconstructed_faces):
        path = basepath.joinpath("reconstructed", faces_names[i])
        cv2.imwrite(str(path), face)




