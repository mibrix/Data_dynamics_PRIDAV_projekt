import pathlib
import os
import cv2
from tqdm import tqdm


def list_images_full():
    return list(os.listdir(basepath.joinpath("photos")))


def download_haarcascade():
    import urllib.request
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, basepath.joinpath("haarcascade_frontalface_default.xml"))


def process_image(image, scale=1.1, minNeighbors=5, minSize=(30, 30)):
    """
    Preprocess images with face detection and center the faces
    :param image: image to process
    :param scale: hyperparameter for face detection (see OpenCV documentation)
    :param minNeighbors:  hyperparameter for face detection (see OpenCV documentation)
    :param minSize: hyperparameter for face detection (see OpenCV documentation)
    :return: list of faces sorted by size - the biggest face is the first (should be the true positive)
    """
    # load the image using OpenCV, convert it to grayscale
    image = cv2.imread(str(basepath.joinpath("photos", image)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # I would resize them but someone at the website admin team did it for me
    # But I would do it like this:
    # gray = cv2.resize(gray, (300, 200))

    # # There are some cooler algorithms like matching 3d models of faces and then centering them # #
    # # Empirical evidence shows that the Haar Cascade is sufficient for this task on this dataset# #

    # detect faces in the grayscale image
    rects = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=minNeighbors, minSize=minSize)
    # there should be only one face in the image but just in case
    faces = []
    for (x, y, w, h) in rects:
        # crop the face
        face = gray[y:y + h, x:x + w]
        # add the face to the list
        faces.append(face)

    # heuristics for picking the biggest face - most likely to be the true positive
    faces.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    # resize the faces to 100x100
    faces_resized = [cv2.resize(face, (100, 100)) for face in faces]
    return faces_resized


def process_images():
    # get image names
    images = list_images_full()
    # iterate over images
    for image in tqdm(images):
        # process the image, reffer to the function documentation
        faces = process_image(image)
        # save the faces
        savepath = basepath.joinpath("faces", image)
        # we know that there is only one face in the image
        # and if there is more than one  it's a false positive
        # in case there is no face
        if len(faces) > 0:  # This is a failsafe for unknown.jpg - the anonymous face image of our faculty
            cv2.imwrite(str(savepath), faces[0])


if __name__ == "__main__":
    basepath = pathlib.Path(__file__).parent.absolute()
    # check if the haarcascade is downloaded, since this is a repo it should be
    if not basepath.joinpath("haarcascade_frontalface_default.xml").exists():
        download_haarcascade()
    # Create the face cascade classifier
    face_cascade = cv2.CascadeClassifier(str(basepath.joinpath("haarcascade_frontalface_default.xml")))

    process_images()


