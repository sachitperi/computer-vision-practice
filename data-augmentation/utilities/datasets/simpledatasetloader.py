# import necessary package
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    """
    Define the constructor to the SimpleDatasetLoader where we can optionally pass in a list of image preprocessors (such as SimpleProceessor) that can
    be sequentially applied to a given image.
    Specifying these preprocessors as a list rather than a single value is important -
    There will be times where we first need to resize the image to a fixed size,
    then perform some sort of scaling (such as mean substraction),
    followed by converting the image array to a format suitable for Keras.
    Each of this preprocessors can be implimented independently, allowing us to apply them sequentially to an image in an efficient manner.
    """
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if preprocessors are none initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    """
    Load method the core of SimpleDatasetLoader
    """
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        #loop over input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming that our path has the following format
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to se if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessor and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our preprocessed image as a `feature vector`by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1)%verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))