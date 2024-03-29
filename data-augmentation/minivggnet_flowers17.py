# import the sklearn package
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images to describe and extract the label names from image paths
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessor
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

# partition the data into training and testing splits using 75% for trraining and remaining 25% for testing
(trainx, testx, trainy, testy) = train_test_split(data, labels, test_size = 0.25, random_state=42)

# convert the labels from integers to vectors
trainy = LabelBinarizer().fit_transform(trainy)
testy = LabelBinarizer().fit_transform(testy)

# initialize the optimizer and the model
print("[INFO] compiling the model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
H = model.fit(trainx, trainy, validation_data=(testx, testy), batch_size=32, epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testx)
print(classification_report(testy.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="training loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0, 100), H.history["accuracy" if "accuracy" in H.history.keys() else "acc"], label="training accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy" if "val_accuracy" in H.history.keys() else "val_acc"], label="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("minivggnet_flowers17.jpg")
plt.show()