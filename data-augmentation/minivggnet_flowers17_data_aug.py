# import the sklearn package
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn import MiniVGGNet
from keras.preprrocessing.image import ImageDataGenerator
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

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and the model
print("[INFO] compiling the model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(aug.flow(trainx, trainy, batch_size=32),
              validation_data=(testx, testy), steps_per_epoch=len(trainx)//32, epochs=100, verbose=1)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testx)
classification_report(testy.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], labels="training loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], labels="validation loss")
plt.plot(np.arange(0, 100), H.history["acc"], labels="training accuracy")
plt.plot(np.arange(0, 100), H.history["val_acc"], labels="validation accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.save("minivggnet_flowers17_data_aug.jpg")
plt.show()