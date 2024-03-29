# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the input image")
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image",
                help="ooutput filename prefix")
args = vars(ap.parse_args())

# load the input image and convert it to numpy array, and then reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# initialize the image generator for data augmentation then initialize the total number of images generated so far
aug = ImageDataGenerator(rotation_range = 30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                        zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
total = 0

# construct the actual python generator
print("[INFP] generating image...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                    save_prefix=args["prefix"], save_format="jpg")

# loop over examples froom our image data augmentation generatorr
for image in imageGen:
    # increment our counter
    total+=1

    # if we have reached 10 examples break from the loop
    if total==10:
        break
