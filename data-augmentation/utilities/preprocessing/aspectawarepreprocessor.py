# import the necessary package
import imutils
import cv2

class AspectAwarePreprocessor():
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # storre the target image width, height and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # grab tne dimensions of the image and then initialize the deltas to use when cropping
        (h, w) = image.shape[:2]
        dw = 0
        dh = 0

        # if the width is smallerr than height then resie along width else resize along height (i.e resize along smaller dimension)
        # then update the deltas to crop the height/width (i.e the longer dimension) to the desired dimension
        if w<h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dh = int((image.shape[1] - h)/2.0)

        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dw = int((image.shape[0] - h)/2.0)

        # now that the images have been resized, re-grab the width and height and crop the image
        (h, w) = image.shape[:2]
        image = image[dh:h - dh, dw:w - dw]

        # finally, resize the image to the spatial dimenstions to ensure the output image is always a fixed size
        return  cv2.resize(image, (self.width, self.height), interpolation = self.inter)