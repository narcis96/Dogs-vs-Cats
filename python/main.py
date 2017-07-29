import matplotlib.pyplot as plt
import os
from skimage.io import imread_collection
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import numpy as np
import cv2
import progressbar
from random import randint
import ProgressBar
from PIL import Image

from PIL import ImageFilter
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
BAR_LENGTH = 300
def ResizeImageList(images, sizeX, sizeY):
    resizedImages = []
    length = len(images)
    bar = progressbar.ProgressBar(maxval = length)
    show = [randint(0, BAR_LENGTH) for i in range(length)]
    for indx in range(length):
        image = cv2.resize(images[indx], dsize=(sizeX, sizeY), interpolation=cv2.INTER_CUBIC)
        resizedImages.append(image)
        if show[indx] == 0:
            bar.update(indx + 1)
    bar.finish()
    return resizedImages

def BlurImageList(images, sizeX, sizeY):
    bluredImages = []

    length = len(images)
    bar = progressbar.ProgressBar(maxval = length)
    show = [randint(0, BAR_LENGTH) for i in range(length)]

    for indx in range(length):
        img = cv2.GaussianBlur(images[indx], (sizeX, sizeY), 0)
        bluredImages.append(img)
        if show[indx] == 0:
           bar.update(indx + 1)
    bar.finish()
    return bluredImages

def GrayImages(images):
    grayimg = []

    length = len(images)
    bar = progressbar.ProgressBar(maxval = length)
    show = [randint(0, BAR_LENGTH) for i in range(length)]

    for indx in range(len(images)):
        img  = cv2.cvtColor(images[indx], cv2.COLOR_BGR2GRAY)
        #img = np.asarray(Image.fromarray(images[indx]).convert('LA'))
        grayimg.append(img)
        if show[indx] == 0:
            bar.update(indx + 1)
    bar.finish()
    return grayimg


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def EdgeDetection(images, sigma = 0.33):
    edgedImages = []
    length = len(images)
    bar = progressbar.ProgressBar(maxval = length)
    show = [randint(0, BAR_LENGTH) for i in range(length)]

    for indx in range(len(images)):
        img  = auto_canny(images[indx])
        edgedImages.append(img)
        if show[indx] == 0:
            bar.update(indx + 1)
    bar.finish()
    return edgedImages

if __name__ == '__main__':
    os.chdir('/Users/gemenenarcis/Documents/Machine-Learning/Dogs-vs-Cats')
    print(os.getcwd())
    catsImageList = imread_collection(os.getcwd() + '/train/cats/*.jpg')
    dogsImageList = imread_collection(os.getcwd() + '/train/dogs/*.jpg')
    SIZE_X = 200
    SIZE_Y = 200
    BLUR_SIZE = 7
    catsImageList = ResizeImageList(images =  catsImageList,sizeX = SIZE_X,sizeY = SIZE_Y)
    dogsImageList = ResizeImageList(images =  dogsImageList,sizeX = SIZE_X,sizeY = SIZE_Y)

    catsImageList = BlurImageList(images = catsImageList, sizeX = BLUR_SIZE, sizeY = BLUR_SIZE)
    dogsImageList = BlurImageList(images = dogsImageList, sizeX = BLUR_SIZE, sizeY = BLUR_SIZE)

    catsImageList = GrayImages(images = catsImageList)
    dogsImageList = GrayImages(images = dogsImageList)

    catsImageList = EdgeDetection(images = catsImageList)
    dogsImageList = EdgeDetection(images = dogsImageList)

    print(len(catsImageList))
    print(len(dogsImageList))
    #img = Image.fromarray(dog2).convert('LA)
    #img = cv2.GaussianBlur(dog2,(3,3),0)
    #plt.imshow(catsImageList[1])
    #plt.imshow(dogsImageList[1])
    classes = 2
    model = Sequential()
    model.add(Convolution2D(3, 3, padding="same",
                            input_shape = (SIZE_X, SIZE_Y, 1), strides = 10))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    cat = catsImageList[1]
    dog = dogsImageList[1]
    train = np.array([cat, cat])
    labels = np.array([0, 1])
    model.fit(train, labels, batch_size=128, epochs=20, verbose=1)

    print(1)