import matplotlib.pyplot as plt
import os
from skimage.io import imread_collection

if __name__ == '__main__':
    os.chdir('/Users/gemenenarcis/Documents/Machine-Learning/Dogs-vs-Cats')
    print(os.getcwd())
    catsImageList = imread_collection(os.getcwd() + '/train/cats/*.jpg')
    dogsImageList = imread_collection(os.getcwd() + '/train/dogs/*.jpg')
    print(len(catsImageList))
    print(len(dogsImageList))
    #plt.imshow(catsImageList[1])
    #plt.imshow(dogsImageList[1])
    print(1)