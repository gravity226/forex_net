from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import io

import matplotlib.pyplot as plt

img = imread('imgs/EURUSD20010103_00-00-00.png')

resize_img = resize(img, (120,160)) # 1/8 original size

io.imshow(resize_img)

plt.show()














#
