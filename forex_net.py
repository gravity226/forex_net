# Net framework copied from:
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from sklearn.cross_validation import train_test_split

import json

def get_data():
    # files = [   "AUDJPY_20110606_01-00-00.json",	"GBPJPY_20100819_19-00-00.json",
    #             "AUDJPY_20130822_12-00-00.json",	"GBPJPY_20140923_11-00-00.json",
    #             "AUDUSD_20081229_03-00-00.json",	"GBPUSD_20040810_11-00-00.json",
    #             'CHFJPY_20030501_23-00-00.json',	"GBPUSD_20080229_20-00-00.json",
    #             "CHFJPY_20050713_03-00-00.json",	'GBPUSD_20131008_23-00-00.json',
    #             "CHFJPY_20050916_19-00-00.json",	"NZDUSD_20030314_06-00-00.json",
    #             "CHFJPY_20071005_03-00-00.json",	"NZDUSD_20060425_02-00-00.json",
    #             "CHFJPY_20090513_23-00-00.json",	"USDCAD_20010703_05-00-00.json",
    #             "EURCHF_20011023_01-00-00.json",	"USDCAD_20080902_03-00-00.json",
    #             "EURCHF_20070809_09-00-00.json",	"USDCAD_20081224_20-00-00.json",
    #             "EURCHF_20081210_19-00-00.json",	"USDCAD_20090416_15-00-00.json",
    #             "EURCHF_20140613_02-00-00.json",	'USDCAD_20110531_22-00-00.json',
    #             'EURGBP_20061205_17-00-00.json',	"USDCHF_20020531_10-00-00.json",
    #             "EURGBP_20130411_16-00-00.json",	"USDCHF_20021211_12-00-00.json",
    #             "EURJPY_20020902_11-00-00.json",	'USDCHF_20031127_01-00-00.json',
    #             "EURJPY_20130107_03-00-00.json",	"USDCHF_20050819_18-00-00.json",
    #             "EURJPY_20130404_08-00-00.json",	"USDCHF_20110221_09-00-00.json",
    #             "EURJPY_20140124_10-00-00.json",	"USDCHF_20110317_05-00-00.json",
    #             "EURUSD_20041231_15-00-00.json",	'USDCHF_20110519_04-00-00.json',
    #             "EURUSD_20061026_19-00-00.json",	'USDJPY_20120314_08-00-00.json',
    #             "EURUSD_20080208_18-00-00.json",	"USDJPY_20141205_13-00-00.json",
    #             "GBPCHF_20020215_07-00-00.json",	'XAGUSD_20111220_14-00-00.json',
    #             "GBPCHF_20040504_09-00-00.json",    "XAUUSD_20021016_05-00-00.json",
    #             "GBPCHF_20140103_11-00-00.json",	'XAUUSD_20071010_03-00-00.json',
    #             "GBPJPY_20030707_09-00-00.json",	"XAUUSD_20080623_03-00-00.json",
    #             "GBPJPY_20060321_04-00-00.json",	"XAUUSD_20090903_05-00-00.json",
    #             "GBPJPY_20060727_00-00-00.json",	"XAUUSD_20120214_18-00-00.json",
    #             "GBPJPY_20071120_13-00-00.json",	"XAUUSD_20130412_19-00-00.json"    ]

    df = pd.read_csv('json_files.csv')
    files = list(df['0'])

    X = []
    y = []
    count = 0
    print "load files..."
    for f in files:
        # print f
        data = {}
        with open(f) as data_file:
            try:
                data = json.load(data_file)
            except:
                print "None for", f

        # data.keys() = [u'dif', u'data_range', u'filename', u'vector', u'pair', u'close', u'class']
        X.append(data['vector'])
        y.append(data['class'])

        print "%s of %s" % (str(count), str(len(files)))
        count += 1

    x_new = []
    count = 0
    print "reshape files..."
                    # X.shape = (56, 120, 160, 4); when given 56 images
    for img in X:       # img.shape = (120, 160, 4)
        temp_colors = []
        for color in xrange(3):   # need to change into a shape = (4, 120, 160); 4 = (r, g, b, alpha)
            temp_rows = []
            for row in img:     # row.shape = (160, 4)
                temp_cols = []
                for col in row:
                    temp_cols.append(col[color])
                temp_rows.append(temp_cols)
            temp_colors.append(temp_rows)
        x_new.append(temp_colors)

        print "%s of %s" % (str(count), str(len(X)))
        count += 1

    x_new = np.array(x_new)
    # x_new.shape = (56, 3, 120, 160)

    return x_new, y


if __name__ == '__main__':
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print "create net..."
    batch_size = 32
    nb_classes = 3
    nb_epoch = 10
    data_augmentation = True

    img_rows, img_cols = 120, 160
    # the CIFAR10 images are RGB
    img_channels = 3

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # net it!
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(  loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy']) # , metrics=['accuracy']

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('Using real-time data augmentation.')
    print "train net..."
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test)) #X_train.shape[0]















#
