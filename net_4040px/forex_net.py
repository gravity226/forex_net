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

def get_data(files, n):

    count = n
    # print "load files..."
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

        # print "%s of %s" % (str(count), str(len(files)))
        count += 1

    return X, y

if __name__ == '__main__':

    # build net
    print "create net..."
    batch_size = 32
    nb_classes = 3
    data_augmentation = True

    img_rows, img_cols = 40, 40
    # the CIFAR10 images are RGB
    img_channels = 3

    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

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


    # get data and train in batches
    df = pd.read_csv('json_files.csv')
    files = list(df['0'])

    count = 0
    batch_size = 1000
    nb_epoch = 5

    for e in xrange(nb_epoch):
        acc = []
        loss = []

        # loading ticks
        ticks = 0
        print ''
        print "-----------------------"
        print "Epoch", e
        for n in xrange(0, len(files), batch_size):
            if n >= float(ticks) / len(files):
                print "X=",
                ticks += 10

            X = []
            y = []
            if n + batch_size < len(files):
                X, y = get_data(files[n:n+batch_size], n)
            else:
                X, y = get_data(files[n:], n)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # convert class vectors to binary class matrices
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            Y_test = np_utils.to_categorical(y_test, nb_classes)

            model.train_on_batch(X_train, Y_train)
            l, a = model.test_on_batch(X_test, Y_test, accuracy=True)

            acc.append(a)
            loss.append(l)
        print ''
        print "Val_loss", (sum(loss) / len(loss))
        print "Val_acc", (sum(acc) / len(acc))

    # with random batch draws....
    # -------
    # Epoch 1
    # Val_loss 0.889370705837
    # Val_acc 0.479363625794

    # -------
    # Epoch 2
    # Val_loss 0.892368757026
    # Val_acc 0.475923728484

    # -------
    # Epoch 3
    # Val_loss 0.889994844194
    # Val_acc 0.478411244931

    # -------
    # Epoch 4
    # Val_loss 0.889986013536
    # Val_acc 0.484586393495


    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255

    # print('Using real-time data augmentation.')
    # print "train net..."
    # # this will do preprocessing and realtime data augmentation
    # datagen = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    #
    # # compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied)
    # datagen.fit(X_train)
    #
    # # fit the model on the batches generated by datagen.flow()
    # model.fit_generator(datagen.flow(X_train, Y_train,
    #                     batch_size=batch_size),
    #                     samples_per_epoch=X_train.shape[0],
    #                     nb_epoch=nb_epoch,
    #                     validation_data=(X_test, Y_test)) #X_train.shape[0]















#
