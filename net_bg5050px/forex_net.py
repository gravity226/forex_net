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
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam

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

    img_rows, img_cols = 50, 50
    # the CIFAR10 images are RGB
    img_channels = 3

    # net it!
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    adagrad = Adagrad(lr=0.01, epsilon=1e-06)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(  loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy']) # , metrics=['accuracy']


    # get data and train in batches
    df = pd.read_csv('json_files.csv')
    files = list(df['0'])
    X_files = 0

    count = 0
    batch_size = 32
    nb_epoch = 12

    print ''

    for e in xrange(nb_epoch):
        acc = []
        loss = []

        for n in xrange(0, len(files), batch_size):

            X = []
            y = []
            if n + batch_size < len(files):
                X, y = get_data(files[n:n+batch_size], n)
            else:
                X, y = get_data(files[n:], n)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # , random_state=0
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # convert class vectors to binary class matrices
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            Y_test = np_utils.to_categorical(y_test, nb_classes)

            model.train_on_batch(X_train, Y_train)
            l, a = model.test_on_batch(X_test, Y_test)

            acc.append(a)
            loss.append(l)
        print "Epoch: %s <==> Val_loss: %s <> Val_acc: %s "% ( str(e), str(sum(loss) / len(loss)), str(sum(acc) / len(acc)) )

'''
fill between on open and close included:
Epoch: 0 <==> Val_loss: 1.19601917664 <> Val_acc: 0.491666666667
Epoch: 1 <==> Val_loss: 0.895532437166 <> Val_acc: 0.441666666667
Epoch: 2 <==> Val_loss: 0.89461884896 <> Val_acc: 0.408333333333
Epoch: 3 <==> Val_loss: 0.900162645181 <> Val_acc: 0.433333333333
Epoch: 4 <==> Val_loss: 0.908274920781 <> Val_acc: 0.4
Epoch: 5 <==> Val_loss: 0.906020263831 <> Val_acc: 0.45
Epoch: 6 <==> Val_loss: 0.910774775346 <> Val_acc: 0.475
Epoch: 7 <==> Val_loss: 0.922956418991 <> Val_acc: 0.45
Epoch: 8 <==> Val_loss: 0.922944271564 <> Val_acc: 0.483333333333
Epoch: 9 <==> Val_loss: 0.921967669328 <> Val_acc: 0.466666666667
Epoch: 10 <==> Val_loss: 0.930095549424 <> Val_acc: 0.475
Epoch: 11 <==> Val_loss: 0.938101998965 <> Val_acc: 0.466666666667
Epoch: 12 <==> Val_loss: 0.951898054282 <> Val_acc: 0.466666666667
Epoch: 13 <==> Val_loss: 0.960816089312 <> Val_acc: 0.466666666667
Epoch: 14 <==> Val_loss: 0.9526841561 <> Val_acc: 0.466666666667
Epoch: 15 <==> Val_loss: 0.960777707895 <> Val_acc: 0.466666666667
Epoch: 16 <==> Val_loss: 0.978754997253 <> Val_acc: 0.483333333333
Epoch: 17 <==> Val_loss: 0.983075928688 <> Val_acc: 0.491666666667
Epoch: 18 <==> Val_loss: 0.997579725583 <> Val_acc: 0.466666666667
Epoch: 19 <==> Val_loss: 0.987256900469 <> Val_acc: 0.483333333333
Epoch: 20 <==> Val_loss: 1.02322816054 <> Val_acc: 0.475
Epoch: 21 <==> Val_loss: 1.010653313 <> Val_acc: 0.491666666667
Epoch: 22 <==> Val_loss: 1.05489060481 <> Val_acc: 0.458333333333
Epoch: 23 <==> Val_loss: 1.04266812801 <> Val_acc: 0.491666666667
Epoch: 24 <==> Val_loss: 1.06978891293 <> Val_acc: 0.458333333333
'''

'''
same net
no fill between open and close
Epoch: 0 <==> Val_loss: 6.20859068235 <> Val_acc: 0.441666666667
Epoch: 1 <==> Val_loss: 0.899810961882 <> Val_acc: 0.425
Epoch: 2 <==> Val_loss: 0.893043160439 <> Val_acc: 0.466666666667
Epoch: 3 <==> Val_loss: 0.892282859484 <> Val_acc: 0.483333333333
Epoch: 4 <==> Val_loss: 0.895598216852 <> Val_acc: 0.533333333333
Epoch: 5 <==> Val_loss: 0.89265619119 <> Val_acc: 0.516666666667
Epoch: 6 <==> Val_loss: 0.892653481166 <> Val_acc: 0.483333333333
Epoch: 7 <==> Val_loss: 0.894468971093 <> Val_acc: 0.5
Epoch: 8 <==> Val_loss: 0.893688444297 <> Val_acc: 0.508333333333
Epoch: 9 <==> Val_loss: 0.893808023135 <> Val_acc: 0.5
Epoch: 10 <==> Val_loss: 0.901040009658 <> Val_acc: 0.516666666667
Epoch: 11 <==> Val_loss: 0.910143550237 <> Val_acc: 0.491666666667
Epoch: 12 <==> Val_loss: 0.908859066168 <> Val_acc: 0.525
Epoch: 13 <==> Val_loss: 0.90589594245 <> Val_acc: 0.508333333333
Epoch: 14 <==> Val_loss: 0.921455484629 <> Val_acc: 0.525
Epoch: 15 <==> Val_loss: 0.922343460719 <> Val_acc: 0.516666666667
Epoch: 16 <==> Val_loss: 0.926814005772 <> Val_acc: 0.5
Epoch: 17 <==> Val_loss: 0.939403957129 <> Val_acc: 0.533333333333
Epoch: 18 <==> Val_loss: 0.931429207325 <> Val_acc: 0.558333333333
Epoch: 19 <==> Val_loss: 0.936733937263 <> Val_acc: 0.55
Epoch: 20 <==> Val_loss: 0.957925109069 <> Val_acc: 0.516666666667
Epoch: 21 <==> Val_loss: 0.955526471138 <> Val_acc: 0.541666666667
Epoch: 22 <==> Val_loss: 0.977391411861 <> Val_acc: 0.508333333333
Epoch: 23 <==> Val_loss: 0.985169166327 <> Val_acc: 0.525
Epoch: 24 <==> Val_loss: 1.00311927795 <> Val_acc: 0.516666666667
'''

'''
tanh
Epoch: 0 <==> Val_loss: 2.49349290133 <> Val_acc: 0.525
Epoch: 1 <==> Val_loss: 1.94115846157 <> Val_acc: 0.45
Epoch: 2 <==> Val_loss: 1.9603246967 <> Val_acc: 0.533333333333
Epoch: 3 <==> Val_loss: 1.95501136382 <> Val_acc: 0.483333333333
Epoch: 4 <==> Val_loss: 1.95045830409 <> Val_acc: 0.491666666667
Epoch: 5 <==> Val_loss: 1.90899360577 <> Val_acc: 0.466666666667
Epoch: 6 <==> Val_loss: 1.11139219602 <> Val_acc: 0.533333333333
Epoch: 7 <==> Val_loss: 0.933968305588 <> Val_acc: 0.45
Epoch: 8 <==> Val_loss: 0.997361326218 <> Val_acc: 0.475
Epoch: 9 <==> Val_loss: 0.979440804323 <> Val_acc: 0.458333333333
Epoch: 10 <==> Val_loss: 0.96356079181 <> Val_acc: 0.5
Epoch: 11 <==> Val_loss: 0.951821847757 <> Val_acc: 0.558333333333
Epoch: 12 <==> Val_loss: 0.953892874718 <> Val_acc: 0.541666666667
Epoch: 13 <==> Val_loss: 0.946783487002 <> Val_acc: 0.55
Epoch: 14 <==> Val_loss: 0.920016264915 <> Val_acc: 0.533333333333
Epoch: 15 <==> Val_loss: 0.916656323274 <> Val_acc: 0.55
Epoch: 16 <==> Val_loss: 0.923007925351 <> Val_acc: 0.541666666667
Epoch: 17 <==> Val_loss: 0.941884243488 <> Val_acc: 0.55
Epoch: 18 <==> Val_loss: 0.929582238197 <> Val_acc: 0.583333333333
Epoch: 19 <==> Val_loss: 0.95065946579 <> Val_acc: 0.566666666667
Epoch: 20 <==> Val_loss: 0.948031206926 <> Val_acc: 0.541666666667
Epoch: 21 <==> Val_loss: 0.945475016038 <> Val_acc: 0.55
Epoch: 22 <==> Val_loss: 0.960780568918 <> Val_acc: 0.533333333333
Epoch: 23 <==> Val_loss: 0.984136362871 <> Val_acc: 0.508333333333
Epoch: 24 <==> Val_loss: 0.967921594779 <> Val_acc: 0.55
'''

'''
model.add(Convolution2D(50, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('tanh'))
model.add(Convolution2D(25, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

Epoch: 0 <==> Val_loss: 2.65037268003 <> Val_acc: 0.483333333333
Epoch: 1 <==> Val_loss: 1.04564802647 <> Val_acc: 0.508333333333
Epoch: 2 <==> Val_loss: 0.980106723309 <> Val_acc: 0.508333333333
Epoch: 3 <==> Val_loss: 0.962748408318 <> Val_acc: 0.508333333333
Epoch: 4 <==> Val_loss: 0.9614960591 <> Val_acc: 0.5
Epoch: 5 <==> Val_loss: 0.993816761176 <> Val_acc: 0.516666666667
Epoch: 6 <==> Val_loss: 0.945232458909 <> Val_acc: 0.525
Epoch: 7 <==> Val_loss: 0.980689430237 <> Val_acc: 0.533333333333
Epoch: 8 <==> Val_loss: 0.942039775848 <> Val_acc: 0.533333333333
Epoch: 9 <==> Val_loss: 1.02951944272 <> Val_acc: 0.55
Epoch: 10 <==> Val_loss: 0.999126847585 <> Val_acc: 0.508333333333
Epoch: 11 <==> Val_loss: 0.969732866685 <> Val_acc: 0.55
'''

'''
model = Sequential()

model.add(Convolution2D(10, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(5, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

Epoch: 0 <==> Val_loss: 1.00674140453 <> Val_acc: 0.483333333333
Epoch: 1 <==> Val_loss: 1.03126868407 <> Val_acc: 0.55
Epoch: 2 <==> Val_loss: 0.971930086613 <> Val_acc: 0.5
Epoch: 3 <==> Val_loss: 0.913444368045 <> Val_acc: 0.55
Epoch: 4 <==> Val_loss: 0.970669265588 <> Val_acc: 0.491666666667
Epoch: 5 <==> Val_loss: 0.968594714006 <> Val_acc: 0.566666666667
Epoch: 6 <==> Val_loss: 0.93704773585 <> Val_acc: 0.508333333333
Epoch: 7 <==> Val_loss: 0.915577570597 <> Val_acc: 0.508333333333
Epoch: 8 <==> Val_loss: 0.896123727163 <> Val_acc: 0.516666666667
Epoch: 9 <==> Val_loss: 0.893686000506 <> Val_acc: 0.516666666667
Epoch: 10 <==> Val_loss: 0.890780099233 <> Val_acc: 0.516666666667
Epoch: 11 <==> Val_loss: 0.882663520177 <> Val_acc: 0.525
'''

'''
model = Sequential()

model.add(Convolution2D(10, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('tanh'))
model.add(Convolution2D(5, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

Epoch: 0 <==> Val_loss: 1.00152328809 <> Val_acc: 0.5
Epoch: 1 <==> Val_loss: 0.893952512741 <> Val_acc: 0.508333333333
Epoch: 2 <==> Val_loss: 0.894261964162 <> Val_acc: 0.508333333333
Epoch: 3 <==> Val_loss: 0.903226172924 <> Val_acc: 0.533333333333
Epoch: 4 <==> Val_loss: 0.892120802402 <> Val_acc: 0.525
Epoch: 5 <==> Val_loss: 0.87221518755 <> Val_acc: 0.558333333333
Epoch: 6 <==> Val_loss: 0.878659363588 <> Val_acc: 0.516666666667
Epoch: 7 <==> Val_loss: 0.87474702994 <> Val_acc: 0.541666666667
Epoch: 8 <==> Val_loss: 0.888190948963 <> Val_acc: 0.55
Epoch: 9 <==> Val_loss: 0.885904963811 <> Val_acc: 0.558333333333
Epoch: 10 <==> Val_loss: 0.893990564346 <> Val_acc: 0.525
Epoch: 11 <==> Val_loss: 0.879280432065 <> Val_acc: 0.533333333333
'''

'''
model = Sequential()

model.add(Convolution2D(25, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(10, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(5, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

Epoch: 0 <==> Val_loss: 1.41610496044 <> Val_acc: 0.533333333333
Epoch: 1 <==> Val_loss: 1.02745776574 <> Val_acc: 0.416666666667
Epoch: 2 <==> Val_loss: 0.921722356478 <> Val_acc: 0.516666666667
Epoch: 3 <==> Val_loss: 0.902410121759 <> Val_acc: 0.508333333333
Epoch: 4 <==> Val_loss: 0.885208189487 <> Val_acc: 0.508333333333
Epoch: 5 <==> Val_loss: 0.893720901012 <> Val_acc: 0.525
Epoch: 6 <==> Val_loss: 0.914862688382 <> Val_acc: 0.491666666667
Epoch: 7 <==> Val_loss: 0.874021136761 <> Val_acc: 0.533333333333
Epoch: 8 <==> Val_loss: 0.893228522937 <> Val_acc: 0.533333333333
Epoch: 9 <==> Val_loss: 0.885608311494 <> Val_acc: 0.525
Epoch: 10 <==> Val_loss: 0.881706702709 <> Val_acc: 0.533333333333
Epoch: 11 <==> Val_loss: 0.889391855399 <> Val_acc: 0.516666666667
'''

'''
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

Epoch: 0 <==> Val_loss: 1.21659333706 <> Val_acc: 0.45
Epoch: 1 <==> Val_loss: 0.897775725524 <> Val_acc: 0.458333333333
Epoch: 2 <==> Val_loss: 0.907198345661 <> Val_acc: 0.533333333333
Epoch: 3 <==> Val_loss: 0.891705838839 <> Val_acc: 0.541666666667
Epoch: 4 <==> Val_loss: 0.899523874124 <> Val_acc: 0.5
Epoch: 5 <==> Val_loss: 0.901229691505 <> Val_acc: 0.525
Epoch: 6 <==> Val_loss: 0.896088079611 <> Val_acc: 0.55
Epoch: 7 <==> Val_loss: 0.89068479538 <> Val_acc: 0.55
Epoch: 8 <==> Val_loss: 0.89365499417 <> Val_acc: 0.558333333333
Epoch: 9 <==> Val_loss: 0.888176850478 <> Val_acc: 0.558333333333
Epoch: 10 <==> Val_loss: 0.893912355105 <> Val_acc: 0.55
Epoch: 11 <==> Val_loss: 0.89276419878 <> Val_acc: 0.516666666667
'''

'''
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('tanh'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('tanh'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

Epoch: 0 <==> Val_loss: 2.14039910436 <> Val_acc: 0.433333333333
Epoch: 1 <==> Val_loss: 1.38197063208 <> Val_acc: 0.508333333333
Epoch: 2 <==> Val_loss: 1.25456033548 <> Val_acc: 0.483333333333
Epoch: 3 <==> Val_loss: 1.1476768295 <> Val_acc: 0.5
Epoch: 4 <==> Val_loss: 1.11000051896 <> Val_acc: 0.491666666667
Epoch: 5 <==> Val_loss: 1.03455753724 <> Val_acc: 0.533333333333
Epoch: 6 <==> Val_loss: 1.02537195285 <> Val_acc: 0.55
Epoch: 7 <==> Val_loss: 1.04221757253 <> Val_acc: 0.516666666667
Epoch: 8 <==> Val_loss: 1.01274039348 <> Val_acc: 0.533333333333
Epoch: 9 <==> Val_loss: 1.02235184113 <> Val_acc: 0.558333333333
Epoch: 10 <==> Val_loss: 1.05282201568 <> Val_acc: 0.516666666667
Epoch: 11 <==> Val_loss: 1.06919843356 <> Val_acc: 0.541666666667
'''

'''
model = Sequential()

model.add(Convolution2D(9, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(9, 3, 3))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Convolution2D(36, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

Epoch: 0 <==> Val_loss: 1.03291877111 <> Val_acc: 0.516666666667
Epoch: 1 <==> Val_loss: 0.911722147465 <> Val_acc: 0.491666666667
Epoch: 2 <==> Val_loss: 0.890598897139 <> Val_acc: 0.466666666667
Epoch: 3 <==> Val_loss: 0.883446009954 <> Val_acc: 0.458333333333
Epoch: 4 <==> Val_loss: 0.893478163083 <> Val_acc: 0.491666666667
Epoch: 5 <==> Val_loss: 0.878948545456 <> Val_acc: 0.516666666667
Epoch: 6 <==> Val_loss: 0.877473505338 <> Val_acc: 0.541666666667
Epoch: 7 <==> Val_loss: 0.875736765067 <> Val_acc: 0.525
Epoch: 8 <==> Val_loss: 0.868899710973 <> Val_acc: 0.55
Epoch: 9 <==> Val_loss: 0.867434096336 <> Val_acc: 0.566666666667
Epoch: 10 <==> Val_loss: 0.864289867878 <> Val_acc: 0.575
Epoch: 11 <==> Val_loss: 0.85851559639 <> Val_acc: 0.558333333333
'''

'''
model = Sequential()

model.add(Convolution2D(9, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(36, 3, 3, border_mode='same'))
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

Epoch: 0 <==> Val_loss: 1.3602963686 <> Val_acc: 0.425
Epoch: 1 <==> Val_loss: 0.933181806405 <> Val_acc: 0.483333333333
Epoch: 2 <==> Val_loss: 0.918871215979 <> Val_acc: 0.508333333333
Epoch: 3 <==> Val_loss: 0.932318496704 <> Val_acc: 0.483333333333
Epoch: 4 <==> Val_loss: 0.917647302151 <> Val_acc: 0.516666666667
Epoch: 5 <==> Val_loss: 0.933803625902 <> Val_acc: 0.45
Epoch: 6 <==> Val_loss: 0.917933682601 <> Val_acc: 0.516666666667
Epoch: 7 <==> Val_loss: 0.897326199214 <> Val_acc: 0.525
Epoch: 8 <==> Val_loss: 0.905102717876 <> Val_acc: 0.483333333333
Epoch: 9 <==> Val_loss: 0.86182440122 <> Val_acc: 0.575
Epoch: 10 <==> Val_loss: 0.915821035703 <> Val_acc: 0.575
Epoch: 11 <==> Val_loss: 0.870023588339 <> Val_acc: 0.533333333333
'''






#
