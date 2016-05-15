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

    img_rows, img_cols = 50, 50
    # the CIFAR10 images are RGB
    img_channels = 3

    # net it!
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3))
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
    X_files = 0

    count = 0
    batch_size = 16
    nb_epoch = 50

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
Epoch: 0 <==> Val_loss: 0.872427247008 <> Val_acc: 0.485821202163
Epoch: 1 <==> Val_loss: 0.867091027006 <> Val_acc: 0.498638177977
Epoch: 2 <==> Val_loss: 0.864071980136 <> Val_acc: 0.501537919702
Epoch: 3 <==> Val_loss: 0.863357153933 <> Val_acc: 0.504704742898
Epoch: 4 <==> Val_loss: 0.860260898013 <> Val_acc: 0.509999413009
Epoch: 5 <==> Val_loss: 0.859725046967 <> Val_acc: 0.508848908194
Epoch: 6 <==> Val_loss: 0.858536580669 <> Val_acc: 0.51375616342
Epoch: 7 <==> Val_loss: 0.855104801667 <> Val_acc: 0.518267198876
Epoch: 8 <==> Val_loss: 0.85326851353 <> Val_acc: 0.523696877204
Epoch: 9 <==> Val_loss: 0.852374039134 <> Val_acc: 0.527893871804
Epoch: 10 <==> Val_loss: 0.848403675053 <> Val_acc: 0.534080770135
Epoch: 11 <==> Val_loss: 0.846974328307 <> Val_acc: 0.540440831184
Epoch: 12 <==> Val_loss: 0.840990015852 <> Val_acc: 0.556128199111
Epoch: 13 <==> Val_loss: 0.838260752847 <> Val_acc: 0.55990842921
Epoch: 14 <==> Val_loss: 0.83404273767 <> Val_acc: 0.570955623386
Epoch: 15 <==> Val_loss: 0.822115894137 <> Val_acc: 0.585838811929
Epoch: 16 <==> Val_loss: 0.812353063624 <> Val_acc: 0.599794552713
Epoch: 17 <==> Val_loss: 0.803689133851 <> Val_acc: 0.611951162247
Epoch: 18 <==> Val_loss: 0.793050735241 <> Val_acc: 0.628034749944
Epoch: 19 <==> Val_loss: 0.776855748839 <> Val_acc: 0.643413946939
Epoch: 20 <==> Val_loss: 0.763429827311 <> Val_acc: 0.65932143696
Epoch: 21 <==> Val_loss: 0.743549739399 <> Val_acc: 0.677541676451
Epoch: 22 <==> Val_loss: 0.719597483317 <> Val_acc: 0.700589927213
Epoch: 23 <==> Val_loss: 0.701117598687 <> Val_acc: 0.716101197464
Epoch: 24 <==> Val_loss: 0.675721942596 <> Val_acc: 0.735985559991
Epoch: 25 <==> Val_loss: 0.650693619113 <> Val_acc: 0.752485912187
Epoch: 26 <==> Val_loss: 0.62490576407 <> Val_acc: 0.772561047196
Epoch: 27 <==> Val_loss: 0.596444518364 <> Val_acc: 0.790957384363
Epoch: 28 <==> Val_loss: 0.567590082376 <> Val_acc: 0.807657313923
Epoch: 29 <==> Val_loss: 0.540377848265 <> Val_acc: 0.826969359004
Epoch: 30 <==> Val_loss: 0.511001292935 <> Val_acc: 0.844080183142
Epoch: 31 <==> Val_loss: 0.483890351051 <> Val_acc: 0.860677389058
Epoch: 32 <==> Val_loss: 0.456737539368 <> Val_acc: 0.876185724351
Epoch: 33 <==> Val_loss: 0.433596887259 <> Val_acc: 0.888999765203
Epoch: 34 <==> Val_loss: 0.407820761372 <> Val_acc: 0.901942944353
Epoch: 35 <==> Val_loss: 0.382315457741 <> Val_acc: 0.913257220005
Epoch: 36 <==> Val_loss: 0.360706369933 <> Val_acc: 0.923265437896
'''


'''
Batch size 32
Epoch: 0 <==> Val_loss: 0.869268443557 <> Val_acc: 0.489167057998
Epoch: 1 <==> Val_loss: 0.866069653891 <> Val_acc: 0.494890232452
Epoch: 2 <==> Val_loss: 0.865739595179 <> Val_acc: 0.49703275417
Epoch: 3 <==> Val_loss: 0.864506076356 <> Val_acc: 0.499967715429
Epoch: 4 <==> Val_loss: 0.863866585525 <> Val_acc: 0.500912772952
Epoch: 5 <==> Val_loss: 0.864004577996 <> Val_acc: 0.501185724351
Epoch: 6 <==> Val_loss: 0.864007721466 <> Val_acc: 0.50122974877 ************
Epoch: 7 <==> Val_loss: 0.86462221404 <> Val_acc: 0.499263324727
Epoch: 8 <==> Val_loss: 0.864987746189 <> Val_acc: 0.498148039449
Epoch: 9 <==> Val_loss: 0.865478817273 <> Val_acc: 0.49770779526
Epoch: 10 <==> Val_loss: 0.866258697083 <> Val_acc: 0.495858769667
Epoch: 11 <==> Val_loss: 0.867218056932 <> Val_acc: 0.494670110357
Epoch: 12 <==> Val_loss: 0.868429158405 <> Val_acc: 0.492644987089
Epoch: 13 <==> Val_loss: 0.869951311066 <> Val_acc: 0.491118807235
Epoch: 14 <==> Val_loss: 0.87197444106 <> Val_acc: 0.489621976993
Epoch: 15 <==> Val_loss: 0.874969355921 <> Val_acc: 0.488433317683
Epoch: 16 <==> Val_loss: 0.878379336445 <> Val_acc: 0.486915942711
Epoch: 17 <==> Val_loss: 0.883033826293 <> Val_acc: 0.484318501997
Epoch: 18 <==> Val_loss: 0.888434239279 <> Val_acc: 0.48343801362
Epoch: 19 <==> Val_loss: 0.894807665358 <> Val_acc: 0.481383540739
Epoch: 20 <==> Val_loss: 0.902503090216 <> Val_acc: 0.483599436489
Epoch: 21 <==> Val_loss: 0.913414705822 <> Val_acc: 0.477846912422
Epoch: 22 <==> Val_loss: 0.92487917012 <> Val_acc: 0.477964310872
Epoch: 23 <==> Val_loss: 0.937490390056 <> Val_acc: 0.47653204978
Epoch: 24 <==> Val_loss: 0.949860579669 <> Val_acc: 0.475534162952
Epoch: 25 <==> Val_loss: 0.964669465291 <> Val_acc: 0.477171871333
Epoch: 26 <==> Val_loss: 0.98342742438 <> Val_acc: 0.477280464901
Epoch: 27 <==> Val_loss: 1.00032684062 <> Val_acc: 0.476218008924
Epoch: 28 <==> Val_loss: 1.01855775077 <> Val_acc: 0.474471706975
Epoch: 29 <==> Val_loss: 1.03354659685 <> Val_acc: 0.478433904674
Epoch: 30 <==> Val_loss: 1.0571572059 <> Val_acc: 0.474248649919
Epoch: 31 <==> Val_loss: 1.07434908748 <> Val_acc: 0.474897276357
Epoch: 32 <==> Val_loss: 1.09566601706 <> Val_acc: 0.4745304062
Epoch: 33 <==> Val_loss: 1.11112817121 <> Val_acc: 0.471580770135
Epoch: 34 <==> Val_loss: 1.13252943446 <> Val_acc: 0.474794552713
Epoch: 35 <==> Val_loss: 1.15231307079 <> Val_acc: 0.472070908667
Epoch: 36 <==> Val_loss: 1.16562553729 <> Val_acc: 0.473450340458
Epoch: 37 <==> Val_loss: 1.18270404152 <> Val_acc: 0.473567738909
Epoch: 38 <==> Val_loss: 1.19795963081 <> Val_acc: 0.472951397044
Epoch: 39 <==> Val_loss: 1.21324249518 <> Val_acc: 0.472070908667
'''











#
