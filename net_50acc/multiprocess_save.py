import math
import multiprocessing
import itertools
from timeit import Timer

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json

from skimage.io import imread
from skimage.transform import resize

# Globals
ticks = 5
path = ''
df = pd.DataFrame()

def check_prime(n): # n = lower range

    fig, ax = plt.subplots(figsize=(.5, .5))
    fig.patch.set_visible(False)
    ax.axis('off')
    r = (n, n+ticks-1)
    # n:n+ticks-1 ----- This will give you exactly n ticks.
    ax.fill_between(range(5), df.ix[r[0]: r[1]].close.values.tolist(), df.ix[r[0]: r[1]].low.values.tolist(), color='k')
    ax.fill_between(range(5), df.ix[r[0]: r[1]].close.values.tolist(), df.ix[r[0]: r[1]].high.values.tolist(), color='r')
    ax.plot(range(ticks), df.ix[r[0]: r[1]].close.values.tolist(), lw=1, color='y', alpha=1)
    # ax.plot(range(ticks), df.ix[r[0]: r[1]].open.values.tolist(), lw=1, color='g', alpha=.5)
    ax.plot(range(ticks), df.ix[r[0]: r[1]].high.values.tolist(), lw=1, color='r', alpha=1)
    ax.plot(range(ticks), df.ix[r[0]: r[1]].low.values.tolist(), lw=1, color='k', alpha=1)

    filename = path + '_' + str(df.ix[n].date) + '_' + str(df.ix[n].time).replace(':', '-')
    plt.savefig('imgs/' + filename + '.png')
    plt.clf()
    plt.close()

    '''
    0. currency pair
    1. filename
    2. vector
    3. range in df
    4. dif between tick and tick + 1
    5. close of tick + 1
    6. classification [1, 0, -1] 1 meaning a gain from 5 to 6, 0 meaning no gain or loss
    '''
    vect = imread('imgs/' + filename + '.png')
    dif = df.ix[r[1]+1].close - df.ix[r[1]].close

    # file_list.append([path, filename, vect, r, dif, df.ix[r[1]+1].close, 1 if dif > 0 else -1 if dif < 0 else 0])

    reshape_vect = []   # Need it in shape = (n, 3, R, G, B) for net
    temp_colors = []
    for color in xrange(3):   # need to change into a shape = (3, 120, 160); 3 = (r, g, b)
        temp_rows = []
        for row in vect:     # row.shape = (160, 4)
            temp_cols = []
            for col in row:
                temp_cols.append(float(col[color])/255)
            temp_rows.append(temp_cols)
        temp_colors.append(temp_rows)
    reshape_vect.append(temp_colors)

    row = [path, filename, reshape_vect[0], r, dif, df.ix[r[1]+1].close, 1 if dif > 0 else -1 if dif < 0 else 0]

    d = {    'pair':        row[0],
             'filename':    row[1],
             'vector':      row[2],
             'data_range':  row[3],
             'dif':         row[4],
             'close':       row[5],
             'class':       row[6]   }
    with open('jsons/' + filename + '.json', 'w') as data_file:
        data = json.dump(d, data_file)

    print path, n
    return 'jsons/' + filename + '.json'


def primes_parallel():
    pool = multiprocessing.Pool(4)
    pool.Process.daemon = True

    # num_times = 2
    num_times = (len(df) - ticks - 2) / ticks  # always need to have 1 extra space for the predicted value

    num_times = 20 # Delete this before sending to AWS

    img_range = range(0, num_times * ticks, ticks)

    return pool.map(check_prime, img_range)


if __name__ == "__main__":
    # t = Timer(lambda: primes_sequential())
    # print "Completed sequential in %s seconds." % t.timeit(1)

    # t = Timer(lambda: primes_parallel())
    # print "Completed parallel in %s seconds." % t.timeit(1)

    plt.clf()

    time = range(86028)
    file_list = []

    # path = 'EURUSD'
    # df = pd.read_csv('data/EURUSD_hour.csv')
    # df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    #
    # primes_sequential()

    # u'<TICKER>', u'<DATE>', u'<TIME>', u'<OPEN>', u'<LOW>', u'<HIGH>', u'<CLOSE>']

    path = 'EURUSD'
    df = pd.read_csv('../data/EURUSD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'AUDJPY'
    df = pd.read_csv('../data/AUDJPY_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'AUDUSD'
    df = pd.read_csv('../data/AUDUSD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'CHFJPY'
    df = pd.read_csv('../data/CHFJPY_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'EURCHF'
    df = pd.read_csv('../data/EURCHF_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'EURGBP'
    df = pd.read_csv('../data/EURGBP_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'EURJPY'
    df = pd.read_csv('../data/EURJPY_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'GBPCHF'
    df = pd.read_csv('../data/GBPCHF_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'GBPJPY'
    df = pd.read_csv('../data/GBPJPY_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'GBPUSD'
    df = pd.read_csv('../data/GBPUSD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'NZDUSD'
    df = pd.read_csv('../data/NZDUSD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'USDCAD'
    df = pd.read_csv('../data/USDCAD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'USDCHF'
    df = pd.read_csv('../data/USDCHF_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'USDJPY'
    df = pd.read_csv('../data/USDJPY_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'XAGUSD'
    df = pd.read_csv('../data/XAGUSD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    path = 'XAUUSD'
    df = pd.read_csv('../data/XAUUSD_hour.csv')
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    file_list += primes_parallel()

    save_df = pd.DataFrame(file_list)
    save_df.to_csv('json_files.csv')
    '''
    0. currency pair
    1. filename
    2. vector
    3. range in df
    4. dif between tick and tick + 1
    5. close of tick + 1
    6. classification [1, 0, -1] 1 meaning a gain from 5 to 6, 0 meaning no gain or loss
    '''
    # d = {}
    # for row in file_list:
    #     d[ row[1][-28:-4] ] = {  'pair':        row[0],
    #                              'filename':    row[1],
    #                              'vector':      row[2],
    #                              'data_range':  row[3],
    #                              'dif':         row[4],
    #                              'close':       row[5],
    #                              'class':       row[6]  }
    # with open('net_data.json', 'w') as data_file:
    #     data = json.dump(d, data_file)
    # with open('test.txt', 'w') as data_file:
    #     data = json.dump(t_Data, data_file)
    # with open('test.json', 'w') as data_file:
    #     data = json.dump(d, data_file)




#
