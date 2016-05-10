import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_data(df, close, path='eurusd/train'):
    next_hour_close = []
    close_dif = []

    for n in range(len(close)-1):
        dif = close[n+1] - close[n]
        close_dif.append(dif)
        next_hour_close.append(close[n] + dif)

    graph_list = []
    count = 0
    files = []

    for n in range(0, len(close)-10, 5):  # need to change this to n - 6
        x = close_dif[n+6]
        g = close[n:n+5]

        filename = '/Volumes/SD/' + path + str(df.ix[count].date) + '_' + str(df.ix[count].time).replace(':', '-') + '.png'
        # fig, ax = plt.subplots()
        # ax.plot(range(5), g)
        #
        # fig.patch.set_visible(False)
        # ax.axis('off')
        #
        # # plt.plot(range(10), g[0])
        # plt.savefig(filename)
        # # plt.show()
        # plt.clf()
        # plt.close()
        count += 1
        # files.append(filename)
        print "%s -- %s of %s " % (path, str(count), str(len(graph_list)))

        '''
        list of values
        0. range to be plotted
        1. (n+6) closing price of the next tick after the range(n : n+5)
        2. the difference between ticks 5 and 6
        3. classification [1, 0, -1] 1 meaning a gain from 5 to 6, 0 meaning no gain or loss
           and -1 meaning a loss between 5 and 6
        4. currency pair
        5. name of the file to be accessed
        '''
        graph_list.append([close[n:n+5], close[n+6], x, 1 if x > 0 else -1 if x < 0 else 0, path, filename])

    #
    # should I make the y sclace the same for all?
    #


    # for g in graph_list:
    #     filename = '/Volumes/SD/' + path + str(df.ix[count].date) + '_' + str(df.ix[count].time).replace(':', '-') + '.png'
    #     fig, ax = plt.subplots()
    #     ax.plot(range(5), g[0])
    #
    #     fig.patch.set_visible(False)
    #     ax.axis('off')
    #
    #     # plt.plot(range(10), g[0])
    #     plt.savefig(filename)
    #     # plt.show()
    #     plt.clf()
    #     plt.close()
    #     count += 1
    #     files.append(filename)
    #     print "%s -- %s of %s " % (path, str(count), str(len(graph_list)))
    #     # if count == 2:
    #     #     break

    return graph_list

if __name__ == '__main__':
    # # EURUSD
    # df = pd.read_csv('data/EURUSD_hour.csv')
    #
    # # time = range(86028)
    #
    # # u'<TICKER>', u'<DATE>', u'<TIME>', u'<OPEN>', u'<LOW>', u'<HIGH>', u'<CLOSE>']
    # df.columns = ['tick', 'date', 'time', 'open', 'low', 'high', 'close']
    #
    # # 1. look at next hour's close
    # # 2. see how much it changed
    # # 3. get and average change
    #
    # #plt.plot(time, df.close.values)
    #
    # #plt.show()
    #
    # close = list (df['close'])
    #
    # files = save_data(close)

    # AUDJPY
    df_AUDJPY = pd.read_csv('data/AUDJPY_hour.csv')
    df_AUDJPY.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    close = list (df_AUDJPY['close'])
    files = save_data(df_AUDJPY, close, 'AUDJPY')

    data = pd.DataFrame(files)

    # AUDUSD
    df_AUDUSD = pd.read_csv('data/AUDUSD_hour.csv')
    df_AUDUSD.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    close = list (df_AUDUSD['close'])
    files = save_data(df_AUDUSD, close, 'AUDUSD')

    data = pd.concat([data, pd.DataFrame(files)], axis=0)

    # CHFJPY
    df_CHFJPY = pd.read_csv('data/CHFJPY_hour.csv')
    df_CHFJPY.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    close = list (df_CHFJPY['close'])
    files = save_data(df_CHFJPY, close, 'CHFJPY')

    data = pd.concat([data, pd.DataFrame(files)], axis=0)

    # EURCHF
    df_EURCHF = pd.read_csv('data/EURCHF_hour.csv')
    df_EURCHF.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    close = list (df_EURCHF['close'])
    files = save_data(df_EURCHF, close, 'EURCHF')

    data = pd.concat([data, pd.DataFrame(files)], axis=0)

    # EURGBP
    df_EURGBP = pd.read_csv('data/EURGBP_hour.csv')
    df_EURGBP.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    close = list (df_EURGBP['close'])
    files = save_data(df_EURGBP, close, 'EURGBP')

    data = pd.concat([data, pd.DataFrame(files)], axis=0)

    data.to_csv('net_data.csv')

    # # EURJPY
    # df_EURJPY = pd.read_csv('data/EURJPY_hour.csv')
    # df_EURJPY.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_EURJPY['close'])
    # files = save_data(df_EURJPY, close, 'EURJPY')
    #
    # # GBPCHF
    # df_GBPCHF = pd.read_csv('data/GBPCHF_hour.csv')
    # df_GBPCHF.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_GBPCHF['close'])
    # files = save_data(df_GBPCHF, close, 'GBPCHF')
    #
    # # GBPJPY
    # df_GBPJPY = pd.read_csv('data/GBPJPY_hour.csv')
    # df_GBPJPY.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_GBPJPY['close'])
    # files = save_data(df_GBPJPY, close, 'GBPJPY')
    #
    # # GBPUSD
    # df_GBPUSD = pd.read_csv('data/GBPUSD_hour.csv')
    # df_GBPUSD.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_GBPUSD['close'])
    # files = save_data(df_GBPUSD, close, 'GBPUSD')
    #
    # # NZDUSD
    # df_NZDUSD = pd.read_csv('data/NZDUSD_hour.csv')
    # df_NZDUSD.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_NZDUSD['close'])
    # files = save_data(df_NZDUSD, close, 'NZDUSD')
    #
    # # USDCAD
    # df_USDCAD = pd.read_csv('data/USDCAD_hour.csv')
    # df_USDCAD.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_USDCAD['close'])
    # files = save_data(df_USDCAD, close, 'USDCAD')
    #
    # # USDCHF
    # df_USDCHF = pd.read_csv('data/USDCHF_hour.csv')
    # df_USDCHF.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_USDCHF['close'])
    # files = save_data(df_USDCHF, close, 'USDCHF')
    #
    # # USDJPY
    # df_USDJPY = pd.read_csv('data/USDJPY_hour.csv')
    # df_USDJPY.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_USDJPY['close'])
    # files = save_data(df_USDJPY, close, 'USDJPY')
    #
    # # XAGUSD
    # df_XAGUSD = pd.read_csv('data/XAGUSD_hour.csv')
    # df_XAGUSD.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_XAGUSD['close'])
    # files = save_data(df_XAGUSD, close, 'XAGUSD')
    #
    # # XAUUSD
    # df_XAUUSD = pd.read_csv('data/XAUUSD_hour.csv')
    # df_XAUUSD.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']
    # close = list (df_XAUUSD['close'])
    # files = save_data(df_XAUUSD, close, 'XAUUSD')













#
