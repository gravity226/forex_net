import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/EURUSD_hour.csv')

time = range(86028)

# u'<TICKER>', u'<DATE>', u'<TIME>', u'<OPEN>', u'<LOW>', u'<HIGH>', u'<CLOSE>']
df.columns = ['tick', 'date', 'time', 'open', 'low', 'high', 'close']

# 1. look at next hour's close
# 2. see how much it changed
# 3. get and average change

#plt.plot(time, df.close.values)

#plt.show()

close = list (df['close'])
next_hour_close = []
close_dif = []

for n in range(len(close)-1):
    dif = close[n+1] - close[n]
    close_dif.append(dif)
    next_hour_close.append(close[n] + dif)

graph_list = []

for n in range(len(close)-10):
    graph_list.append([close[n:n+10], close_dif[n]])

#
# should I make the y sclace the same for all?
#

count = 0
files = []
for g in graph_list:
    filename = '/Volumes/SD/eurusd/train' + str(df.ix[count].date) + '_' + str(df.ix[count].time).replace(':', '-') + '.png'
    fig, ax = plt.subplots()
    ax.plot(range(10), g[0])

    fig.patch.set_visible(False)
    ax.axis('off')

    # plt.plot(range(10), g[0])
    # plt.savefig(filename)
    plt.show()
    plt.clf()
    count += 1
    files.append(filename)
    print "%s of %s " % (str(count), str(len(graph_list)))
    break


















#
