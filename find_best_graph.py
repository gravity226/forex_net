import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # EURUSD
    df = pd.read_csv('data/EURUSD_hour.csv')

    time = range(86028)

    # u'<TICKER>', u'<DATE>', u'<TIME>', u'<OPEN>', u'<LOW>', u'<HIGH>', u'<CLOSE>']
    df.columns = ['sym', 'date', 'time', 'open', 'low', 'high', 'close']

    # 1. look at next hour's close
    # 2. see how much it changed
    # 3. get and average change
    r = range(7105, 7110)

    fig, ax = plt.subplots(figsize=(.5,.5))
    fig.patch.set_visible(False)
    ax.axis('off')
    # ax.fill(0,df.close.values[7000],'r')
    # ax.fill_between(range(5), df.close.values[7000:7005], [df.close[7000:7005].min()] * 5, color='r')
    ax.fill_between(range(5), df.close.values[r], df.low.values[r], color='r')
    ax.fill_between(range(5), df.close.values[r], df.high.values[r], color='k')
    ax.fill_between(range(5), df.close.values[r], df.open.values[r], color='y')
    ax.plot(range(5), df.close.values[r], lw=2, color='y', alpha=1)
    ax.plot(range(5), df.open.values[r], lw=1, color='y', alpha=.5)
    ax.plot(range(5), df.low.values[r], lw=1, color='r', alpha=1)
    ax.plot(range(5), df.high.values[r], lw=1, color='k', alpha=1)

    plt.savefig('smaller_graph10.png')
    plt.show()













#
