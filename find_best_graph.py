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

    fig, ax = plt.subplots(figsize=(.5, .5))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.plot(range(5), df.close.values[7000:7005], lw=2, color='b', alpha=1)
    # ax.plot(range(10), df.open.values[7000:7010], lw=1, color='g', alpha=.5)
    # ax.plot(range(10), df.low.values[7000:7010], lw=1, color='r', alpha=.5)
    # ax.plot(range(10), df.high.values[7000:7010], lw=1, color='k', alpha=.5)

    plt.savefig('smaller_graph10.png')
    plt.show()













#
