# Forex Net
### Summary
I am working on this project to get a better understanding of neural nets.  The goal is to create images from historical forex data and see if a net can find trends and make accurate predictions.

# Workflow
### Creating Images and Vectors
 - Launched an AWS instance to created the images and vectors.
 - Successfully implemented parallel processing to cut down on creation time (still took about 4 hours on a 40 core machine)
 - Created over 272,000 images and their corresponding vectors; each containing 5 ticks on an hourly scale from 2001 to 2015.

<img src="https://raw.githubusercontent.com/gravity226/forex_net/master/imgs/EURUSD_20010103_00-00-00.png" width="320" height="240">

### Building the Net
 - Transferred the vectors to a GPU AWS instance to begin training the net.
 - Altered the net framework from the cifar10_cnn Keras Net (this example utilizes RGB color)
 - Tried running the net as a classifier; predicting whether the market will go up, down, or stay the same in the next tick.
 - I am hoping that this will give me an indiction of whether this idea has any potential.

### Working Classifier nets
#### [Net_4040px](https://github.com/gravity226/forex_net/tree/master/net_4040px)
 - Using 40x40px images; 5 ticks; open, close, high, and low data.
 - Best validation accuracy is about %48...

#### [Net_50100px_10t](https://github.com/gravity226/forex_net/tree/master/net_50100px_10t)
 - Using 50x100px images; 10 ticks; open, close, high, and low data.
 - Best validation accuracy is about %48...  Not much difference.

#### [net_c5050px_5t](https://github.com/gravity226/forex_net/tree/master/net_c5050px_5t)
 - Using 50x50px images; 5 ticks; close data only.
 - Currently training

### Things to try
 - Only include images from prime trading hours.
 - Only graph closing ticks, or some other variety.
 - Change number of ticks being graphed.
 - Allow for overlap in ticks on each graph. Currently no graph data overlaps any other graph.
