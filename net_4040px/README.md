# Forex Net
### Summary
Second attempt at building this net.  The problem is the instance can't handle the size of the 272,000 images together.

# Workflow
### Creating Images and Vectors
 - Changed image sizes to 40 x 40px on creation (no more resizing images later)
 - Improved workflow - now images and vectors are created at the same time (no more running 2 scripts to create each)
 - Improved scalability - now the same script is used on my local computer as on the AWS instance.  Previously a slightly different script was used on each to account for the different sizes of the datasets.

<img src="https://raw.githubusercontent.com/gravity226/forex_net/master/imgs/EURUSD_20010103_00-00-00.png" width="320" height="240">

 - Pictures are much smaller now :)

<img src="https://github.com/gravity226/forex_net/blob/master/net_4040px/imgs/AUDJPY_20010103_00-00-00.png">

### Building the Net
 - Transferred the vectors to a GPU AWS instance to begin training the net.
 - Copied the net framework from the cifar10_cnn Keras Net (this example utilizes RGB color)
 - Tried running the net as a classifier; predicting whether the market will go up, down, or stay the same in the next tick.
 - I am hoping that this will give me an indiction of whether this idea has any potential.

### Current Issues
 - I overloaded the memory with the 272,000 images. The script got through about 14,000 images before crashing with the message "Killed" given.
 - Need to set up Spark to handle the memory problems.
 - Need to reshape the vectors so that they are in the correct format before running the net script.  Currently I am reshaping the vectors immediately before creating the net.

### Things to try
 - Only include images from prime trading hours.
 - Only graph closing ticks, or some other variety.
 - Change number of ticks being graphed.
 - Allow for overlap in ticks on each graph. Currently no graph data overlaps any other graph.
