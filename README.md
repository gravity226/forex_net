# Forex Net
### Summary
I am working on this project to get a better understanding of neural nets.  The goal to create images from historical forex data and see is a net can find trends and make accurate predictions.

# Workflow
### Creating Images and Vectors
 - Launched an AWS instance to created the images and vectors.
 - Successfully implemented multiprocessing to cut down on creation time (still took about 4 hours)
 - Created over 272,000 images and vectors saved; each containing 5 ticks on an hourly scale from 2001 to 2015.

### Building the Net
 - Transferred the vectors to GPU AWS instance to begin training the net.
 - Copied the framework from the cifar10_cnn Keras Net (this example utilizes RGB color)
 - Tried running the net as a classifier; predicting whether the market will go up, down, or stay the same in the next tick.
 - I am hoping that this will give me an indiction of whether this idea has any potential.

### Current Issues
 - I overloaded the memory with the 272,000 images. The script got through 14,000 images before crashing with the message "Killed" given.
 - Need to set up Spark to handle the memory problems.
 - Need to re-processes the vectors so that they are in the correct format before running the net script.  Currently I am reshaping the vectors immediately before creating the net.
