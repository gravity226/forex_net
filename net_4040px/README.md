# Net_4040px
### Summary
Second attempt at building this net.  The problem is the instance can't handle the size of the 272,000 images together.

# Workflow
### Creating Images and Vectors
 - Changed image sizes to 40 x 40px on creation (no more resizing images later)
 - Improved workflow - now images and vectors are created at the same time (no more running 2 scripts to create each)
 - Improved scalability - now the same script is used on my local computer as on the AWS instance.  Previously a slightly different script was used on each to account for the different sizes of the datasets.

<img src="https://raw.githubusercontent.com/gravity226/forex_net/master/imgs/EURUSD_20010103_00-00-00.png" width="320" height="240">

 - Pictures are much smaller now :)
 - The json vector files are less than a 1/10 the size of the original files.  Hopefully this will solve the memory problems I am having.

<img src="https://github.com/gravity226/forex_net/blob/master/net_4040px/imgs/AUDJPY_20010103_00-00-00.png">
