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

### Results
 - Not great...
<br/>
<br/>
 Epoch 0<br/>
 Val_loss 0.898417556461<br/>
 Val_acc 0.471739901837<br/>
<br/>
 -------<br/>
 Epoch 1<br/>
 Val_loss 0.891484757701<br/>
 Val_acc 0.47570062224<br/>
<br/>
 -------<br/>
 Epoch 2<br/>
 Val_loss 0.887869497795<br/>
 Val_acc 0.480078248423<br/>
<br/>
 -------<br/>
 Epoch 3<br/>
 Val_loss 0.889913597605<br/>
 Val_acc 0.478520459745<br/>
<br/>
 -------<br/>
 Epoch 4<br/>
 Val_loss 0.888639910972<br/>
 Val_acc 0.479168471781<br/>

### Next steps
 - Try with more ticks and a wider image
