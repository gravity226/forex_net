# Net_4080px_10t
### Summary
Third attempt at building this net. Going to use 10 ticks in every image and widen the image to 40 x 80px. Hopefully this will help?

# Workflow
### Creating Images and Vectors
 - Change the images to 40 x 80px with 10 ticks

<img src="https://github.com/gravity226/forex_net/blob/master/net_4080px_10t/imgs/GBPUSD_20010103_10-00-00.png">

### Results
 - Not great...
 - Based on 272,564 images
 - Each image contains 5 ticks of hourly open, close, high, low data
 - No images contain overlapping data / no two images share an hourly tick
 - image sizes are 40 x 40px
 - Each batch fed into the net contains 1000 images
<br/>
<br/>
 Epoch 0<br/>
 Val_loss 0.921941109817<br/>
 Val_acc 0.463531130422<br/>

 Epoch 1<br/>
 Val_loss 0.887297276163<br/>
 Val_acc 0.475725203861<br/>

 Epoch 2<br/>
 Val_loss 0.886245138889<br/>
 Val_acc 0.478469728771<br/>

 Epoch 3<br/>
 Val_loss 0.885666501783<br/>
 Val_acc 0.479929584004<br/>

 Epoch 4<br/>
 Val_loss 0.885022032<br/>
 Val_acc 0.482595965701<br/>
