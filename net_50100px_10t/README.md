# Net_50100px_10t
### Summary
Third attempt at building this net. Going to use 10 ticks in every image and widen the image to 50 x 100px. Hopefully this will help?

# Workflow
### Creating Images and Vectors
 - Change the images to 50 x 100px with 10 ticks

<img src="https://github.com/gravity226/forex_net/blob/master/net_4080px_10t/imgs/GBPUSD_20010103_10-00-00.png">

### Results
 - Not great...
 - Based on 136,270 images
 - Each image contains 10 ticks of hourly open, close, high, low data
 - No images contain overlapping data / no two images share an hourly tick
 - image sizes are 50 x 100px
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

 Epoch 5<br/>
 Val_loss 0.884365815316<br/>
 Val_acc 0.481165307173<br/>

 Epoch 6<br/>
 Val_loss 0.883893987558<br/>
 Val_acc 0.481787893568<br/>