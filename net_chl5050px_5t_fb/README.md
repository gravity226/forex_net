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
 Val_loss 0.896974503121<br/>
 Val_acc 0.479199128164<br/>

 -----------------------<br/>
 Epoch 1<br/>
 Val_loss 0.887065941379<br/>
 Val_acc 0.482546438555<br/>

 -----------------------<br/>
 Epoch 2<br/>
 Val_loss 0.884792789216<br/>
 Val_acc 0.48448850471<br/>

 -----------------------<br/>
 Epoch 3<br/>
 Val_loss 0.884615197723<br/>
 Val_acc 0.484105578436<br/>

 -----------------------<br/>
 Epoch 4<br/>
 Val_loss 0.883717042623<br/>
 Val_acc 0.48563936451<br/>

 -----------------------<br/>
 Epoch 5<br/>
 Val_loss 0.88313270685<br/>
 Val_acc 0.487125868217<br/>

 -----------------------<br/>
 Epoch 6<br/>
 Val_loss 0.882755677779<br/>
 Val_acc 0.488086249802<br/>

 -----------------------<br/>
 Epoch 7<br/>
 Val_loss 0.882460864909<br/>
 Val_acc 0.489206454529<br/>

 -----------------------<br/>
 Epoch 8<br/>
 Val_loss 0.882262820508<br/>
 Val_acc 0.488481854617<br/>

 -----------------------<br/>
 Epoch 9<br/>
 Val_loss 0.882037024577<br/>
 Val_acc 0.488750267542<br/>
 
