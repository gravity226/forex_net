# Net_chl5050px_5t_fb
### Summary
Fourth attempt at building this net. Using 5 ticks with close, high, and low data.  Also adding the fill_between to git the net more to learn on.  

# Workflow
### Creating Images and Vectors
 - Change the images to 50 x 50px with close, high and low data.

<img src="https://github.com/gravity226/forex_net/blob/master/net_chl5050px_5t_fb/imgs/AUDUSD_20010103_05-00-00.png">

### Results
 - Not great...
 - Based on 272,564 images
 - Each image contains 5 ticks of hourly close, high, and low data
 - No images contain overlapping data / no two images share an hourly tick
 - image sizes are 50 x 50px
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
