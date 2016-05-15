# Net_50acc
### Summary
Changed the structure of the net and batch size.

# Workflow
### Creating Images and Vectors
 - Images are 50 x 50px with close, high and low data.

<img src="https://github.com/gravity226/forex_net/blob/master/net_chl5050px_5t_fb/imgs/AUDUSD_20010103_05-00-00.png">

### Results
 - Not great...
 - Based on 272,564 images
 - Each image contains 5 ticks of hourly close, high, and low data
 - No images contain overlapping data / no two images share an hourly tick
 - image sizes are 50 x 50px
 - Each batch fed into the net contains 32 images
<br/>
<br/>
Epoch: 0 <==> Val_loss: 0.869268443557 <> Val_acc: 0.489167057998<br/>
Epoch: 1 <==> Val_loss: 0.866069653891 <> Val_acc: 0.494890232452<br/>
Epoch: 2 <==> Val_loss: 0.865739595179 <> Val_acc: 0.49703275417<br/>
Epoch: 3 <==> Val_loss: 0.864506076356 <> Val_acc: 0.499967715429<br/>
Epoch: 4 <==> Val_loss: 0.863866585525 <> Val_acc: 0.500912772952<br/>
Epoch: 5 <==> Val_loss: 0.864004577996 <> Val_acc: 0.501185724351  ----<br/>
Epoch: 6 <==> Val_loss: 0.864007721466 <> Val_acc: 0.50122974877<br/>
Epoch: 7 <==> Val_loss: 0.86462221404 <> Val_acc: 0.499263324727<br/>
Epoch: 8 <==> Val_loss: 0.864987746189 <> Val_acc: 0.498148039449<br/>
Epoch: 9 <==> Val_loss: 0.865478817273 <> Val_acc: 0.49770779526<br/>
Epoch: 10 <==> Val_loss: 0.866258697083 <> Val_acc: 0.495858769667<br/>
Epoch: 11 <==> Val_loss: 0.867218056932 <> Val_acc: 0.494670110357<br/>
Epoch: 12 <==> Val_loss: 0.868429158405 <> Val_acc: 0.492644987089<br/>
Epoch: 13 <==> Val_loss: 0.869951311066 <> Val_acc: 0.491118807235<br/>
Epoch: 14 <==> Val_loss: 0.87197444106 <> Val_acc: 0.489621976993<br/>
Epoch: 15 <==> Val_loss: 0.874969355921 <> Val_acc: 0.488433317683<br/>
Epoch: 16 <==> Val_loss: 0.878379336445 <> Val_acc: 0.486915942711<br/>
Epoch: 17 <==> Val_loss: 0.883033826293 <> Val_acc: 0.484318501997<br/>
Epoch: 18 <==> Val_loss: 0.888434239279 <> Val_acc: 0.48343801362<br/>
Epoch: 19 <==> Val_loss: 0.894807665358 <> Val_acc: 0.481383540739<br/>
Epoch: 20 <==> Val_loss: 0.902503090216 <> Val_acc: 0.483599436489<br/>
Epoch: 21 <==> Val_loss: 0.913414705822 <> Val_acc: 0.477846912422<br/>
Epoch: 22 <==> Val_loss: 0.92487917012 <> Val_acc: 0.477964310872<br/>
Epoch: 23 <==> Val_loss: 0.937490390056 <> Val_acc: 0.47653204978<br/>
Epoch: 24 <==> Val_loss: 0.949860579669 <> Val_acc: 0.475534162952<br/>
Epoch: 25 <==> Val_loss: 0.964669465291 <> Val_acc: 0.477171871333<br/>
Epoch: 26 <==> Val_loss: 0.98342742438 <> Val_acc: 0.477280464901<br/>
Epoch: 27 <==> Val_loss: 1.00032684062 <> Val_acc: 0.476218008924<br/>
Epoch: 28 <==> Val_loss: 1.01855775077 <> Val_acc: 0.474471706975<br/>
Epoch: 29 <==> Val_loss: 1.03354659685 <> Val_acc: 0.478433904674<br/>
Epoch: 30 <==> Val_loss: 1.0571572059 <> Val_acc: 0.474248649919<br/>
Epoch: 31 <==> Val_loss: 1.07434908748 <> Val_acc: 0.474897276357<br/>
Epoch: 32 <==> Val_loss: 1.09566601706 <> Val_acc: 0.4745304062<br/>
Epoch: 33 <==> Val_loss: 1.11112817121 <> Val_acc: 0.471580770135<br/>
Epoch: 34 <==> Val_loss: 1.13252943446 <> Val_acc: 0.474794552713<br/>
Epoch: 35 <==> Val_loss: 1.15231307079 <> Val_acc: 0.472070908667<br/>
Epoch: 36 <==> Val_loss: 1.16562553729 <> Val_acc: 0.473450340458<br/>
Epoch: 37 <==> Val_loss: 1.18270404152 <> Val_acc: 0.473567738909<br/>
Epoch: 38 <==> Val_loss: 1.19795963081 <> Val_acc: 0.472951397044<br/>
Epoch: 39 <==> Val_loss: 1.21324249518 <> Val_acc: 0.472070908667<br/>
