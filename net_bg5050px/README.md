# Net_bg5050px
### Summary
Changed the structure of the net, batch size and added more to the image.

# Workflow
### Creating Images and Vectors
 - Images are 50 x 50px with close, high and low data.

<img src="https://github.com/gravity226/forex_net/blob/master/net_bg5050px/graph_5050_bg.png">

### Results
 - Breaking even...
 - Based on 272,564 images
 - Each image contains 5 ticks of hourly close, high, and low data
 - Included a unique image background color for each currency pair
 - No images contain overlapping data / no two images share an hourly tick
 - image sizes are 50 x 50px
 - Each batch fed into the net contains 32 images
<br/>
<br/>
Epoch: 0 <==> Val_loss: 0.870437248523 <> Val_acc: 0.486525592865 <br/>
Epoch: 1 <==> Val_loss: 0.866868987956 <> Val_acc: 0.493810166706 <br/>
Epoch: 2 <==> Val_loss: 0.866648206726 <> Val_acc: 0.496642404321 <br/>
Epoch: 3 <==> Val_loss: 0.866503527059 <> Val_acc: 0.496994599672 <br/>
Epoch: 4 <==> Val_loss: 0.865937763878 <> Val_acc: 0.498315332238 <br/>
Epoch: 5 <==> Val_loss: 0.865761209133 <> Val_acc: 0.497898567742 <br/>
Epoch: 6 <==> Val_loss: 0.865089756729 <> Val_acc: 0.49913125147 <br/>
Epoch: 7 <==> Val_loss: 0.86493003453 <> Val_acc: 0.500085113879 <br/>
Epoch: 8 <==> Val_loss: 0.864686184196 <> Val_acc: 0.499953040623 <br/>
Epoch: 9 <==> Val_loss: 0.864249192617 <> Val_acc: 0.499894341397 <br/>
Epoch: 10 <==> Val_loss: 0.864302581095 <> Val_acc: 0.499644869691 <br/>
Epoch: 11 <==> Val_loss: 0.864154175179 <> Val_acc: 0.501523244896 <br/>
