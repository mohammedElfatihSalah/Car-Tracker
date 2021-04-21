# Car-Tracker
An Implementation of simple car tracker using Detectron2 as the core detection system.

<img src='https://github.com/mohammedElfatihSalah/Car-Tracker/blob/master/30.jpg?raw=true' width=200 height=400 />

## Methodology

I created a matching score matrix M as follows
M[i,j] = iou_overlap(i, j)
M[i,j] = 0 if M[i , j] < T

Where iou_overlap(i,j) calculates box over union intersection between box i and box j and T is a threshold and it is set to 0.4

And For the best match j_best (which is the index for object in the second image that match i in the first image) can be found by
j_best = argmax(M[i,j])
If the ith row in M is all zero  this j_best is ignored, because it means we didn’t find a match for i.
Also if j_best is already assigned to a previous object with index k < i we ignore it.

For the detection model I implemented mask_rcnn_R_101_FPN_3x.


## Run the code yourself
Just download the notebook, and it contains instructions of how to track a sequence of images

