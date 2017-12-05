# Vehicle Detection with RetinaNet

Vehicle and pedestrian detection and tracking play a vital role in autonomous driving.  In previous project, I implemented a [vehicle detection and tracking pipeline](https://github.com/yangliupku/vehicle_detection) based on traditional computer vision techniques. This project is to explore application of RetinaNet on the vehicle detection taask. 

## Dataset 
The training and evaluation of this project is based on the [Udacity annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations). It includes driving in Mountain View California and neighboring cities during daylight conditions. I combined the two datasets and only retained bounding box annotations for car, truck, and pedestrian. The combined dataset 

![example images][image1] ![example images][image2]



[//]: # (Image References)

[image1]: ./img/figure1_dataset_example.png "dataset"
[image2]: ./img/figure2_dataset_cat.png "cat"
