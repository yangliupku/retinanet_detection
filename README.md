# Vehicle Detection with RetinaNet

Vehicle and pedestrian detection and tracking play a vital role in autonomous driving.  In previous project, I implemented a [vehicle detection and tracking pipeline](https://github.com/yangliupku/vehicle_detection) based on traditional computer vision techniques. This project is to explore application of RetinaNet on the vehicle detection taask. 

## Dataset 
The training and evaluation of this project is based on the [Udacity annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations). It includes driving in Mountain View California and neighboring cities during daylight conditions. I combined the two datasets and only retained bounding box annotations for car, truck, and pedestrian. The combined dataset 

Here's an overview of the dataset

<img src="./img/figure1_dataset_example.png " width="480"/> ![images][image2]

## Model evalulation

In this project, I'm interested in the detection accuracy of the models as well as their inference speed. The goal is to find a model that can detect vehicles with good accuracy in real time.  

The accuracy of models is primarily evaluated by mean Average Precision (mAP) and mean Average Recall (mAR) at IOU of 0.5. 

The models being benchmarked are

1. sliding window method based on HOG feature and linear classifier
2. RetinaNet with ResNet50 backbone, pre-trained on COCO
3. RetinaNet with ResNet18 backbone, trained on driving dataset
4. RetinaNet with MobileNet backbone, trained on driving dataset




## Main results

### Benchmark

| Model    | AP50 (car) | AP50 (truck) | AP50 (pedestrian) |# of parameters|CPU inference <br/>(s/frame)| GPU inference <br/>(s/frame)|
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| HOG      | 24.6 | -| - |-|6.9|
| RetinaNet-ResNet50 <br />pre-trained on COCO     | 71.8 |53.4  |   32.4 | 37.4|2.0|0.14|
| RetinaNet-ResNet18-64 | 66.7 |54.1 |27.2 | 12.0 | 1.4|0.1|
| RetinaNet-ResNet18-48 | 66.1 |51.0 |18.8 | 7.0 |1.2|0.09|
| RetinaNet-ResNet18-32 | 71.9 |55.2 |34.7 | 3.4 |0.97|0.09
| RetinaNet-MobileNet-1 | 73.3 |54.6 |42.4 | 4.4 | 1.1| 0.1|
| RetinaNet-MobileNet-0.75 | 67.6 |57.2 |29.6 | 2.8 | 1.0 | 0.07|
| RetinaNet-MobileNet-0.5 | 65.3 |55.2 |36.3 | 1.6 | 0.77 | 0.055|
| RetinaNet-MobileNet-0.25 |67.6|54.1 |38.2 | 0.84 |0.54| 0.05| 

### Example detection result
![images][image3]
![images][image4]

### Vehicle tracking on movie

Here's the result of running RetinaNet-ResNet50-COCO on a dash camera video
<img src="./img/video_out_coco.gif" width="720"/>


Here's the result of running RetinaNet-MobileNet-0.25 on a dash camera video

<img src="./img/video_out_mobilenet_0.25.gif" width="720"/>

## Appendix

The following graph shows the structure of feature pyramid net (FPN) built on top of ResNet backbone. 
![images][image5]


The following graph showes the structure of regress and classification subnet. 
![images][image6]




[//]: # (Image References)

[image1]: ./img/figure1_dataset_example.png 
[image2]: ./img/figure2_dataset_cat.png 
[image3]: ./img/figure3_example_mobilenet0.25.png
[image4]: ./img/figure4_example_coco.png
[image5]: ./img/figure5_retinanet_structure.png
[image6]: ./img/figure6_subnet.png

