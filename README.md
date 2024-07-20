# DADS 7202 Deer Family Image Classification
* Perform multi-class classification on a total of 4 classes of Deer family images dataset and compare the performance of CNN architectures.
* The objective is to optimize the Loss function and performance metric (Accuracy score).
* The study also includes GradCam analysis and Eyeball analysis.

## Contributors
* Itthisak Pratukaew
* Pinyawat Sabsanhor
* Pisit Kuensuwan
* Wichanee Maneelok

## Dataset
A total of 800 images were used from 4 deer family classes including 200 images from each class (Caribou, Deer, Elk, and Moose). We get images using web scraping from 2 main sources which are Bing Downloader and Yahoo.
![output](https://github.com/user-attachments/assets/926ff6aa-eb1c-4b52-a060-a7ef09ab712b)

## EDA
Our dataset is balanced in each class as shown in the figures below.

![Number of Images by Train Class](https://github.com/user-attachments/assets/9b5597f0-3e56-4830-861e-e072d1ebec7f)
![Number of Images by Test Class](https://github.com/user-attachments/assets/388f85c9-85a3-4554-8ce5-b8f8f526de27)

We plot the sizes of images for each class to observe the variation in size.

![Train Dataset Image's Siz](https://github.com/user-attachments/assets/acd903b6-ccc2-4f76-a979-7dc4f4962cd7)
![Test Dataset Image's Size](https://github.com/user-attachments/assets/5b74c673-d514-4b53-8877-1438f369d87b)

## Splitting Data
From a total of 800 images (200 images for each class). We split 80:20 by hand. Resulting in 640 images for training and 160 images for testing.
From 640 training images, we split 90:10. Resulting in 576 images for training and 64 images for validation.
We also resized all images to be 224 x 224

![image](https://github.com/user-attachments/assets/91bfaab8-6453-4197-85ff-5884a2fb4db3)

## Data Augmentation
We using these functions for augmenting our data
  * RandomFlip
  * RandomTranslation
  * RandomRotation
  * RandomZoom
  * RandomBrightness
  * Rescaling

  ![augmentd](https://github.com/user-attachments/assets/1cc3722f-9eb5-4e2b-9c94-571869556e48)

## Classifier Modification
We create our classifier to use with selected CNN models with pre-trained weight from imagenet.
1. DenseNet201

![image](https://github.com/user-attachments/assets/d9b09b2b-c605-45fe-9779-263a0ae5b91e)


2. ResNet152V2

![image](https://github.com/user-attachments/assets/21669261-951d-44e4-85a6-d582d319fe8f)


3. VGG16  

![image](https://github.com/user-attachments/assets/1bc2c4e0-5fb7-4293-bd41-8103a100eab1)

## Training Method

|                      |                        **DenseNet201**                       |                        **ResNet152V2**                       |                           **VGG16**                          |
|:--------------------:|:------------------------------------------------------------:|:------------------------------------------------------------:|:------------------------------------------------------------:|
|       Optimizer      |                             Adam                             |                             Adam                             |                             Adam                             |
|         Loss         |               Sparse Categorical Cross Entropy               |               Sparse Categorical Cross Entropy               |               Sparse Categorical Cross Entropy               |
|         Epoch        |                              200                             |                              80                              |                              100                             |
|     Learning Rate    |                             0.003                            |                            0.0001                            |                            0.0001                            |
|    Early Stopping    |        Monitor on Val Accuracy with 60 patience epochs       |        Monitor on Val Accuracy with 60 patience epochs       |        Monitor on Val Accuracy with 50 patience epochs       |
| Reduce LR on Plateau | Monitor on Val Loss min lr = 0.000001 with 2 patience epochs | Monitor on Val Loss min lr = 0.00005 with 20 patience epochs | Monitor on Val Loss min lr = 0.00005 with 10 patience epochs |
|   Model Checkpoint   |                      Monitor on Val Loss                     |                      Monitor on Val Loss                     |                      Monitor on Val Loss                     |
|      Training on     |                 Last Conv Layer + Classifier                 |                 Last Conv Layer + Classifier                 |                 Last Conv Layer + Classifier                 |

## Evaluation
Since the classification of the deer family have all class equal importance, we use an accuracy metric for the evaluation.

### Learning Curve
1. DenseNet201

![densenet](https://github.com/user-attachments/assets/47a48fc0-469e-4f20-82d2-b80b798ca397)

2. ResNet152V2

<img src="https://github.com/user-attachments/assets/6c493310-a8e8-4e87-865a-709a46f19eb6" width="468" height="256">


3. VGG16

![vgg](https://github.com/user-attachments/assets/f66cad5a-d65b-4ea7-bbcc-c5533d1c31d5)

### Confusion Matrix
1. DenseNet201

![densecm](https://github.com/user-attachments/assets/41a011fa-a4dd-44d3-885d-57d9bca41356)

2. ResNet152V2

<img src="https://github.com/user-attachments/assets/6832e34d-0cdb-4cda-9f67-bb215faf438e" width="468" height="391">


3. VGG16

![vggcm](https://github.com/user-attachments/assets/f7d3e450-7a4e-4043-be19-ef893c6cd409)


### Accuracy Score
|                    | **DenseNet201** | **ResNet152V2** |   **VGG16**   |
|:------------------:|:---------------:|:---------------:|:-------------:|
| Mean Accuracy ± SD |  0.93 ± 0.0036  |  0.84 ± 0.0308  | 0.61 ± 0.0353 |

### GradCam Analysis
1. DenseNet201

<img src="https://github.com/user-attachments/assets/5b8d8a35-c058-42cd-a3c2-3aad2263e9e4" width="468" height="312">


2. ResNet152V2

<img src="https://github.com/user-attachments/assets/b4ae8373-e12c-4407-900f-afa1f8aa9878" width="468" height="312">


3. VGG16

<img src="https://github.com/user-attachments/assets/bb57d333-fb52-4733-b745-aaac3ef58f7c" width="468" height="312">

