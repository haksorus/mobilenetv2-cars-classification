# MobileNetV2 Stanford Cars Classification 
## About work

This is a MobileNetV2-based cars recognition (Stanford Cars-196 Dataset classification)


There are several important criteria of work:

* Accuracy

* Speed & lightweight

Thus, the classifier based on Mobile Net V2 was chosen because of its small number of parameters, small size and the ability to recognize almost real-time based on mobile devices.

<img src="https://user-images.githubusercontent.com/69139386/178808525-61f546ac-6747-469e-8596-537e1b31ea04.png" width="500">


This repository consists of two Google Colaboratory Notebooks:
1. Training, Finetuning, MobileNetV2 (Preprocessing & training) + Prediction (Evaluation)
2. MobileNetV2 Grad-CAM (class activation visualization)

## Dataset

The Stanford Cars-196 dataset consists of 16185 images of cars of 196 classes: 

* Train folder: 8144 images

* Test folder: 8041 images

Despite the large size, the number of images for each class is relatively small (avg 41.25 images per class), and since the cars are visually very similar, this makes it difficult to differentiate brands and models.

## Training 

In my work I used PyTorch-implemented (torchvision.models) pre-trained MobileNetV2 with transfer learning. 

All layers was fine-tuned and the last layer was replased (changing classifier output-size)

Also, I used:

* Cross-Entropy loss
* Adam optimizer (with L2-penalty)

The model has been trained for 10 epochs.

Final score: **0.8486 Accuracy**

## Grad-CAM
Finally, [Grad-CAM](https://github.com/frgfm/torch-cam?ysclid=l5k0ej29tm554836498) (Gradient-weighted Class Activation Heatmap) was used for MobileNetV2 class activation exploration - overlay visualization. 

<img width="895" alt="image" src="https://user-images.githubusercontent.com/69139386/178817486-10d9236f-c759-43e6-ab89-f8449bbff7ff.png">

