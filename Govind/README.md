# Video Super Resolution
## By Govind Kumar

This is the my implementation of Video Super Resolution for Seasons Of Code 2022.

## Problem Statement

Consider a single low-resolution image, we first upscale it to the desired size using bicubic interpolation to obtain Y. Our goal is to recover from Y, an image F(Y) that is as similar as possible to the actual high resolution image X. This is the "Image Super Resolution". We then extend this model to perform super resolution frame-by-frame on a video, thus achieving the "NAIVE VIDEO SUPER RESOLUTION"

## Step 1 - Preparing Data

Preprocess the data to get HDF5 files. I have used the [T91 dataset] for this step.
Prepare the appropriate pytorch dataset for the preprocessed data.

## Step 2 - Bicubic Interpolation

Perform bicubic interpolation on the images to obtain poor-quality upscaling. This will act as a basis of input for the neural network.

## Step 3 - Training the Model

Create a convolutional neural network model with appropriate parameters and hyperparameters.

## Step 4 - Testing the Model

Test the model using eval.py for automatic evaluation on testdataset and test.py for manual side-by-side comparison.
I have used [Set5 dataset] for auto evaluation using HDF5 files. 

## Step 5 - Extend the model to VSR

Extend the model to naive VSR by using model frame-by-frame on a video.

## Conclusion

- Through this project I have implemented my srcnn network based on the "9-3-5 SRCNN" architecture.
- The model weights file has been saved in outputs folder.
- The instructions to run the code could be found in run.txt. 
- Some resuts of the model can be found under VSR_results and ImageSR_results

[T91 dataset]: https://drive.google.com/drive/folders/1ACLEQqxj4OEyw5HnWV8M-YyQ1OYctw20?usp=sharing
[Set5 dataset]: https://drive.google.com/drive/folders/1xhO83Ru5vp59xsveTai3oQOjVp1pxb3c?usp=sharing
