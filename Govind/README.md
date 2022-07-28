# Video Super Resolution
## Govind Kumar

This is the my implementation of Video Super Resolution for Seasons Of Code 2022.

## Problem Statement

Consider a single low-resolution image, we first upscale it to the desired size using bicubic interpolation to obtain Y. Our goal is to recover from Y, an image F(Y) that is as similar as possible to the actual high resolution image X. This is the "Image Super Resolution". We then extend this model to perform super resolution frame-by-frame on a video, thus achieving the "NAIVE VIDEO SUPER RESOLUTION"

## Step 1 - Preparing Data

Prepare the appropriate pytorch dataset for the preprocessed data in the form of HDF5 file.
I have used the T91 data for this step.

## Step 2 - Bicubic Interpolation

Perform bicubic interpolation on the images to obtain poor-quality upscaling. This will act as a basis of input for the neural network.

## Step 3 - Training the Model

Create a convolutional neural network model with appropriate parameters and hyperparameters.

## Step 4 - Testing the Model

Test the model using eval.py for automatic evaluation on testdataset and test.py for manual side-by-side comparison.

## Step 5 - Extend the model to VSR

Extend the model to naive VSR by using model frame-by-frame on a video

## Conclusion

Through this project I have implemented my srcnn network based on the "9-3-5 SRCNN" architecture.
