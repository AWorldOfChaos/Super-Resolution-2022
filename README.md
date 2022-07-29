# Super-Resolution
This repository will host the work done as a part of Seasons of Code 2022 - Video Super Resolution under the Web & Coding Club of IIT Bombay. 

## Introduction

Super Resolution is the process of recovering a High Resolution (HR) image from a given Low Resolution (LR) image. An image may have a “lower resolution” due to a smaller spatial resolution (i.e. size) or due to a result of degradation (such as blurring). Video super-resolution is the task of upscaling a video from a low-resolution to a high-resolution.

Existing video super-resolution (SR) algorithms usually assume that the blur kernels in the degradation process are known and do not model the blur kernels in the restoration. However, this assumption does not hold for blind video SR and usually leads to over-smoothed super-resolved frames. In this paper, we propose an effective blind video SR algorithm based on deep convolutional neural networks (CNNs). Our algorithm first estimates blur kernels from low-resolution (LR) input videos. Then, with the estimated blur kernels, we develop an effective image deconvolution method based on the image formation model of blind video SR to generate intermediate latent frames so that sharp image contents can be restored well. To effectively explore the information from adjacent frames, we estimate the motion fields from LR input videos, extract features from LR videos by a feature extraction network, and warp the extracted features from LR inputs based on the motion fields. Moreover, we develop an effective sharp feature exploration method that first extracts sharp features from restored intermediate latent frames and then uses a transformation operation based on the extracted sharp features and warped features from LR inputs to generate better features for HR video restoration. We formulate the proposed algorithm into an end-to-end trainable framework and show that it performs favorably against state-of-the-art methods.

## Mentors

- Kartik Gokhale
- Hastyn Doshi

## Mentees

- Umang Agarwal
- Dhananjay Raman
- Govind Kumar
- Om Godage
- Advait Risbud
- Abeer Mishra
- Gaurav Misra

## Schedule

### Week 1 - Basic Skills

1. Taking Stock of Past Experience
2. Learning/Revising Python
3. Getting familiar with Version Control Systems(VCS) such as Git, as well as Github.
4. Brush up on Terminal commands

To-Do:

- Create one repository per team(for freshies) or individually(for non-freshies) on GitHub and add all team members and AWorldOfChaos(Kartik) as collaborators.
- Complete the following Python task

[Python Task Week 1](https://www.notion.so/Python-Task-Week-1-bd9eb6c10a5e4b0fb6f65e059d1c3f95)

### Week 2 - Machine Learning

1. Machine Learning - Linear and Logistic Regression.
2. Primary Excercises on Linear and Logistic Regression using Pytorch

To-Do:

- Complete the first 4 weeks of the Andrew Ng Course on Coursera.
- Complete the following Machine Learning Exercises

[Linear Regression](https://www.notion.so/Linear-Regression-b611e01be8004f9289c1de9832248477)

[Recognizing Handwritten Digits](https://www.notion.so/Recognizing-Handwritten-Digits-1685b02aabb943c9a4628a63c92e05b5)

- Challenge Task

[Cats Vs Dogs](https://www.notion.so/Cats-Vs-Dogs-6556243f7c64408ab0cdab13db0a4e2b)

### Week 3 - Intro To Deep Learning

1. Dive into Deep Learning with Neural Networks
2. Study Convolutional Neural Networks

To-Do:

- Read the introductory chapter from “Introduction To Machine Learning - Gurney et. al.” and optionally the theory
- Watch the playlist on implementing Neural networks in pytorch
- Complete the following task

[Cats vs Dogs 2](https://www.notion.so/Cats-vs-Dogs-2-065dd59fe31148efb7e127be4b6cbc12)

### Week 4+5 - Dive into CNNs and Image Processing

1. Play around with Convolutional Neural Networks
2. Learn Basics of Image Processing
3. Learn about Kernels
4. Study the Problem of Super Resolution

To-Do

- Read the Material on Convolutional Neural Networks and Implement them on the Cats Vs Dogs task
- Read up on Kernels
- Read up on the Image Super-Resolution Problem

### Week 6 - Midsem Week

---

### Week 7 - Getting Started with Super Resolution

1. Extend Image Super-Resolution to Video Super-Resolution using Convolutional Neural Networks

To-Do:

- Complete Material on Video Super-Resolution Techniques.
- Identify dataset for final Project
- Submit Major Assignment

### Week 8+ - Final Code

1. Complete the Video Super-Resolution Project :)

To-Do:

- Code the Model for Video Super-Resolution using chosen Architecture
- Train the Model on the chosen Dataset
- Acquire Sample Results
- Improve Accuracy and Performance
- Submit Final Project Code with Results and Documentation

### Over and Beyond

1. Learn up On Blind Video Super-Resolution
2. Implement Deep Blind Video Super-Resolution
3. Explore other architectures for modeling degradation and implementing suitable image restoration

------

# Resources
Check the Resources folder

# Results
Check the individual mentees folders
