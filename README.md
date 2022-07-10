# Super-Resolution
This repository will host the work done as a part of Seasons of Code 2022 - Video Super Resolution under the Web & Coding Club of IIT Bombay. 

## Introduction

Super Resolution is the process of recovering a High Resolution (HR) image from a given Low Resolution (LR) image. An image may have a “lower resolution” due to a smaller spatial resolution (i.e. size) or due to a result of degradation (such as blurring). Video super-resolution is the task of upscaling a video from a low-resolution to a high-resolution.

Existing video super-resolution (SR) algorithms usually assume that the blur kernels in the degradation process are known and do not model the blur kernels in the restoration. However, this assumption does not hold for blind video SR and usually leads to over-smoothed super-resolved frames. In this paper, we propose an effective blind video SR algorithm based on deep convolutional neural networks (CNNs). Our algorithm first estimates blur kernels from low-resolution (LR) input videos. Then, with the estimated blur kernels, we develop an effective image deconvolution method based on the image formation model of blind video SR to generate intermediate latent frames so that sharp image contents can be restored well. To effectively explore the information from adjacent frames, we estimate the motion fields from LR input videos, extract features from LR videos by a feature extraction network, and warp the extracted features from LR inputs based on the motion fields. Moreover, we develop an effective sharp feature exploration method that first extracts sharp features from restored intermediate latent frames and then uses a transformation operation based on the extracted sharp features and warped features from LR inputs to generate better features for HR video restoration. We formulate the proposed algorithm into an end-to-end trainable framework and show that it performs favorably against state-of-the-art methods.

## Mentors

- Kartik Gokhale
- Hastyn Doshi

## Mentees

- Sahil Gawade
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

## **Git and Github**

We'll start by learning version control as it's really important to maintain and organize projects.

- [Resources compiled for the git session conducted by me for SOC mentees](https://www.notion.so/Version-Control-with-Git-51504dd7484e446aa5f5a50b757a29b4)
- For beginners to using the terminal checkout this video:
    
    [Introduction to Linux and Basic Linux Commands for Beginners](https://youtu.be/IVquJh3DXUA)
    
- A git crash course to begin with: [](https://youtu.be/SWYqp7iY_Tc)
    
    [Git & GitHub Crash Course For Beginners](https://youtu.be/SWYqp7iY_Tc)
    
- An awesome interactive way to learn git:
    
    [Git Immersion](https://gitimmersion.com/)
    

## **Python Basics**

- A good place to start with Python coding is the micro-course on Python, available on Kaggle. It provides tutorials in the form of Jupyter notebooks, and hands-on coding experience in the form of Kaggle kernels, which are modeled as Jupyter notebooks too. Here is the link: [https://www.kaggle.com/learn/python](https://www.kaggle.com/learn/python)
- Following tutorials from the Krittika Summer Projects Tutorials are relevant for the purpose of this project: 1, 2, 3, 4, 7, 9, 11, and 14. Here is the link: [https://github.com/krittikaiitb/tutorials](https://github.com/krittikaiitb/tutorials)
- An alternative, to those who prefer youtube, can utilize the following video

[Learn Python - Full Course for Beginners [Tutorial]](https://www.youtube.com/watch?v=rfscVS0vtbw&t=3683s)

- Alternatively, you can use this as well

[Learn Python - Free Interactive Python Tutorial](https://www.learnpython.org/)

## **Supplementary Material for Probability**

Brushing up on probability, and understanding common probability distributions will help you understand and appreciate the concepts better. I'm attaching a few resources you can refer to for brushing up on probability. You can go through them at your own pace.

- Basic Probability: [](https://www.youtube.com/playlist?list=PLvxOuBpazmsOGOursPoofaHyz_1NpxbhA)(a very short playlist)
    
    [Basics of Probability](https://www.youtube.com/playlist?list=PLvxOuBpazmsOGOursPoofaHyz_1NpxbhA)
    
- Probability Distributions:
    
    [https://drive.google.com/drive/folders/1Y_rU4bl4vgMhNrIzrXNxSEO5dyx17j6T?usp=sharing](https://drive.google.com/drive/folders/1Y_rU4bl4vgMhNrIzrXNxSEO5dyx17j6T?usp=sharing)
    
- **Alternate resource:** This is a very awesome interactive website to learn/revise probability concepts: [](https://seeing-theory.brown.edu/)

[Seeing Theory](https://seeing-theory.brown.edu/)

- Bayesian Statistics - Useful for Prior Statistics

[17. Bayesian Statistics](https://www.youtube.com/watch?v=bFZ-0FH5hfs)

## Machine Learning

- Machine Learning by Andrew Ng

[Machine Learning](https://www.coursera.org/learn/machine-learning)

- Pytorch Tutorial In Python

[Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M)

- TensorFlow Library: You need to do any *one* of Pytorch/Tensorflow. Do NOT waste time doing both *at the moment*.

[TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial](https://www.youtube.com/watch?v=tPYj3fFJGjk)

## Deep Learning

- Textbook on Neural Networks

[](https://www.inf.ed.ac.uk/teaching/courses/nlu/assets/reading/Gurney_et_al.pdf)

- Pytorch Neural Networks Playlist

[Introduction - Deep Learning and Neural Networks with Python and Pytorch p.1](https://www.youtube.com/watch?v=BzcBsTou0C0)

- 3b1b Video on Neural Networks

[But what is a neural network? | Chapter 1, Deep learning](https://www.youtube.com/watch?v=aircAruvnKk&vl=en)

## Convolutional Neural Networks

- Convolutional Neural Networks

[](https://en.wikipedia.org/wiki/Convolutional_neural_network)

- MIT OCW Lecture on CNNs

[MIT 6.S191 (2020): Convolutional Neural Networks](https://www.youtube.com/watch?v=iaSUYvmCekI)

- CNNs in Pytorch

[PyTorch Tutorial 14 - Convolutional Neural Network (CNN)](https://www.youtube.com/watch?v=pDdP0TFzsoQ)

- 3D Convolutional Neural Networks

[Step by Step Implementation: 3D Convolutional Neural Network in Keras](https://towardsdatascience.com/step-by-step-implementation-3d-convolutional-neural-network-in-keras-12efbdd7b130)

## Image Processing

- Kernels in Image Processing

[Image Kernels explained visually](https://setosa.io/ev/image-kernels/)

- Image Super-Resolution With CNNs

[](https://arxiv.org/pdf/1501.00092v3.pdf)

- Image Super-Resolution With GANs

[](https://arxiv.org/pdf/1609.04802v5.pdf)

## Video Super-Resolution

- Video Super-Resolution With Convolutional Neural Networks

[](https://ivpl.northwestern.edu/wp-content/uploads/2019/02/07444187.pdf)

- Video Super-resolution using 3D Convolutional Neural Networks

[](https://arxiv.org/pdf/1812.09079v2.pdf)

## Deep Blind Video Super-Resolution

- **ICCV 2021 Paper** on the Same

[](https://openaccess.thecvf.com/content/ICCV2021/papers/Pan_Deep_Blind_Video_Super-Resolution_ICCV_2021_paper.pdf)

- **Temporal Kernel Consistency for Blind Video Super-Resolution**

[Temporal Kernel Consistency for Blind Video Super-Resolution](https://www.arxiv-vanity.com/papers/2108.08305/)

## Miscellaneous Resources

### Good coding practices

- Corey's tips and tricks for coding in Python
    
    [10 Python Tips and Tricks For Writing Better Code](https://youtu.be/C-gEQdGVXbk)
    
- You don't need to install Pylint. Most IDEs like PyCharm and VSCode integrate this and give you the required tips
    
    [Pylint Tutorial - How to Write Clean Python](https://youtu.be/fFY5103p5-c)
    
- Optional:
    
    [Raymond Hettinger - Beyond PEP 8 -- Best practices for beautiful intelligible code - PyCon 2015](https://youtu.be/wf-BqAjZb8M)
    

### Brief tutorials for useful tools

- Markdown:
    - Reading Tutorial: [http://agea.github.io/tutorial.md/](http://agea.github.io/tutorial.md/)
    - Video Tutorial:
        
        [Markdown Crash Course](https://youtu.be/HUBNt18RFbo)
        
- Terminal Commands:

[Beginner's Guide To The Linux Terminal](https://www.youtube.com/watch?v=s3ii48qYBxA)

- Vim:
    - Update web version of the user manual (recommended)
        
        [Vim: usr_toc.txt](https://vimhelp.org/usr_toc.txt.html#user-manual)
        
    - Run `vimtutor` on the command line (recommended)
    - Another short primer
        
        [Vim 101: A Beginner's Guide to Vim - Linux.com](https://www.linux.com/training-tutorials/vim-101-beginners-guide-vim/)

## Major Assignment 1

Join Assignment 1 on Github classroom. All details are available there

[Build software better, together](https://classroom.github.com/a/kby66-lI)

## Final Project Code

The final submissions which are successful will be listed here.

Passing Criteria:- Completing the naive VSR algorithm and pipeline
