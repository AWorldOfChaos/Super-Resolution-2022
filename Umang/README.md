# Seasons of Code 2022 - Video Super Resolution (VSR)

**PROBLEM STATEMENT** 

Consider a single low-resolution image, we first upscale it to the desired size using bicubic interpolation to obtain Y. Our goal is to recover from Y, an image F(Y) that is as similar as possible to the actual high resolution image X. The Image Super Resolution Problem has been taken further, into what we can call "Naive Video Super Resolution", where each frame of the low resolution video is processed as above to obtain it in high resolution, and all these frames are put back together to create the higher resolution video.

For training I have used the T91 files in HD5 binary format which can be found in the google drive link: https://drive.google.com/drive/folders/1tyzFZLuH5MHsPUdfRfxf3XIJyae-N1vb?usp=sharing

On these input videos, bicubic interpolation is applied to get the input for neural networks.

Three convolutional levels have been created with a ReLu activation layer, details about each layer can be seen in the "models.py" file uploaded in the directory. I have padded the images. Only scale 2 (x2) has been used and the details about its parameters are available in "parameters.txt" file. 


For training the model on the input images and videos provided in the folder "iamge_sr" and "VideoSR", I have used the commands as specified in the "run.txt" file. 

My final output videos are available in the "VideoSR" and "image_sr" folders. Various files of "Image_bicubic_x2", "Image_srcnn_x2", "Video_bicubic_x2" and "Video_vsr_x2" are available. All these files accurately solve the problem statement. 


Note: best.pth is the best epoch with psnr value 36.64. Epoch number 367.
