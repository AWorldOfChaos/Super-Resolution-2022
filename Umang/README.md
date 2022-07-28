# Seasons of Code 2022 - Video Super Resolution (VSR)

# Problem Statement
Consider a single low-resolution image, we first upscale it to the desired size using bicubic interpolation to obtain Y. Our goal is to recover from Y, an image F(Y) that is as similar as possible to the actual high resolution image X. The Image Super Resolution Problem has been taken further, into what we can call "Naive Video Super Resolution", where each frame of the low resolution video is processed as above to obtain it in high resolution, and all these frames are put back together to create the higher resolution video.
best.pth is the best epoch with psnr value 36.64
