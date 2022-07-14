[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7954983&assignment_repo_type=AssignmentRepo)
# Seasons Of Code 2022 - Video Super Resolution.
The required files for testing will be available in [this folder](https://drive.google.com/drive/folders/19iuC6Q3snaowbprPOkqkn1WmRYISRkzI?usp=sharing).
P.S.- It opens only using LDAP.

We have used an SRCNN to conduct frame-wise video super resolution, using _sparce coding_ methods, using references from [this paper](https://arxiv.org/pdf/1501.00092v3.pdf).

The architechture of the network is the same as the one discussed in the paper, consisting of a 9-1-5 filter sizes in the convolutional layers. The total learnable parameters in the Convolutional Neural Network implemented in this code is __8032__.

The metric used was psnr, the values are given in the results.

___Frosty-Chaos___
