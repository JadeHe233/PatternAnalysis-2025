# 3D U-Net for Prostate MRI Segmentation
**Author:** Qilin He (s4801815)  
**Course:** COMP3710 – Pattern Analysis 2025  
**Project 7 -** Segment the (downsampled) Prostate 3D data set with the normal 3D U-Net  
**Date:** 23 October 2025  

---

## Project Overview
This project implements a **3D U-Net** architecture for volumetric segmentation of 3D prostate MRI from [CSIRO Data Access Portal - Labelled weekly MR images of the male pelvis](https://data.csiro.au/collection/csiro:51392v2?redirected=true). The MRI data has been pre-labeled with 6 regions – Background, Body, Bones, Bladder, Rectum, and Prostate. The goal of this project is to achieve **Dice Similarity coefficient ≥ 0.7** for all classes on the test set.  

## Model Architecture
The 3D U-Net model implemented in this project is based on the architecture suggested by Çiçek et al. (2016). The 3D U-Net architecture is proposed to solve the difficulties of segmenting volumetric biomedical data. To perform segmentation with the original 2D U-Net architecture, the volumetric data first need to be divided into 2D slices, where neighbouring slices would show almost no variation, and the 2D slices were processed and learnt independently in the 2D U-Net. Therefore, the annotation of large volume medical data with segmentation labels, in a slice-by-slice manner is very inefficient and does not generalise well (Çiçek et al, 2016).
<p align="center">
  <img src="Figures/3dunet_arch.png" width="700"/>
  <br><em>Figure 1: 3D U-Net Architecture</em>
</p>
Similar to the structure of the 2D U-Net, The 3D U-Net is also a convolutional autoencoder that replaces all the 2D operations with 3D convolutions, so the volumetric medical data can be processed by voxels instead of pixels of independent 2D slices. Figure 1 shows the architecture of the 3D U-Net suggested by Çiçek et al. (2016). In the contraction path, each level consists of two sets of a 3x3x3 convolution followed by a batch normalisation (BN) and a rectified linear unit (ReLu), then a max pooling layer of 2x2x2 with a stride of 2 to extract feature information and reduce the spatial resolution. Since the architecture is symmetric, the expansion path mirrors the structure of the contraction path. Each level starts with a 2x2x2 transpose convolution to recover the spatial resolution, followed by two sets of a 3x3x3 convolution, a BN and a ReLu. This helps to segment the locations of objects in the data. Additionally, each level has a skip path that concatenates the encoder with its corresponding decoder to pass the high-resolution spatial information that is lost through down-sampling. Finally, a 1x1x1 convolution is used to reduce the output channels to the number of labels - 3 in Figure 1. For this architecture, the weighted softmax loss function is used. It allows the network to train on sparse annotations. By setting the weight of unlabeled voxel to 0, the model can only learn from the labelled voxels to generalise the whole volume (Çiçek et al, 2016).  

## Appendix/Reference
Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In: Ourselin, S., Joskowicz, L., Sabuncu, M., Unal, G., Wells, W. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016. MICCAI 2016. Lecture Notes in Computer Science(), vol 9901. Springer, Cham. https://doi.org/10.1007/978-3-319-46723-8_49

Dowling, Jason; & Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data Collection. https://doi.org/10.25919/45t8-p065