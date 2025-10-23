# 3D UNet for Prostate MRI Segmentation
**Author:** Qilin He (s4801815)  
**Course:** COMP3710 – Pattern Analysis 2025  
**Project 7 -** Segment the (downsampled) Prostate 3D data set with the normal 3D UNet  
**Date:** 23 October 2025  

---

## Project Overview
This project implements a **3D UNet** architecture for volumetric segmentation of 3D prostate MRI from [CSIRO Data Access Portal - Labelled weekly MR images of the male pelvis](https://data.csiro.au/collection/csiro:51392v2?redirected=true). The MRI data has been pre-labeled with 6 regions – Background, Body, Bones, Bladder, Rectum, and Prostate. The goal of this project is to achieve **Dice Similarity coefficient ≥ 0.7** for all classes on the test set.  

## Appendix/Reference
Dowling, Jason; & Greer, Peter (2021): Labelled weekly MR images of the male pelvis. v2. CSIRO. Data Collection. https://doi.org/10.25919/45t8-p065