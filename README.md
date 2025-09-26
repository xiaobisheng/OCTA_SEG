# OCTA_SEG
This is the code for OCTA segmentation task.

## Raw Dataset:
We use OCTA-500 dataset, please find the dataset here: https://ieee-dataport.org/open-access/octa-500.

## Data processing
depth_map_generation.py: This file visualizes the 3D data using open3d and generates DCCM and DCIM.

## Baseline methods
A few popular methods are collected and implemented.
Codes for this part are mostly from te following sources:

https://github.com/chaosallen/IPNV2_pytorch

https://github.com/milesial/Pytorch-UNet

https://github.com/rishikksh20/ResUnet

https://github.com/Andy-zhujunwen/UNET-ZOO

learning rate scheduler codes are from: https://github.com/sooftware/pytorch-lr-scheduler/tree/main

## Depth Enhanced Cascaded Framework for OCTA Segmentation

OCTA_2D is the code for 'Depth Enhanced Cascaded Framework for OCTA Segmentation'
