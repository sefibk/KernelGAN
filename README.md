# Blind Super-Resolution Kernel Estimation using an Internal-GAN
# "KernelGAN"
### Sefi Bell-Klgiler, Assaf Shocher, Michal Irani 
*(Official implementation)*

Paper: https://arxiv.org/abs/1909.06581

Project page: http://www.wisdom.weizmann.ac.il/~vision/kernelgan/  

**Accepted NeurIPS 2019 (oral)**


## Usage:

### Quick usage on your data:  
To run KernelGAN on all images in <input_image_path>:

``` python train.py --input-dir <input_image_path> ```


This will produce kernel estimations in the results folder

### Extra configurations:  
```--X4``` : estimate the X4 kernel

```--SR``` : perform ZSSR using the estimated kernel

```--real``` : real-image configuration (effects only the ZSSR)

```--output-dir``` : output folder for the images (default is results)
