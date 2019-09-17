# "KernelGAN"
# Blind Super-Resolution Kernel Estimation using an Internal-GAN
### Official implementation for paper by: Sefi Bell-Klgiler, Assaf Shocher, Michal Irani

Paper: https://arxiv.org/abs/1909.06581

Project page: http://www.wisdom.weizmann.ac.il/~vision/kernelgan/  

// **Accepted NeurIPS 2019 (oral)**


# Usage:

## Quick usage on your data:  
python train.py --input-dir <input_image_path>

## Extra configurations:  
"--X4" : estimate the X4 kernel

"--SR" : perform ZSSR using the estimated kernel

"--real" : perform SR with real-image configuration (effects the ZSSR method)

"--output-dir" : define the output folder for the images
