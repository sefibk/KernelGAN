# "KernelGAN" Blind Super-Resolution Kernel Estimation using an Internal-GAN
### Official implementation for paper by: Sefi Bell-Klgiler, Assaf Shocher, Michal Irani

Paper: ????https://arxiv.org/abs/1712.06087  ????
Project page: ???? http://www.wisdom.weizmann.ac.il/~vision/zssr/ (See our results and visual comparison to other methods)????

**Accepted NuerIPS 2019**

----------
![sketch](/figs/sketch.png)
----------
If you find our work useful in your research or publication, please cite our work:

```
??????
```
----------
# Usage:

## Quick usage on your data:  
Place images in the """<KernelGAN path>/test_images"""
Results will be saved to """<KernelGAN path>/results"""
```
python train.py
```

## Extra usages:  
For scale X4 kernel - add to the image name: """X4"""
In case you are interested in SR - add to the image name """ZSSR"""
In case you are dealing with a real image - add the word """real""" to the image name (relevant for ZSSR configuration)

## General usage:
```
python run_ZSSR.py <config> <gpu-optional>
```
While ``` <config> ``` is an instance of configs.Config class (at configs.py) or 0 for default configuration.  
Please see configs.py to determine configuration (data paths, scale-factors etc.)  
``` <gpu-optional> ``` is an optional parameter to determine how to use available GPUs (see next section).

For using given kernels, you must have a kernels for each input file and each scale-factor named as follows:  
``` <inpu_file_name>_<scale_factor_ind_starting_0>.mat ```  
Kernels are MATLAB files containing a matrix named "Kernel".  

If gound-truth exists and true-error monitoring is wanted, then ground truth should be named as follows:  
``` <inpu_file_name>_gt.png ```  
