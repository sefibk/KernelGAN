# "KernelGAN" Blind Super-Resolution Kernel Estimation using an Internal-GAN
### Official implementation for paper by: Sefi Bell-Klgiler, Assaf Shocher, Michal Irani

Paper: ????https://arxiv.org/abs/1712.06087  ????
Project page: ???? http://www.wisdom.weizmann.ac.il/~vision/zssr/ (See our results and visual comparison to other methods)????

// **Accepted NeurIPS 2019**

----------
\\ ![sketch](/figs/sketch.png)
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

## Extra configurations:  
For scale X4 kernel - add to the image name: """X4"""
In case you are interested in SR - add to the image name """ZSSR"""
In case you are dealing with a real image - add the word """real""" to the image name (relevant for ZSSR configuration)
