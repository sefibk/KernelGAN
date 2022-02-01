# Zero Shot Super Resolution using KernelGAN estimated Kernel and Discriminator
# "ZSSRGAN"
### Dan Bar, Neta Shaul, Yeari Vigder 

## Usage:

### Quick usage on your data:  
To run ZSSRGAN on all images in <input_image_path>:

``` python train.py --input-dir <input_image_path> ```


This will produce kernel estimations and super resolution image in the results folder

### Extra configurations:  
```--X4``` : Estimate the X4 kernel

```--UK``` : Perform ZSSR using the estimated kernel

```--DL``` : Perform ZSSR using the discriminator loss

```--real``` : Real-image configuration (effects only the ZSSR)

```--output-dir``` : Output folder for the images (default is results)


### Data:
Download the DIV2KRK dataset: [dropbox](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip)