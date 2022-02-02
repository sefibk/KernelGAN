# import cv2 library
from PIL import Image ,ImageOps , ImageDraw, ImageFont
import os
from threading import Event
from math import log10, sqrt
import torch
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable


def main():
    img_path1 = '/train/baboon.png'
    img_path2 = '/train/baboon.png'
    cwd = os.getcwd()
    img1 = Image.open(cwd + img_path1)
    img2 = Image.open(cwd + img_path2)
    convert_tensor = transforms.ToTensor()
    img1_tensor = convert_tensor(img1)
    img2_tensor = convert_tensor(img2)
    txt = make_the_text_image(str(PSNR(img1_tensor,img2_tensor)) +"/"+ str(SSIM(img1_tensor,img2_tensor)))
    imgs = visualizer(img1,img2)
    img = get_concat_v_blank(imgs, txt)
    img.show()

def visualizer(img1, img2):
    image1 = ImageOps.expand(img1, border=5, fill='black')
    image2 = ImageOps.expand(img2, border=5, fill='black')
    cwd = os.getcwd()
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.save(cwd + '/train/merge.png', "PNG")
    # new_image.show()
    return new_image

def make_the_text_image(txt):
    cwd = os.getcwd()
    img = Image.new('RGB', (100, 30), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "PSNR/SSIM " + txt + "db",  fill=(255, 255, 0))
    img.save(cwd + 'pil_text_font.png', "PNG")
    # img.show()
    return img

def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        print("Its the same image")
        return 1
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(img1, img2):
    img1_v = torch.unsqueeze(img1, 0)
    img2_v = torch.unsqueeze(img2, 0)
    # calculate ssim & ms-ssim for each image
    ssim_val = ssim(img1_v, img2_v, data_range=255, size_average=False)  # return (N,)
    ms_ssim_val = ms_ssim(img1_v, img2_v, data_range=255, size_average=False)  # (N,)
    return ssim_val.item()
    # print(ms_ssim_val.item())

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width)+10, im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

if __name__ == "__main__":
    main()