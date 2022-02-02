import os
import torch
from math import log10, sqrt
from PIL import Image ,ImageOps , ImageDraw, ImageFont
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from configs import Config
from train import create_params, train
from argparse import Namespace
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run_zssrgan(filename, dir):
    args = Namespace(DL=True, X4=False, UK=True, input_dir=dir, noise_scale=1.0, output_dir='results', real=False)
    conf = Config().parse(create_params(filename, args))
    return train(conf)

def compare_with_gt(img_path, gt_path):
    img = Image.open(img_path)
    gt = Image.open(gt_path)
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img)
    gt_tensor = convert_tensor(gt)
    txt = make_the_text_image("(PSNR/SSIM) = (" + str(PSNR(img_tensor,gt_tensor)) +"/"+ str(SSIM(img_tensor,gt_tensor)) + ")")
    imgs = visualizer(img,gt)
    img = get_concat_v_blank(imgs, txt)
    img.show()

def compare_SR_results(img1_path, img2_path, img1_txt, img2_txt):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img1)
    gt_tensor = convert_tensor(img2)
    img1 = get_concat_v_blank(img1, make_the_text_image(img1_txt))
    img2 = get_concat_v_blank(img2, make_the_text_image(img2_txt))
    imgs = visualizer(img1,img2)
    imgs.show()

def visualizer(img1, img2):
    image1 = ImageOps.expand(img1, border=5, fill='black')
    image2 = ImageOps.expand(img2, border=5, fill='black')
    cwd = os.getcwd()
    image1_size = image1.size
    new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.save(cwd + '/train/merge.png', "PNG")
    return new_image

def make_the_text_image(txt):
    cwd = os.getcwd()
    img = Image.new('RGB', (100, 30), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "PSNR/SSIM " + txt + "db",  fill=(255, 255, 0))
    img.save(cwd + 'pil_text_font.png', "PNG")
    return img

def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        print("Its the same image")
        return '\u221e'
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

if __name__ == '__main__':
    cwd = os.getcwd()
    train_folder_path = cwd + "\\data\\train"
    gt_folder_path = cwd + "\\data\\gt"
    for filename in os.listdir(os.path.abspath(train_folder_path)):
        result_path = run_zssrgan(filename, train_folder_path)
        gt_path = os.path.join(gt_folder_path, filename)
        compare_with_gt(result_path, gt_path)
