import os
import torch
import pandas as pd
import scipy.io as sio
from PIL import Image ,ImageOps , ImageDraw
from main import create_params
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM
from Utils import configs
from train import train, train_zssr_only
from argparse import Namespace
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run(filename, dir, use_kernel, disc_loss, kernel=None, real=False):
    args = Namespace(DL=disc_loss, X4=False, UK=use_kernel, input_dir=dir, noise_scale=1.0, output_dir='results', real=real, type="fixed")
    configs.parse(create_params(filename, args))
    if use_kernel and kernel is None:
        return train()
    else:
        return train_zssr_only(kernel)

def compare_with_gt(img_path, gt_path):
    img = Image.open(img_path)
    gt = Image.open(gt_path)
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img)[:3,:,:]
    gt_tensor = convert_tensor(gt)[:3,:,:]
    #txt = make_the_text_image("(PSNR/SSIM) = (" + str(psnr(img_tensor,gt_tensor)) +"/"+ str(SSIM(img_tensor,gt_tensor)) + ")")
    #imgs = visualizer(img,gt)
    #img = get_concat_v_blank(imgs, txt)
    #img.show()
    return psnr(img_tensor,gt_tensor), SSIM(img_tensor,gt_tensor)

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

def visualizer(img1, img2, img3, filename):
    image1 = ImageOps.expand(img1, border=5, fill='black')
    image2 = ImageOps.expand(img2, border=5, fill='black')
    image3 = ImageOps.expand(img3, border=5, fill='black')
    cwd = os.getcwd()
    image1_size = image1.size
    new_image = Image.new('RGB', (3 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.paste(image3, (2 * image1_size[0], 0))
    new_image.save(cwd + '/Compared/' + filename, "PNG")
    return new_image

def make_the_text_image(txt):
    cwd = os.getcwd()
    img = Image.new('RGB', (250, 30), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), txt,  fill=(255, 255, 0))
    img.save(cwd + 'pil_text_font.png', "PNG")
    return img

#Taken from HW3
def psnr(im, ref, margin=2):
  """
   Args:
    im (torch.Tensor): Image to be evaluated.
    Has shape `(num_channels, height, width)`.
    ref (torch.Tensor): reference image.
    Has shape `(num_channels, height, width)`.
  Returns:
    psnr (int): psnr value of the images.
  """
  # assume images are tensors float 0-1.
  # im, ref = (im*255).round(), (ref*255).round()
  rgb2gray = torch.Tensor([65.481, 128.553, 24.966]).to(im.device)[None, :, None, None]
  gray_im = torch.sum(im * rgb2gray, dim=1) + 16
  gray_ref = torch.sum(ref * rgb2gray, dim=1) + 16
  clipped_im = torch.clamp(gray_im, 0, 255).squeeze()
  clipped_ref = torch.clamp(gray_ref, 0, 255).squeeze()
  shaved_im = clipped_im[margin:-margin, margin:-margin]
  shaved_ref = clipped_ref[margin:-margin, margin:-margin]
  psnr_tensor = 20 * torch.log10(torch.tensor(255.)) -10.0 * ((shaved_im) - (shaved_ref)).pow(2.0).mean().log10()
  return float(round(psnr_tensor.item(),3))

def SSIM(img1, img2):
    img1_v = torch.unsqueeze(img1, 0)*255
    img2_v = torch.unsqueeze(img2, 0)*255
    # calculate ssim & ms-ssim for each image
    ssim_val = ssim(img1_v, img2_v, data_range=255, size_average=False)  # return (N,)
    ms_ssim_val = ms_ssim(img1_v, img2_v, data_range=255, size_average=False)  # (N,)
    return float(round(ms_ssim_val.item(),3))
    # print(ms_ssim_val.item())

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width)+10, im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def psnr_ssim_comparison():
    cwd = os.getcwd()
    train_folder_path = cwd + "\\data\\train"
    gt_folder_path = cwd + "\\data\\gt"
    run_list = []
    avg_psnr = 0
    avg_ssim = 0
    avg_ker_psnr = 0
    avg_ker_ssim = 0
    avg_zssr_only_psnr = 0
    avg_zssr_only_ssim = 0
    count = 0
    for filename in os.listdir(os.path.abspath(train_folder_path)):
        zssr_path = run(filename, train_folder_path, use_kernel=False, disc_loss=False)
        zssrgan_path = run(filename, train_folder_path, use_kernel=True, disc_loss=True)
        mat_path = os.path.join(os.path.dirname(zssrgan_path), filename.split('.')[0] + "_kernel_x2.mat")
        mat = sio.loadmat(mat_path)
        final_kernel = mat['Kernel']
        kergan_path = run(filename, train_folder_path, use_kernel=True, disc_loss=False, kernel=final_kernel)

        gt_path = os.path.join(gt_folder_path, filename)
        zssrgan_psnr, zssrgan_ssim = compare_with_gt(zssrgan_path, gt_path)
        kergan_psnr, kergan_ssim = compare_with_gt(kergan_path, gt_path)
        zssr_psnr, zssr_ssim = compare_with_gt(zssr_path, gt_path)
        print("zssrgan-psnr/ssim = " + str(zssrgan_psnr) + "/" + str(zssrgan_ssim))
        print("kergan-psnr/ssim = " + str(kergan_psnr) + "/" + str(kergan_ssim))
        print("zssrgan-psnr/ssim = " + str(zssr_psnr) + "/" + str(zssr_ssim))
        run_list.append({'image_path': str(filename),
                         'zssrgan-psnr/ssim': str(zssrgan_psnr) + "/" + str(zssrgan_ssim),
                         'kergan-psnr/ssim': str(kergan_psnr) + "/" + str(kergan_ssim),
                         'zssr-psnr/ssim': str(zssr_psnr) + "/" + str(zssr_ssim)
                         })
        avg_psnr += zssrgan_psnr
        avg_ssim += zssrgan_ssim
        avg_ker_psnr += kergan_psnr
        avg_ker_ssim += kergan_ssim
        avg_zssr_only_psnr += zssr_psnr
        avg_zssr_only_ssim += zssr_ssim
        count += 1

    # compute average psnr
    avg_psnr = avg_psnr / count
    avg_ker_psnr = avg_ker_psnr / count
    run_list.append({'image_path': 'PSNR_AVG',
                     'zssrgan-psnr': round(avg_psnr, 3),
                     'kergan-psnr': round(avg_ker_psnr, 3),
                     'zssr-psnr': round(avg_zssr_only_psnr_psnr, 3)
                     })

    # compute average ssim
    avg_ssim = avg_ssim / count
    avg_ker_ssim = avg_ker_ssim / count
    run_list.append({'image_path': 'SSIM_AVG',
                     'zssrgan-ssim': round(avg_ssim, 3),
                     'kergan-ssim': round(avg_ker_ssim, 3),
                     'zssr-ssim': round(avg_zssr_only_ssim, 3)
                     })

    # create results file
    run_df = pd.DataFrame(run_list)
    run_df.to_csv(cwd + '\summarize.csv')

def visual_comparison():
    cwd = os.getcwd()
    train_folder_path = cwd + "\\data\\real_photos"
    # real_folder_path = cwd + "\\data\\test_photos"
    for filename in os.listdir(os.path.abspath(train_folder_path)):
        zssr_path = run(filename, train_folder_path, use_kernel=False, disc_loss=False, real=True)
        zssrgan_path = run(filename, train_folder_path, use_kernel=True, disc_loss=True , real=True)
        mat_path = os.path.join(os.path.dirname(zssrgan_path), filename.split('.')[0] + "_kernel_x2.mat")
        mat = sio.loadmat(mat_path)
        final_kernel = mat['Kernel']
        kergan_path = run(filename, train_folder_path, use_kernel=True, disc_loss=False, kernel=final_kernel, real=True)
        # real_path = os.path.join(real_folder_path, filename)
        zssr_img = Image.open(zssr_path)
        zssrgan_img = Image.open(zssrgan_path)
        kergan_img = Image.open(kergan_path)
        # real_img = Image.open(real_path)
        visualizer(zssr_img,zssrgan_img,kergan_img,filename)


if __name__ == '__main__':
    psnr_ssim_comparison()
    visual_comparison()