import os
import sys
import torch
import lpips
sys.path.append(".")
import cv2
from glob import glob
from libs.utils.metrics.metrics import psnr_metric
from libs.utils.metrics.pytorch_ssim import ssim as ssim_func
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

image_dir = 'exp/CoreView_male-4-casual_1709841604_run_mono_1_1_3_true/wandb/run-20240311_210758-826z60oq/images'

image_list = sorted(glob(os.path.join(image_dir, '*.png')))
order = 0


# i=0
# new_image_list = []
# while i < len(image_list)//(4*2):
#     s = i*8
#     e = s+7
    
#     group = image_list[s:e]
#     new_image_list.append(image_list[s+order*2])
#     new_image_list.append(image_list[s+order*2+1])
#     i = i + 1

# image_list = new_image_list
psnr_list = []
ssim_list = []
lpips_list = []

# lpips_func = lpips.LPIPS(net='vgg').cuda()
# lpips_func = lpips.LPIPS(net='alex').cuda()
lpips_func = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim_func = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

for i in range(len(image_list)//2):
    img1_path = image_list[2*i]
    img2_path = image_list[2*i+1]
    img1 = cv2.imread(img1_path)/255
    img2 = cv2.imread(img2_path)/255
    img1_ = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).cuda().float()
    img2_ = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).cuda().float()
    
    psnr = psnr_metric(img1_, img2_)
    psnr_list.append(psnr)
    
    ssim = ssim_func(img1_, img2_)
    ssim_list.append(ssim.item())
    
    lpips1 = lpips_func.forward(img2_, img1_)
    lpips_list.append(lpips1.item())
    
    print(f"PSNR: {psnr}, SSIM: {ssim.item()}, LPIPS: {lpips1.item()}")
    
print('PSNR:', sum(psnr_list)/len(psnr_list))
print('SSIM:', sum(ssim_list)/len(ssim_list))
print('LPIS:', sum(lpips_list)/len(lpips_list))