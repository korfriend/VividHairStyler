import os
import cv2
import sys
import gdown
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from src.models.networks.models import build_model

class ARGS():
    encoder = 'resnet50_GN_WS'
    decoder = 'fba_decoder'
    weights = 'FBA_Matting/FBA.pth'



class PredDataset(Dataset):
    ''' Reads image and trimap pairs from folder.

    '''

    def __init__(self, img_dir, trimap_dir):
        self.img_dir, self.trimap_dir = img_dir, trimap_dir
        self.img_names = [x for x in os.listdir(self.img_dir) if 'png' in x]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        image = read_image(os.path.join(self.img_dir, img_name))
        trimap = read_trimap(os.path.join(self.trimap_dir, img_name))
        pred_dict = {'image': image, 'trimap': trimap, 'name': img_name}

        return pred_dict


def read_image(name):
    return (cv2.imread(name) / 255.0)[:, :, ::-1]


def read_trimap(name):
    trimap_im = cv2.imread(name, 0) / 255.0
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap

def np_to_torch(x, permute=True):
    if permute:
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
    else:
        return torch.from_numpy(x)[None, :, :, :].float().cuda()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale



def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    ''' Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    '''
    h, w = trimap_np.shape[:2]
    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(
            trimap_transform(trimap_scale_np), permute=False)
        image_transformed_torch = normalise_image(
            image_torch.clone())

        output = model(
            image_torch,
            trimap_torch,
            image_transformed_torch,
            trimap_transformed_torch)
        output = cv2.resize(
            output[0].cpu().numpy().transpose(
                (1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)

    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]

    return fg, bg, alpha


class Trimap():
    TRIMAP_UNKNOWN = 128
    TRIMAP_HAIR = 255
    TRIMAP_BACKGROUND = 0
    

    def __init__(self) -> None:
        self.args = ARGS()
        if not os.path.exists(self.args.weights):
            gdown.download("https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1", output=self.args.weights)
        self.model = build_model(self.args).cuda()
        self.model.eval()

    def mask_to_trimap(self, image, mask, kernel_size=20):
        image = image / 255.0

        # Initialize trimap with unknown region
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        trimap_im = np.full(mask.shape, self.TRIMAP_UNKNOWN, dtype=np.uint8)

        trimap_im[eroded_mask == self.TRIMAP_HAIR] = self.TRIMAP_HAIR  
        trimap_im[dilated_mask == self.TRIMAP_BACKGROUND] = self.TRIMAP_BACKGROUND  
        trimap_im = trimap_im/255.0
        h, w = trimap_im.shape
        trimap = np.zeros((h, w, 2))
        trimap[trimap_im == 1, 1] = 1
        trimap[trimap_im == 0, 0] = 1
        
        fg, bg, alpha = pred(image, trimap, self.model)
        return fg, bg, alpha