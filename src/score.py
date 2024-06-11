import sys
sys.path.append('Barbershop')
from Barbershop.losses import lpips, masked_lpips
from Barbershop.losses.style.style_loss import StyleLoss
from Barbershop.utils.bicubic import BicubicDownSample

import torch
import torch.nn as nn
import numpy as np

import skimage.metrics as sm

from src.utils import parse_yaml

class EvalMetrix():
    def __init__(self, opts) -> None:
        self.device = opts.device
        self.load_loss_fn()
        self.downsample_256 = BicubicDownSample(factor=opts.size // 256)

    def load_loss_fn(self):
        self._l2_loss_fn = nn.MSELoss().to(self.device)
        self._percept_loss_fn = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True if self.device=='cuda' else False)
        self._percept_loss_fn.eval()
        self._mask_percep_loss_fn = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=True if self.device=='cuda' else False
        )
        self._mask_percep_loss_fn.eval()
        self._style_loss_fn = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(self.device)
        self._style_loss_fn.eval()

        self.loss_dict = {
            'mse': self.l2_loss_fn, 
            'percept': self.percept_loss_fn, 
            'mask_percept': self.mask_percep_loss_fn, 
            'style': self.style_loss_fn, 
            'psnr': self.psnr, 
            'ssim': self.ssim, 
        }

    def _im_preprocess(self, im):
        # im_numpy = None
        # im_tensor = None
        if isinstance(im, np.ndarray):
            im_numpy = im.copy()
            im_tensor = torch.from_numpy(im / 255).permute(2, 0, 1).to(torch.float32).to(self.device)
        elif isinstance(im, torch.Tensor):
            im_numpy = im.detach().cpu().permute(1, 2, 0).numpy() * 255
            im_numpy = im_numpy.astype(np.uint8)
            im_tensor = im.detach().to(self.device)
        else:
            return None, None
        return im_numpy, im_tensor

    def im_preprocess(self, im1, im2, mask1=None, mask2=None, return_type = "np", unsqueeze=False, downsample=False):
        im1 = self._im_preprocess(im1)
        im2 = self._im_preprocess(im2)
        mask1 = self._im_preprocess(mask1)
        mask2 = self._im_preprocess(mask2)
        if return_type == "np":
            return im1[0], im2[0], mask1[0], mask2[0]
        elif unsqueeze:
            im1, im2, mask1, mask2 = im1[1].unsqueeze(0), im2[1].unsqueeze(0), mask1[1].unsqueeze(0), mask2[1].unsqueeze(0)
        else:
            im1, im2, mask1, mask2 = im1[1], im2[1], mask1[1], mask2[1]
        if downsample:
            im1, im2 = self.downsample_256(im1), self.downsample_256(im2)
            if mask1 is not None:
                mask1 = self.downsample_256(mask1)
            if mask2 is not None:
                mask2 = self.downsample_256(mask2)
        return im1, im2, mask1, mask2
    
    def psnr(self, im1, im2, mask1, mask2=None):
        im1, im2, mask1, mask2 = self.im_preprocess(im1, im2, mask1, mask2, return_type="np")
        return sm.peak_signal_noise_ratio(im1*mask1, im2*mask1)
    
    def ssim(self, im1, im2, mask1, mask2=None, win_size=3):
        im1, im2, mask1, mask2 = self.im_preprocess(im1, im2, mask1, mask2, return_type="np")
        return sm.structural_similarity(im1*mask1, im2*mask1, win_size=win_size)
    
    def l2_loss_fn(self, im1, im2, mask1, mask2=None, weight = 1.0):  # , **kwargs
        im1, im2, mask1, mask2 = self.im_preprocess(im1, im2, mask1, mask2, return_type="pt")
        if mask1 is None:
            return self._l2_loss_fn(im1, im2) * weight * 255 * 255
        return self._l2_loss_fn(im1*mask1, im2*mask1) * weight * 255 * 255

    def percept_loss_fn(self, im1, im2, mask1, mask2, weight = 1.0):
        im1, im2, mask1, mask2 = self.im_preprocess(im1, im2, mask1, mask2, return_type="pt")
        return self._percept_loss_fn(im1, im2).sum() * weight
    
    def mask_percep_loss_fn(self, im1, im2, mask1, mask2, weight = 1.0):
        im1, im2, mask1, mask2 = self.im_preprocess(im1, im2, mask1, mask2, return_type="pt")
        return self._mask_percep_loss_fn(im1, im2, mask1).sum() * weight

    def style_loss_fn(self, im1, im2, mask1, mask2, weight = 1.0):
        im1, im2, mask1, mask2 = self.im_preprocess(im1, im2, mask1, mask2, return_type="pt", unsqueeze=True, downsample = True)
        return self._style_loss_fn(im1*mask1[:,:1,:,:], im2*mask2[:,:1,:,:], mask1[:,:1,:,:], mask2[:,:1,:,:]) * weight
    
    def get_matrix(self, im1, im2, mask1, mask2):
        matrix = {}
        for key, value in self.loss_dict.items():
            score = value(im1, im2, mask1, mask2)
            if isinstance(score, torch.Tensor):
                score = score.item()
            matrix[key] = score
        return matrix
