import os
import cv2
import sys
import gdown
import torch

import numpy as np
import matplotlib.pyplot as plt

from src.models.FBA_Matting.demo import np_to_torch, pred, scale_input
from src.models.FBA_Matting.dataloader import read_image, read_trimap
from src.models.FBA_Matting.networks.models import build_model
import torch
import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt
import gdown


class ARGS():
    encoder = 'resnet50_GN_WS'
    decoder = 'fba_decoder'
    weights = 'src/models/FBA_Matting/FBA.pth'



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