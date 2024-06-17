import torch
import numpy as np

from .networks.level_mapper import LevelMapper
from ..utils.data_utils import load_latent_W

class Bald():

    def __init__(self, model_path = "", device="cuda") -> None:
        self.device = device
        self.mapper = LevelMapper(input_dim=512).eval().to(device)
        ckpt = torch.load(model_path)
        self.alpha = float(ckpt['alpha']) * 1.2
        self.mapper.load_state_dict(ckpt['state_dict'], strict=True)


    def make_bald(self, latent_W_1, mix_ratio: float = 0.8, style_range = range(7, 18)):
        latent_W_1  = load_latent_W(latent_W_1)
    
        # mapper_input = 
        # mapper_input_tensor = mapper_input.clone()# torch.from_numpy().cuda().float()
        edited_latent_codes = latent_W_1.clone()
        edited_latent_codes[:, :8, :] += self.alpha * self.mapper(latent_W_1.clone()).detach()# .numpy()
    
        mixed_wps=edited_latent_codes
        mixed_wps[:,style_range,:]*=1-mix_ratio
        mixed_wps[:,style_range,:]+=latent_W_1[:,style_range,:]*mix_ratio
    
        return mixed_wps