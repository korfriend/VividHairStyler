import cv2
from HairMapper.styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from tqdm import tqdm
from HairMapper.classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from HairMapper.mapper.networks.level_mapper import LevelMapper
import torch
import glob
from HairMapper.diffuse.inverter_remove_hair import InverterRemoveHair
import numpy as np
import os

class HairRemover():
    def __init__(
        self, 
        args, 
        model_name='stylegan2_ada',
        latent_space_type='wp'
    ) -> None:
        self.args=args
        print(f'Initializing generator.')
        self.model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=args.truncation_psi)
        self.mapper = LevelMapper(input_dim=512).eval().cuda()
        ckpt = torch.load('./HairMapper/mapper/checkpoints/final/best_model.pt')
        self.alpha = float(ckpt['alpha']) * 1.2
        self.mapper.load_state_dict(ckpt['state_dict'], strict=True)
        self.kwargs = {'latent_space_type': latent_space_type}
        self.parsingNet = get_parsingNet(save_pth='./HairMapper/ckpts/face_parsing.pth')
        self.inverter = InverterRemoveHair(
            model_name,
            Generator=self.model,
            learning_rate=0.01,
            reconstruction_loss_weight=1.0,
            perceptual_loss_weight=5e-5,
            truncation_psi=args.truncation_psi,
            logger=None
        )
    
    def remove(
        self, 
        latent: np.ndarray, # (1, 18, 512)
        latent_is_ndarray: bool =False
    ):
        if latent_is_ndarray:
            mapper_input = latent.copy()
            mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
            edited_latent_codes = latent.copy()
        else:
            mapper_input_tensor = latent.detach().clone().cuda().float()
            edited_latent_codes = latent.detach().clone().cpu().numpy()
        edited_latent_codes[:, :8, :] += self.alpha * self.mapper(mapper_input_tensor).to('cpu').detach().numpy()

        outputs = self.model.easy_style_mixing(latent_codes=edited_latent_codes,
                                            style_range=range(7, 18),
                                            style_codes=latent,
                                            mix_ratio=0.8,
                                            **self.kwargs
                                            )
        # print(outputs.keys()) # mixed_wps, stylespace_latent
        # print(outputs['mixed_wps'].shape)
        # print(outputs['stylespace_latent'][0].shape)
        edited_img = outputs['image'][0]#[:, :, ::-1]
        return edited_img, outputs['mixed_wps']
    
    def get_mask(self, image_path):
        if self.args.remain_ear:
            hair_mask = get_hair_mask(img_path=image_path, net=self.parsingNet, include_hat=True, include_ear=False)
        else:
            hair_mask = get_hair_mask(img_path=image_path, net=self.parsingNet, include_hat=True, include_ear=True)
        mask_dilate = cv2.dilate(hair_mask,
                                    kernel=np.ones((self.args.dilate_kernel_size, self.args.dilate_kernel_size), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(self.args.blur_kernel_size, self.args.blur_kernel_size))
        mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)
        face_mask = 255 - mask_dilate_blur
        return hair_mask, face_mask


    def blend(
        self, 
        image_path, 
        edited_image, 
        mask_path=None, 
        input_is_array: bool=False,
        return_mask: bool=False
    ):

        if not input_is_array:
            image = cv2.imread(image_path)
        else:
            image = image_path.copy()
        
        if mask_path is not None:
            hair_mask = cv2.imread(mask_path) # get_hair_mask의 결과가 rgb/bgr 이다.
        else:
            if self.args.remain_ear:
                hair_mask = get_hair_mask(img_path=image_path, net=self.parsingNet, include_hat=True, include_ear=False)
            else:
                hair_mask = get_hair_mask(img_path=image_path, net=self.parsingNet, include_hat=True, include_ear=True)

        mask_dilate = cv2.dilate(hair_mask,
                                    kernel=np.ones((self.args.dilate_kernel_size, self.args.dilate_kernel_size), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(self.args.blur_kernel_size, self.args.blur_kernel_size))
        mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)
        face_mask = 255 - mask_dilate_blur

        index = np.where(face_mask > 0)
        cy = (np.min(index[0]) + np.max(index[0])) // 2
        cx = (np.min(index[1]) + np.max(index[1])) // 2
        center = (cx, cy)
        mixed_clone = cv2.seamlessClone(image, edited_image, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)
        if return_mask:
            return mixed_clone, face_mask
        return mixed_clone

