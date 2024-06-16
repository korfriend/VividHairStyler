import torch
from torch import nn
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import torchvision
import scipy
from .face_parsing.model import BiSeNet, seg_mean, seg_std
from .optimizer.ClampOptimizer import ClampOptimizer
from src.losses.blend_loss import BlendLossBuilder
from src.utils.bicubic import BicubicDownSample
from src.models.stylegan2.model import Generator
import torch.nn.functional as F
import cv2
from src.utils.data_utils import load_FS_latent, get_mask_dict, load_image
from src.utils.model_utils import download_weight

# from src.embedding import Embedding

toPIL = torchvision.transforms.ToPILImage()




class Blending():
    def __init__(self, opts, embedding = None):
        super(Blending, self).__init__()
        self.opts = opts
        self.device = opts.device
        if embedding:
            self.image_transform = embedding.image_transform_256
            self.seg = embedding.seg
            self.generator = embedding.net.generator
            self.downsample = embedding.downsample_512
            self.downsample_256 = embedding.downsample
        # self.load_segmentation_network()
        # self.load_downsampling()
        self.setup_blend_loss_builder()
        # self.load_generator()


    def cuda_unsqueeze(self, li_variables=None, device='cuda'):

        if li_variables is None:
            return None

        cuda_variables = []

        for var in li_variables:
            if not var is None:
                var = var.to(device).unsqueeze(0)
            cuda_variables.append(var)

        return cuda_variables

    def load_generator(self):
        self.generator = Generator(self.opts.size, self.opts.latent, self.opts.n_mlp, channel_multiplier=self.opts.channel_multiplier).to(self.device)
        checkpoint = torch.load(self.opts.ckpt)
        self.generator.load_state_dict(checkpoint['g_ema'])
        self.latent_avg = checkpoint['latent_avg'].to(self.device)

    def load_downsampling(self):
        self.downsample = self.embedding.downsample_512
        self.downsample_256 = self.embedding.downsample_256


    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()


    def setup_blend_optimizer(self):

        interpolation_latent = torch.zeros((18, 512), requires_grad=True, device=self.opts.device)

        opt_blend = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=self.opts.learning_rate)

        return opt_blend, interpolation_latent

    def setup_blend_loss_builder(self):
        self.loss_builder = BlendLossBuilder(self.opts)

    def dilate_erosion_mask_tensor(self, mask, dilate_erosion=5):
        hair_mask = mask.clone()
        hair_mask = hair_mask.numpy()
        hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion, border_value=0)
        hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion, border_value=0)

        hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
        hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

        return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()

    def blend_images(self, images, Ws, Fs, F7_blend_1_2, HM_1_2, m1_tensor_1024, m3_tensor_1024, pbar=None):

        device = self.opts.device
        F7_blend_1_2 = F7_blend_1_2.detach().clone()

        # Normalize 필수
        I_1 = self.image_transform(Image.fromarray(images[0])).to(device)
        I_3 = self.image_transform(Image.fromarray(images[2])).to(device)

        # mask
        HM_1D, _ = self.dilate_erosion(m1_tensor_1024, device)
        HM_3D, HM_3E = self.dilate_erosion(m3_tensor_1024, device)

        target_hairmask = (HM_1_2 == 10) * 1.0
        target_hairmask = target_hairmask.float().unsqueeze(0)
        HM_1_2D, HM_1_2E = self.dilate_erosion(target_hairmask, device)

        downsampled_hair_mask = F.interpolate(HM_1_2E, size=(256, 256), mode='bilinear', align_corners=False)
        upsampled_hair_mask = F.interpolate(HM_1_2E, size=(1024, 1024), mode='bilinear', align_corners=False)

        interpolation_latent = torch.zeros((18, 512), requires_grad=True, device=device)
        # opt_blend = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=ii2s.opts.learning_rate)
        opt_blend = torch.optim.Adam([interpolation_latent], lr=self.opts.learning_rate)
        with torch.no_grad():
            I_X, _ = self.generator([Ws[0]], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
            I_X_0_1 = (I_X + 1) / 2
            IM = (self.downsample(I_X_0_1) - seg_mean) / seg_std
            down_seg, _, _ = self.seg(IM)
            current_mask = torch.argmax(down_seg, dim=1).long().cpu().float()
            HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
            HM_X = F.interpolate(HM_X.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()
            HM_XD, _ = self.cuda_unsqueeze(self.dilate_erosion_mask_tensor(HM_X), device)
            target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        I_G, _ = self.generator([Ws[2]], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
        for step in range(300):
            opt_blend.zero_grad()
            latent_mixed = Ws[0] + interpolation_latent.unsqueeze(0) * (Ws[2] - Ws[0])
            
            I_G, _ = self.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=Fs[2])
            I_G_1_2, _ = self.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
            
            # G_lab_ori = rgb_to_lab(I_G)
            # G_lab = ii2s.image_transform(rgb_to_lab(I_G_1_2))

            im_dict = {
                'gen_im': self.downsample_256(I_G),
                'im_1': I_1,
                'im_3': I_3,
                'mask_face': target_mask,
                'mask_hair': HM_3E,
                'mask_2_hair': downsampled_hair_mask,
            }

            total_loss = 0
            face_loss = self.loss_builder._loss_face_percept(im_dict['gen_im'], im_dict['im_1'], im_dict['mask_face'])
            hair_loss = self.loss_builder._loss_hair_percept(im_dict['gen_im'], im_dict['im_3'], im_dict['mask_hair'])
            # hair_lab_loss = blend.loss_builder._loss_hair_percept(blend.downsample_256(G_lab_ori), I_3_lab, im_dict['mask_hair'])

            H1_region = self.downsample_256(I_G_1_2) * im_dict['mask_2_hair']
            H2_region = im_dict['im_3'] * im_dict['mask_hair']
            style_loss = self.loss_builder.style_loss(H2_region, H1_region, im_dict['mask_hair'], im_dict['mask_2_hair'])
            
            # H1_region_lab = blend.downsample_256(G_lab) * im_dict['mask_2_hair']
            # H2_region_lab = I_3_lab * im_dict['mask_hair']
            # style_lab_loss = blend.loss_builder.style_loss(H2_region_lab, H1_region_lab, im_dict['mask_hair'], im_dict['mask_2_hair'])
            
            total_loss += face_loss+ hair_loss + 10000*style_loss
            opt_blend.zero_grad()
            total_loss.backward(retain_graph=True)
            opt_blend.step()
            if pbar is not None:
                pbar.progress(int(step / self.opts.W_steps * 100), text=f'Blending... ({step} / {self.opts.W_steps})')
        if pbar is not None:
            pbar.empty()

        I_G_blend1, _ = self.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
        return I_G_blend1

    def save_blend_results(self, im_name_1, im_name_2, im_name_3, sign,  gen_im, latent_in, latent_F):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Blend_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}_{}_{}.npz'.format(im_name_1, im_name_2, im_name_3))
        image_path = os.path.join(save_dir, '{}_{}_{}.png'.format(im_name_1, im_name_2, im_name_3))
        output_image_path = os.path.join(self.opts.output_dir, '{}_{}_{}_{}.png'.format(im_name_1, im_name_2, im_name_3, sign))

        save_im.save(image_path)
        save_im.save(output_image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())


    def dilate_erosion_mask_path(im_path, seg_net, dilate_erosion=5):
        # # Mask
        # mask = Image.open(mask_path).convert("RGB")
        # mask = mask.resize((256, 256), PIL.Image.NEAREST)
        # mask = transforms.ToTensor()(mask)  # [0, 1]

        IM1 = (BicubicDownSample(factor=2)(torchvision.transforms.ToTensor()(Image.open(im_path))[:3].unsqueeze(0).cuda()).clamp(
            0, 1) - seg_mean) / seg_std
        down_seg1, _, _ = seg_net(IM1)
        mask = torch.argmax(down_seg1, dim=1).long().cpu().float()
        mask = torch.where(mask == 10, torch.ones_like(mask), torch.zeros_like(mask))
        mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()

        # Hair mask + Hair image
        hair_mask = mask
        hair_mask = hair_mask.numpy()
        hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion)
        hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion)

        hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
        hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

        return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()

    # def dilate_erosion_mask_tensor(mask, dilate_erosion=5):
    #     hair_mask = mask.clone()
    #     hair_mask = hair_mask.numpy()
    #     hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion)
    #     hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion)

    #     hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
    #     hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

    #     return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()

    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
            free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
            free_mask_D, free_mask_E = self.cuda_unsqueeze(self.dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
            return free_mask_D, free_mask_E