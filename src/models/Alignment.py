import sys
import torch
from torch import nn
import numpy as np
import scipy
import os
from functools import partial
from tqdm import tqdm
import PIL
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from typing import Any, Tuple
from src.utils.slic_utils import slic_custom

from typing import Any, Optional, Tuple
import face_alignment

# from Barbershop.models.stylegan2.model import Generator
from src.utils.data_utils import convert_npy_code
from .face_parsing.model import seg_mean, seg_std
from src.losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
import cv2
from src.utils.data_utils import load_FS_latent, load_latent_W, load_image
from src.utils.model_utils import download_weight


# from src.embedding import Embedding

toPIL = torchvision.transforms.ToPILImage()

class Alignment():
    def __init__(self, opts, embedding=None):
        self.opts = opts
        self.device = opts.device
        self.save_dir = opts.save_dir
        if embedding:
            self.generator = embedding.net.generator
            self.latent_avg = embedding.net.latent_avg
            self.seg = embedding.seg
            self.downsample = embedding.downsample_512
            self.downsample_256 = embedding.downsample
            self.image_transform = embedding.image_transform
        else:
            raise("Please use Embedding instance")
            
        self.setup_align_loss_builder()

        ### Style-Your-Hair
        if self.opts.kp_loss:
            if self.opts.kp_type == '2D':
                self.kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=opts.device)
            else:
                self.kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=opts.device)
            for param in self.kp_extractor.face_alignment_net.parameters():
                param.requires_grad = False
            self.l2 = torch.nn.MSELoss()
        # self.image_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def get_seg(self, image, target=10):
        im = self.preprocess_img(image)
        down_seg, _, _ = self.seg(im)
        seg_target = torch.argmax(down_seg, dim=1).squeeze()
        if target is None:
            return seg_target
        # 1, 512, 512
        return seg_target == target

    def load_generator(self):
        # self.generator = self.embedding.generator
        checkpoint = torch.load(self.opts.ckpt)
        self.generator.load_state_dict(checkpoint['g_ema'])
        self.latent_avg = checkpoint['latent_avg'].to(self.device)

    def load_segmentation_network(self):
        self.seg = self.embedding.seg
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):
        self.downsample = self.embedding.downsample_512
        self.downsample_256 = self.embedding.downsample_256

    def setup_align_loss_builder(self):
        self.loss_builder = AlignLossBuilder(self.opts)

    def preprocess_img(self, img_path):
        im = load_image(img_path)
        im = torchvision.transforms.ToTensor()(im)[:3].unsqueeze(0).to(self.device)
        im = (self.downsample(im).clamp(0, 1) - seg_mean) / seg_std
        return im

    def setup_align_optimizer(self, latent_path=None,ex_mode=False, latent_in = None):
        if latent_path is None:
            latent_W = latent_in.to(self.device).requires_grad_(True)
        else:
            if not ex_mode:
                latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path, allow_pickle=True))).to(self.device).requires_grad_(True)
    
            # latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(self.device).requires_grad_(True)
            else:
                latent_W = load_latent_W(latent_path, device=self.device, allow_pickle=True).requires_grad_(True)

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)

        return optimizer_align, latent_W

    def dilate_erosion_mask_tensor(self, mask, dilate_erosion=5):
        hair_mask = mask.clone()
        hair_mask = hair_mask.numpy()
        hair_mask_dilate = scipy.ndimage.binary_dilation(hair_mask, iterations=dilate_erosion, border_value=0)
        hair_mask_erode = scipy.ndimage.binary_erosion(hair_mask, iterations=dilate_erosion, border_value=0)

        hair_mask_dilate = np.expand_dims(hair_mask_dilate, axis=0)
        hair_mask_erode = np.expand_dims(hair_mask_erode, axis=0)

        return torch.from_numpy(hair_mask_dilate).float(), torch.from_numpy(hair_mask_erode).float()

    def create_down_seg(self, latent_in, is_downsampled=True):
        gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2

        # get hair mask of synthesized image
        im = (self.downsample(gen_im_0_1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(im)

        if is_downsampled == False:
            down_seg = F.interpolate(down_seg, size=(self.opts.size,self.opts.size))
            
        return down_seg, gen_im

    def cuda_unsqueeze(self, li_variables=None, device='cuda'):

        if li_variables is None:
            return None

        cuda_variables = []

        for var in li_variables:
            if not var is None:
                var = var.to(device).unsqueeze(0)
            cuda_variables.append(var)

        return cuda_variables

    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = self.cuda_unsqueeze(self.dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def save_align_results(self, im_name_1, im_name_2, sign, gen_im, latent_in, latent_F, save_dir, save_intermediate=True):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        # save_dir = os.path.join(self.opts.output_dir, 'Align_{}'.format(sign))
        # os.makedirs(save_dir, exist_ok=True)

        W_latent_path = os.path.join(save_dir, '{}_{}'.format(im_name_1, im_name_2))
        FS_latent_path = os.path.join(save_dir, '{}_{}.npz'.format(im_name_1, im_name_2))
        if save_intermediate:
            image_path = os.path.join(save_dir, '{}_{}.png'.format(im_name_1, im_name_2))
            save_im.save(image_path)

        np.save(W_latent_path, latent_in.detach().cpu().numpy())
        np.savez(FS_latent_path , latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())

    def create_target_segmentation_mask_with_bald(
            self, 
            img_path1, 
            img_path2,
            latent_W_bald, 
            is_downsampled=True, 
            user_sketch = False,
            user_mask = None,
            pbar=None
            ) :
        im1 = self.preprocess_img(img_path1)
        down_seg, _, _ = self.seg(im1)
        seg_target1 = torch.argmax(down_seg, dim=1).long()

        im2 = self.preprocess_img(img_path2)
        down_seg2, _, _ = self.seg(im2)
        seg_target2 = torch.argmax(down_seg2, dim=1).long()
        seg_target2_temp = seg_target2.clone()

        G_bald, _ = self.generator([latent_W_bald], input_is_latent=True, return_latents=False,
                                        start_layer=0, end_layer=8)
        G_bald_0_1 = (G_bald + 1) / 2
        im = (self.downsample(G_bald_0_1) - seg_mean) / seg_std
        bald_seg, _, _ = self.seg(im)
        bald_target1 = torch.argmax(bald_seg, dim=1).long()
        bald_target1 = bald_target1[0].byte()
        self.save_vis_mask(img_path1, img_path2, bald_target1.cpu().squeeze(), self.save_dir, count='bald_seg')

        hair_mask1 = torch.where(seg_target1 == 10, torch.ones_like(seg_target1), torch.zeros_like(seg_target1))  # 10 : hair
        seg_target1 = seg_target1[0].byte()
        seg_target1 = torch.where(seg_target1 == 12, torch.zeros_like(seg_target1), seg_target1)  # hair 부분 제외한 나머지 segmap      
        seg_target1 = torch.where(seg_target1 == 10, torch.zeros_like(seg_target1), seg_target1)  # hair 부분 제외한 나머지 segmap
        if self.opts.optimize_warped_trg_mask:
            im1_for_kp = F.interpolate(im1, size=(256, 256))
            im1_for_kp = ((im1_for_kp + 1) / 2).clamp(0, 1)  # [0, 1] 사이로
            src_kp_hm = self.kp_extractor.face_alignment_net(im1_for_kp)
            im2, warped_latent_2, _ = self.warp_target(img_path2, src_kp_hm, img_path1, None)  # Warping !!
            save_im = toPIL(((im2 + 1) / 2).clamp(0, 1).squeeze().cpu())
            save_im.save(os.path.join(self.opts.save_dir, '5_Aligned_src_img.png'))
            warped_down_seg, im2 = self.create_down_seg(warped_latent_2, is_downsampled=is_downsampled)
            if is_downsampled == False:
                warped_seg = F.interpolate(warped_down_seg, size=(self.opts.size, self.opts.size))
                seg_target2 = torch.argmax(warped_seg, dim=1).long()  # todo : debug for k,  512 or 256
            else:
                seg_target2 = torch.argmax(warped_down_seg, dim=1).long()
            warped_down_seg = torch.argmax(warped_down_seg.clone().detach(), dim=1).long()  # 512, 512

        hair_mask2 = torch.where(seg_target2 == 10, torch.ones_like(seg_target2), torch.zeros_like(seg_target2))
        seg_target2 = seg_target2[0].byte()
        
        ##########
        from src.utils.seg_utils import vis_seg
        Image.fromarray(self.tensor_to_numpy(im2)).save("./_temp2/gen_im.png")
        np.save("./_temp2/latent.npy", warped_latent_2.detach().cpu().numpy())
        Image.fromarray(
            vis_seg(seg_target2.detach().cpu().squeeze().numpy())
        ).save("_temp2/seg.png")
        if user_sketch:
            sketch_result = Image.open("./_temp2/r_result.png")
            sketch_latent = torch.from_numpy(np.load("./_temp2/r_latent.npy")).to(self.device)
            sketch_down_seg, gen_im = self.create_down_seg(sketch_latent)
            sketch_down_seg = torch.argmax(sketch_down_seg, dim=1)
            sketch_mask = (sketch_down_seg == 10).unsqueeze(0).float()
            sketch_mask = F.interpolate(sketch_mask, (32, 32))
            warped_latent_2 = sketch_latent.detach().clone()

            seg_target2 = sketch_down_seg.detach().clone().squeeze().byte()
            # seg_target2 = sketch_down_seg.detach.clone()
        if user_mask is not None:

            seg_target2, _ = self.create_down_seg(warped_latent_2)
            seg_target2 = torch.argmax(seg_target2, dim=1)

            # 마스크 전처리. np -> tensor
            if isinstance(user_mask, np.ndarray):
                user_mask = torch.from_numpy(user_mask).to(self.device).unsqueeze(0)
            target_mask = torch.where(user_mask, 10, seg_target2)


            im = target_mask.detach().squeeze().cpu().numpy()
            optimizer_align, latent_align = self.setup_align_optimizer(warped_latent_2, ex_mode=True)
            latent_end = latent_align[:, 6:, :].clone().detach()
            for step in range(61):
                optimizer_align.zero_grad()
                latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
                down_seg, gen_im = self.create_down_seg(latent_in)

                loss_dict = {}

                # ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
                ce_loss = self.loss_builder.weight_cross_entropy_loss(down_seg, target_mask)


                loss_dict["ce_loss"] = ce_loss.item()
                loss = ce_loss


                loss.backward()
                optimizer_align.step()
                # if step % 10 == 0:
                #     st.image(
                #         cv2.resize(self.tensor_to_numpy(gen_im), (256, 256)), 
                #         caption=f"step: {step}"
                #     )
            latent_in = torch.cat([latent_align[:, :6, :], latent_end], dim=1)
            down_seg, gen_im = self.create_down_seg(latent_in)
            seg_target2 = torch.argmax(down_seg, dim=1).long()
            warped_down_seg = torch.argmax(down_seg.clone().detach(), dim=1).long()  # 512, 512
            warped_latent_2 = latent_in.detach().clone()
        
        ##########
        new_target = torch.where(seg_target2 == 10, 10 * torch.ones_like(seg_target1), seg_target1) # put target hair on the target seg 1 (Here, seg_target1 has no hair region)

        self.save_vis_mask(img_path1, img_path2, seg_target1.cpu(), self.save_dir, count='0_erased_src_seg')
        self.save_vis_mask(img_path1, img_path2, new_target.cpu(), self.save_dir, count='0_initial_target_seg')

        if self.opts.mean_seg:
            if self.opts.warped_seg:  # mean_seg is the warped target img's seg
                mean_seg = warped_down_seg.squeeze().type(torch.ByteTensor)  # 512, 512 or 256, 256
                self.save_vis_mask(img_path1, img_path2, mean_seg.cpu(),self.save_dir,count='1_warped_target_seg')
            
            bald_target1 = torch.where(bald_target1 == 10, torch.zeros_like(bald_target1), bald_target1)  # hair 부분 제외한 나머지 segmap
        
            M_hole = (1 - (1-hair_mask1) - hair_mask2).clamp(min=0)
            save_img = toPIL(((M_hole + 1) / 2).clamp(0, 1).squeeze().cpu())
            save_img.save(os.path.join(self.save_dir, "M_hole.png"))
            
            
            bald_target1_face = torch.where((bald_target1 >= 1) & (bald_target1 <= 6), bald_target1, torch.zeros_like(bald_target1)) * 1.0
            masked_bald_down_seg = bald_target1_face* M_hole
            new_target = torch.where((new_target == 0) & M_hole.bool(), masked_bald_down_seg, new_target)
            self.save_vis_mask(img_path1, img_path2, new_target.cpu().squeeze(), self.save_dir, count='1st_target_seg')

            if torch.any(seg_target2 == 14) :
                new_target = torch.where((new_target == 0) & (seg_target2 == 14) & M_hole.bool(), seg_target2, new_target)
                self.save_vis_mask(img_path1, img_path2, new_target.cpu().squeeze(), self.save_dir, count='2nd_target_seg')

            if torch.any(bald_target1 == 14) :
                new_target = torch.where((new_target == 0) & (bald_target1 == 14) & M_hole.bool(), bald_target1, new_target)
                self.save_vis_mask(img_path1, img_path2, new_target.cpu().squeeze(), self.save_dir, count='3rd_target_seg')
            
            if torch.any(seg_target2 == 15) :
                new_target = torch.where((new_target == 0) & (seg_target2 == 15) & M_hole.bool(), seg_target2, new_target)
                self.save_vis_mask(img_path1, img_path2, new_target.cpu().squeeze(), self.save_dir, count='4th_target_seg')

            if torch.any(bald_target1 == 15) :
                new_target = torch.where((new_target == 0) & (bald_target1 == 15) & M_hole.bool(), bald_target1, new_target)
                self.save_vis_mask(img_path1, img_path2, new_target.cpu().squeeze(), self.save_dir, count='5th_target_seg')
            
            inpaint_seg = torch.where(M_hole.bool(), new_target, torch.zeros_like(new_target))
            self.save_vis_mask(img_path1, img_path2, inpaint_seg.cpu().squeeze(), self.save_dir, count='inpaint_target_seg')
            
            new_target_mean_seg = torch.where((new_target == 0) * (seg_target2.to(self.opts.device) != 0), seg_target2.to(self.opts.device), new_target)  # 220213 edited by taeu
            # self.save_vis_mask(img_path1, img_path2, new_target_mean_seg.cpu(), self.save_dir,count='1_warped_target+source_seg')
            
            # if latent_sketch is not None : 
            #     G_sketch, _ = self.generator([latent_sketch], input_is_latent=True, return_latents=False,
            #                             start_layer=0, end_layer=8)
            #     G_sketch_0_1 = (G_sketch + 1) / 2
            #     im = (self.downsample(G_sketch_0_1) - seg_mean) / seg_std
            #     sketch_seg, _, _ = self.seg(im)
            #     sketch_target1 = torch.argmax(sketch_seg, dim=1).long()
            #     sketch_target1 = sketch_target1[0].byte()
            #     sketch_hair_mask = torch.where(sketch_target1==10, torch.ones_like(sketch_target1),torch.zeros_like(sketch_target1))
            #     new_hair_mask = binary_sketch_mask * sketch_hair_mask


            target_mask = new_target_mean_seg.unsqueeze(0).long().to(self.opts.device)
        else:
            target_mask = new_target.unsqueeze(0).long().to(self.opts.device)

        self.save_vis_mask(img_path1, img_path2, target_mask.squeeze().cpu(),self.save_dir, count='2_final_target_seg')

        #####################  Save Visualization of Target Segmentation Mask
        hair_mask_target = torch.where(target_mask == 10, torch.ones_like(target_mask), torch.zeros_like(target_mask))
        if is_downsampled:
            hair_mask_target = F.interpolate(hair_mask_target.float(), size=(512, 512), mode='nearest')
        else:
            hair_mask_target = F.interpolate(hair_mask_target.float().unsqueeze(0), size=(self.opts.size, self.opts.size), mode='nearest')

        # if generated_mask is not None:
        #     generated_mask_tensor = generated_mask.unsqueeze(0).unsqueeze(0).float().to(self.opts.device)
        #     generated_mask_resized = F.interpolate(generated_mask_tensor, size=target_mask.shape[-2:], mode='nearest').squeeze(0)
        #     target_mask = torch.where(generated_mask_resized == 1, 10 * torch.ones_like(target_mask), target_mask)

        if self.opts.optimize_warped_trg_mask:
            hair_mask2 = torch.where(seg_target2_temp == 10, torch.ones_like(seg_target2_temp), torch.zeros_like(seg_target2_temp))
            return target_mask, seg_target2, hair_mask1, inpaint_seg, bald_target1, warped_latent_2
        else:
            return target_mask, seg_target2, hair_mask1, inpaint_seg, bald_target1, None
 
    def M2H_test(
            self, 
            generated_mask, 
            warped_latent_2,
            img_path1, 
            img_path2, 
            FS_dir, 
            W_dir, 
            baldFS_dir, 
            bald_dir, 
            save_dir, 
            all_inpainting = True, 
            init_align = False, 
            sign='realistic', 
            smooth=5, 
            user_sketch=False, 
            user_mask=None,
            ):

        device = self.device
        self.opts.output_dir = save_dir
        self.save_dir = save_dir

        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]

        latent_baldW_path_1 = os.path.join(bald_dir, f'{im_name_1}.npy')
        latent_bald = load_latent_W(latent_baldW_path_1, device)

        if generated_mask is not None:
            target_mask = generated_mask.unsqueeze(0).unsqueeze(0).long()
            print("step2 : ", target_mask.shape)
            
            im1 = self.preprocess_img(img_path1)
            down_seg, _, _ = self.seg(im1)
            seg_target1 = torch.argmax(down_seg, dim=1).long()

            hair_mask1 = torch.where(seg_target1 == 10, torch.ones_like(seg_target1), torch.zeros_like(seg_target1))  # 10 : hair

            latent_FS_path_1 = os.path.join(FS_dir, f'{im_name_1}.npz')

            latent_W_path_1 = os.path.join(W_dir, f'{im_name_1}.npy')
            latent_W_path_2 = os.path.join(W_dir, f'{im_name_2}.npy')
            seg_target2 = None
            inpaint_seg = None
            bald_target1 = None

        else :
            target_mask, seg_target2, hair_mask1, inpaint_seg, bald_target1, warped_latent_2 = self.create_target_segmentation_mask_with_bald(
            img_path1=img_path1, 
            img_path2=img_path2,
            latent_W_bald=latent_bald, 
            user_mask=user_mask,
            user_sketch=user_sketch,
            )

            W_latent_path = os.path.join(save_dir, 'warped_latent_2')
            np.save(W_latent_path, warped_latent_2.detach().cpu().numpy())

            latent_FS_path_1 = os.path.join(FS_dir, f'{im_name_1}.npz')
            latent_W_path_1 = os.path.join(W_dir, f'{im_name_1}.npy')
            latent_FS_path_2 = os.path.join(FS_dir, f'{im_name_2}.npz')
            latent_W_path_2 = os.path.join(W_dir, f'{im_name_2}.npy')

        latent_1, latent_F_1 = load_FS_latent(latent_FS_path_1, device)

        #####
        with torch.no_grad(): 
            M_hair = (target_mask == 10) * 1.0
            # M_hair, _ = self.dilate_erosion(M_hair, device, dilate_erosion=smooth)
            M_hair_down_32 = F.interpolate(M_hair.float(), size=(32, 32), mode='area')

            M_src = 1 - hair_mask1.unsqueeze(0)
            M_src_down_32 = F.interpolate(M_src.float(), size=(32, 32), mode='area')

            M_hole = (1 - M_src - M_hair.unsqueeze(0)).clamp(min=0)
            M_hole_down_32 = F.interpolate(M_hole.squeeze(0).float(), size=(32, 32), mode='area')[0]

            M_union = 1 - (1 - hair_mask1.unsqueeze(0)) * (1 - M_hair.unsqueeze(0))
            M_union, _ = self.dilate_erosion(M_union.squeeze(0), device, dilate_erosion=smooth)
            M_union_down_32 = F.interpolate(M_union.float(), size=(32, 32), mode='area')[0]
            M_keep = 1 - M_union_down_32
        
            height, width = M_hair.squeeze().shape
            M_hair_rgb = torch.zeros((3, height, width)).cpu()
            mask = M_hair.squeeze() != 0
            M_hair_rgb[:, mask] = torch.tensor([255.0, 0.0, 144.0]).view(3, 1)

            F_align, _ = self.generator([warped_latent_2], input_is_latent=True, return_latents=False,
                                                start_layer=0, end_layer=3)
            
            F_src = latent_F_1.clone()


        if init_align :
            optimizer_align, latent_align_1 = self.setup_align_optimizer(latent_W_path_2)
        else :
            optimizer_align, latent_align_1 = self.setup_align_optimizer(latent_W_path_1)

        pbar = tqdm(range(self.opts.align_steps1), desc='Align img1 to seg_mask', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()

            if all_inpainting :
                latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
                down_seg, gen_im = self.create_down_seg(latent_in)

            else :
                F_fill, _ = self.generator([latent_align_1], input_is_latent=True, return_latents=False,
                                    start_layer=0, end_layer=3)
                
                latent_F_mixed = F_fill + M_hair_down_32 * (F_align - F_fill)

                gen_im, _ = self.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                                    end_layer=8, layer_in=latent_F_mixed)

                G_0_1 = (gen_im + 1) / 2

                im = (self.downsample(G_0_1) - seg_mean) / seg_std
                down_seg, _, _ = self.seg(im)

            loss_dict = {}

            ##### Cross Entropy Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask.squeeze(0))
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            loss.backward(retain_graph=True)
            optimizer_align.step()
            # if step % 10 == 0 or step == self.opts.align_steps1 - 1:
                
            #     target_dir = f"_temp_w_bald"
            #     I = toPIL(((gen_im + 1) / 2).clamp(0, 1).squeeze().cpu())
            #     I.save(f"{target_dir}/{str(step).zfill(3)}.png")

        seg_target1 = torch.argmax(down_seg, dim=1).long()
        seg_target1 = seg_target1[0].byte().cpu().detach()

        if generated_mask is not None :
            self.save_vis_mask(img_path1, img_path2, seg_target1.cpu(), self.save_dir, count='down_seg_w/_target_mask')
        
        else : 
            self.save_vis_mask(img_path1, img_path2, seg_target1.cpu(), self.save_dir, count='down_seg')
            self.save_vis_mask(img_path1, img_path2, target_mask.squeeze().cpu(), self.save_dir, count='target_mask')
            
        save_im = toPIL(((gen_im + 1) / 2).clamp(0, 1).squeeze().cpu())
        save_im.save(os.path.join(self.opts.save_dir, '4_Aligned_src_img.png'))
        aligned_latent_1, _ = self.generator([latent_align_1], input_is_latent=True, return_latents=False,
                                                start_layer=0, end_layer=3)
        F_new = aligned_latent_1.clone().detach()

        ##############################################    

        with torch.no_grad():
            warped_gen_im, _ = self.generator([warped_latent_2], input_is_latent=True, return_latents=False,
                                                    start_layer=0, end_layer=8)
            warped_gen_im_256 = self.downsample_256(warped_gen_im)
            warped_gen_im = toPIL(((warped_gen_im + 1) / 2).clamp(0, 1).squeeze().cpu())
            warped_gen_im.save(os.path.join(self.opts.save_dir, 'warped_gen_im.png'))


            height, width = hair_mask1.squeeze(0).shape
            hair_mask1_rgb = torch.zeros((3, height, width)).cpu()
            hair_mask1_rgb1 = torch.zeros((3, height, width)).cpu()
            mask = hair_mask1.squeeze(0) == 0
            mask1 = hair_mask1.squeeze(0) != 0
            hair_mask1_rgb[:, mask] = torch.tensor([128.0, 53.0, 14.0]).view(3, 1)
            hair_mask1_rgb1[:, mask] = torch.tensor([128.0, 53.0, 14.0]).view(3, 1)
            hair_mask1_rgb1[:, mask1] = torch.tensor([0.0, 183.0, 235.0]).view(3, 1)


        latent_F_mixed = F_new + M_hair_down_32 * (F_align - F_new)
        latent_F_mixed = latent_F_mixed + M_keep.unsqueeze(0) * (F_src - latent_F_mixed)

        gen_im, _ = self.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                                end_layer=8, layer_in=latent_F_mixed)

        self.save_align_results(im_name_1, im_name_2, sign, gen_im, latent_1, latent_F_mixed, save_dir,
                                save_intermediate=True)

        return gen_im, latent_F_mixed, seg_target1, M_hair_rgb, hair_mask1_rgb1, hair_mask1_rgb,bald_target1, target_mask.squeeze().cpu(), warped_latent_2, seg_target2, inpaint_seg, bald_target1

    def Hair_Line_editing(self, generated_mask, bald_W, latent_FS_path_1, latent_W_path_1, smooth=5 ) :
        device = self.device

        latent_1, latent_F_1 = load_FS_latent(latent_FS_path_1, device)

        M_hair = (generated_mask == 10) * 1.0
        M_hair, _ = self.dilate_erosion(M_hair.unsqueeze(0).unsqueeze(0), device, dilate_erosion=smooth)
        M_hair_down_32 = F.interpolate(M_hair.float(), size=(32, 32), mode='area')
        target_mask = generated_mask.unsqueeze(0).long()
        
        optimizer_align, latent_align_1 = self.setup_align_optimizer(latent_W_path_1)

        with torch.no_grad():
            tmp_latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
            down_seg_tmp, I_Structure_Style_changed = self.create_down_seg(tmp_latent_in)

            current_mask_tmp = torch.argmax(down_seg_tmp, dim=1).long()
            HM_Structure = torch.where(current_mask_tmp == 10, torch.ones_like(current_mask_tmp),
                                       torch.zeros_like(current_mask_tmp))
            HM_Structure = F.interpolate(HM_Structure.float().unsqueeze(0), size=(256, 256), mode='nearest')

        pbar = tqdm(range(self.opts.align_steps2), desc='Align Step 2', leave=False)
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
            down_seg, gen_im = self.create_down_seg(latent_in)

            Current_Mask = torch.argmax(down_seg, dim=1).long()
            HM_G_512 = torch.where(Current_Mask == 10, torch.ones_like(Current_Mask),
                                   torch.zeros_like(Current_Mask)).float().unsqueeze(0)
            HM_G = F.interpolate(HM_G_512, size=(256, 256), mode='nearest')

            loss_dict = {}

            ########## Segmentation Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss

            #### Style Loss
            H1_region = self.downsample_256(I_Structure_Style_changed) * HM_Structure
            H2_region = self.downsample_256(gen_im) * HM_G
            style_loss = self.loss_builder.style_loss(H1_region, H2_region, mask1=HM_Structure, mask2=HM_G)

            loss_dict["style_loss"] = style_loss.item()
            loss += style_loss

            loss.backward()
            optimizer_align.step()

        seg_target1 = torch.argmax(down_seg, dim=1).long()
        seg_target1 = seg_target1[0].byte().cpu().detach()

        latent_F_out_new, _ = self.generator([latent_in], input_is_latent=True, return_latents=False,
                                                 start_layer=0, end_layer=3)
        latent_F_out_new = latent_F_out_new.clone().detach()
        latent_F_mixed = bald_W + M_hair_down_32* (latent_F_out_new - bald_W)
        gen_im, _ = self.generator([latent_1], input_is_latent=True, return_latents=False, start_layer=4,
                            end_layer=8, layer_in=latent_F_mixed)
        

        return gen_im, latent_F_mixed, seg_target1, M_hair_down_32


    """
    0521
    Style-Your-Hair 의 Alignment 이식 부분입니다.
    
    """

    def warp_target(
            self, 
            img_path2: str, 
            src_kp_hm: Any, 
            img_path1: str,
            generated_mask: Optional[torch.Tensor] = None
        )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        img_path1(A, src)이미지의 얼굴을 img_path2(B, ref) 의 얼굴로 정렬하는 함수입니다.

        Args:
            img_path2 (str): 목표로 하는 얼굴(B, ref)의 파일 경로입니다.
            src_kp_hm (kp): 초기 이미지의 얼굴(A, src) 키포인트입니다. None을 입력하면 내부에서 자동으로 생성합니다.
            img_path1 (str): 초기 이미지의 얼굴(A, src)의 파일 주소입니다.

        Returns:
            gen_im: StyleGAN의 결과 이미지입니다.(tensor)
            latent_in: warp된 latent vector입니다. (tensor)
            warped_down_seg: 최종 결과의 segment입니다. (tensor, (1, 512, 512))

        """
        im_name_1 =  os.path.splitext(os.path.basename(img_path1))[0]
        output_dir = self.opts.output_dir
        embedding_dir = self.opts.embedding_dir
        is_downsampled = self.opts.size > 256
        device = self.opts.device
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]  # target image : hair

        if generated_mask is not None:
            # latent_FS_path_2 = os.path.join(output_dir, f'{im_name_2}.npz')
            latent_W_path_2 = os.path.join(output_dir, f'{im_name_2}.npy')
        else:
            # latent_FS_path_2 = os.path.join(embedding_dir, 'FS', f'{im_name_2}.npz')
            latent_W_path_2 = os.path.join(embedding_dir, 'W+', f'{im_name_2}.npy')
        
        # latent_2, latent_F_2 = load_FS_latent(latent_FS_path_2, device)  # [1,18,512], [1, 512, 32, 32]


        # todo : change 40 to self.opts.warp_steps
        optimizer_warp_w, latent_warped_2 = self.setup_align_optimizer(latent_W_path_2)
        pbar = tqdm(range(self.opts.warp_steps), desc='Warp Target Step 1', leave=False)
        latent_W_optimized = latent_warped_2
        latent_F_optimized = None
        mode = 'w+_total'
        if self.opts.warp_front_part:
            mode = 'w+_6'

        cur_check_dir = None
        # cur_check_dir = f'{self.opts.output_dir}warped_result_{mode}_{self.opts.kp_type}/'
        # if self.opts.warp_loss_with_prev_list is not None:
        #     cur_check_dir += f'{self.opts.warp_loss_with_prev_list}/'
        # os.makedirs(cur_check_dir, exist_ok=True)

        if src_kp_hm is None:
            im1 = cv2.imread(img_path1)
            im1 = self.image_transform(im1).unsqueeze(0).to(self.device)
            im1_for_kp = F.interpolate(im1, size=(256, 256))
            im1_for_kp = ((im1_for_kp + 1) / 2).clamp(0, 1) # [0, 1] 사이로
            src_kp_hm = self.kp_extractor.face_alignment_net(im1_for_kp)
        warped_down_seg = None
        latent_in, warped_down_seg = self.optimize_warping(pbar, optimizer_warp_w, latent_W_optimized, latent_F_optimized, mode, is_downsampled, src_kp_hm, im_name_1, im_name_2, cur_check_dir, img_path1, img_path2)
        latent_F = None

        ## save img
        gen_im = self.save_warp_result(latent_F, latent_in, is_downsampled, cur_check_dir,im_name_2, im_name_1)
        return gen_im, latent_in, warped_down_seg

    def save_warp_result(self, latent_F, latent_in, is_downsampled, cur_check_dir, im_name_2, im_name_1):
        if latent_F is not None:
            gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False,
                                           start_layer=4,
                                           end_layer=8, layer_in=latent_F)
        else:
            _, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
        # if cur_check_dir is not None:
        #     save_im = toPIL(((gen_im + 1) / 2).clamp(0, 1).squeeze().cpu())
        #     save_im.save(cur_check_dir + f'{im_name_2}_with_{im_name_1}_pose.png')
        return gen_im
    
    def optimize_warping(self, pbar, optimizer_warp, latent_W_optimized, latent_F_optimized, mode, is_downsampled,
                         src_kp_hm,
                         im_name_1, im_name_2, cur_check_dir, img_path1, img_path2):

        if 'w+_6' == mode:
            latent_end = latent_W_optimized[:, 6:, :].clone().detach()

        # for style_loss
        ref_im = Image.open(img_path2).convert('RGB')
        ref_im256 = ref_im.resize((256, 256), PIL.Image.LANCZOS)
        ref_im256 = self.image_transform(ref_im256).unsqueeze(0).to(self.opts.device)

        self.seg_transform = transforms.Compose([transforms.Resize((512, 512)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        ref_im512 = self.seg_transform(ref_im).unsqueeze(0).to(self.opts.device)
        down_seg_ref, _, _ = self.seg(ref_im512)  # 512 512
        ref_seg = torch.argmax(down_seg_ref.clone().detach(), dim=1).long()
        seg_hair_ref = torch.where((ref_seg == 10), torch.ones_like(ref_seg),
                                   torch.zeros_like(ref_seg))
        seg_hair_ref256 = F.interpolate(seg_hair_ref.unsqueeze(0).float(), size=(256, 256))

        prev_im = ref_im256
        prev_seg = ref_seg

        if 'delta_w' in self.opts.warp_loss_with_prev_list:
            latent_W_optimized_prev = latent_W_optimized[:, :6, :].clone().detach() # todo : changed, front 만 동작하게 되어있음

        if 'style_hair_slic_large' in self.opts.warp_loss_with_prev_list:
            self.slic_compactness = 20  # 100
            self.slic_numSegments = 5 # todo : opts
            lambda_hair = 1000 #  todo : opts

            # cur_check_dir += f'{self.slic_compactness}_{self.slic_numSegments}_{lambda_hair}/'
            # os.makedirs(cur_check_dir, exist_ok=True)

            ref_im256_slic = (((ref_im256[0] + 1) / 2).clamp(0, 1)).permute(1, 2, 0).detach().cpu().numpy()
            seg_hair_ref256_slic = seg_hair_ref256[0].detach().cpu().numpy()
            prev_slic_segments, prev_centroids, _ = slic_custom(ref_im256_slic, mask=seg_hair_ref256_slic,
                                                                compactness=self.slic_compactness,
                                                                n_segments=self.slic_numSegments, sigma=5)

        for step in pbar:
            optimizer_warp.zero_grad()
            if 'w+_total' == mode:
                latent_in = latent_W_optimized  # torch.cat([latent_warped_2[:, :6, :], latent_2[:, 6:, :]], dim=1) ## 220205
                # latent_in = torch.cat([latent_warped_2[:, :6, :], latent_2[:, 6:, :]], dim=1) # 220205
                down_seg, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
            elif 'w+_6' == mode:
                latent_in = torch.cat([latent_W_optimized[:, :6, :], latent_end], dim=1)
                down_seg, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
            else:
                # todo : implement cat latent vector some part fixed, the other part to be optimized
                pass

            loss_dict = {}
            loss = 0

            # 220303 added
            gen_im1024 = gen_im.clone()
            gen_im1024 = ((gen_im1024 + 1) / 2).clamp(0, 1)

            if self.opts.size > 256:
                gen_im = F.interpolate(gen_im, size=(256, 256))
            gen_im = ((gen_im + 1) / 2).clamp(0, 1)
            gen_kp_hm = self.kp_extractor.face_alignment_net(gen_im) # 1,68,64,64

            # keypoint loss
            kp_loss = self.l2(src_kp_hm[:, :], gen_kp_hm[:, :])  # no restriction
            lambda_kp = 1000 # todo opts
            loss_dict["kp_loss"] = kp_loss.item() * lambda_kp
            loss += kp_loss * lambda_kp

            # early stop : if Keypoint loss is below 0.1
            if kp_loss * lambda_kp < 0.05:
                print(f"Early stop, Key point loss below 0.05 : {kp_loss:.3f}")
                break

            curr_seg = torch.argmax(down_seg.clone().detach(), dim=1).long()

            if self.opts.warp_loss_with_prev_list is not None:
                # 220303 added
                try:
                    if 'style_hair_slic_large' in self.opts.warp_loss_with_prev_list:

                        seg_hair_gen = torch.where((curr_seg == 10), torch.ones_like(curr_seg),
                                                   torch.zeros_like(curr_seg))
                        seg_hair_gen256 = F.interpolate(seg_hair_gen.unsqueeze(0).float(), size=(256, 256))

                        gen_im256_slic = gen_im[0].permute(1, 2, 0).detach().cpu().numpy()
                        seg_hair_gen256_slic = seg_hair_gen256[0].detach().cpu().numpy()

                        if step == 0:
                            prev_centroids_ref = prev_centroids.copy()

                        sp_gen_mask_large256, sp_gen_im, prev_slic_segments, prev_centroids, closest_indices \
                            = self.get_sp_mask(gen_im256_slic, seg_hair_gen256_slic, prev_centroids=prev_centroids,
                                               im_path=None, im1024=gen_im1024)

                        if step == 0:
                            sp_ref_mask_large256, sp_ref_im, ref_slic_segments, ref_centroids, _ \
                                = self.get_sp_mask(ref_im256_slic, seg_hair_ref256_slic, prev_centroids=prev_centroids_ref,
                                                   im_path=img_path2)
                            points = prev_centroids[0].copy()  # n, 2
                            points_prev = ref_centroids[0].copy()  # 6, 2
                            points_repeat = np.repeat(np.array(points)[:, np.newaxis], ref_centroids.shape[0],
                                                      axis=1)  # 3, 1, 2 -> 3, 6, 2
                            closest_indices = np.argmin(np.linalg.norm(points_repeat - points_prev[np.newaxis,], axis=2),
                                                        axis=1)  # 1 6 2

                        sp_ref_im, sp_ref_mask_large256 = sp_ref_im[closest_indices], sp_ref_mask_large256[closest_indices]
                        hair_loss = self.loss_builder.style_loss(sp_gen_im, sp_ref_im, mask1=sp_gen_mask_large256,
                                                                 mask2=sp_ref_mask_large256)

                        loss_dict["style_loss_prev_hair_large_slic"] = hair_loss.item() * lambda_hair
                        loss += hair_loss * lambda_hair  # 0.001
                except :
                    pass

                # perceptual loss (lpips)
                # perceptual_loss = self.loss_builder._loss_hair_percept(gen_im, ref_im256, mask=seg_hair_ref256)
                # lambda_perceptual = 100  # 가중치 조정 필요
                # perceptual_loss_scalar = perceptual_loss.item()  # 스칼라 값으로 변환

                # loss_dict["perceptual_loss"] = perceptual_loss_scalar * lambda_perceptual
                # loss += perceptual_loss_scalar * lambda_perceptual

                if 'delta_w' in self.opts.warp_loss_with_prev_list:  # 1-hair 의 교집합
                    delta_w_loss = self.l2(latent_W_optimized[:, :6, :], latent_W_optimized_prev)
                    lambda_delta_w = 1000
                    loss_dict["delta_w"] = delta_w_loss.item() * lambda_delta_w
                    loss += delta_w_loss * lambda_delta_w

                if 'style_hair' in self.opts.warp_loss_with_prev_list:
                    seg_hair_gen = torch.where((curr_seg == 10), torch.ones_like(curr_seg),
                                               torch.zeros_like(curr_seg))
                    seg_hair_gen256 = F.interpolate(seg_hair_gen.unsqueeze(0).float(), size=(256, 256))

                    hair_loss = self.loss_builder.style_loss(gen_im, ref_im256, mask1=seg_hair_gen256,
                                                             mask2=seg_hair_ref256)

                    lambda_hair = 100
                    loss += hair_loss / lambda_hair
                    loss_dict["hair_loss"] = hair_loss.item() / lambda_hair



            latent_W_optimized_prev = latent_W_optimized[:, :6, :].clone().detach() # todo :이것도 6 기준 opt 로

            loss.backward()
            optimizer_warp.step()
            # if step % 10 == 0 : ### warped result save step size
            #     cur_check_dir = f'{self.opts.output_dir}check_hair/'
            #     os.makedirs(cur_check_dir, exist_ok=True)
            #     print(f'{step}: ', loss_dict)
            #     save_im = toPIL(gen_im.squeeze().cpu())
            #     aaa = torch.zeros((3, 256, 256))
            #     kp_prob =  F.interpolate(torch.max(gen_kp_hm, dim=1)[0].unsqueeze(0).cpu(), size=(256, 256))
            #     aaa[0] = kp_prob[0][0]
            #     kp_im = toPIL(torch.cat((gen_im.squeeze().cpu(), aaa), dim=-1))
            #
            #     save_im.save(cur_check_dir + f'{im_name_2}_with_{im_name_1}_pose_{step}.png')
            #     # added for debug
            #     if 'style_hair_slic_large' in self.opts.warp_loss_with_prev_list:
            #         save_image(torch.cat([sp_gen_im * sp_gen_mask_large256, sp_ref_im * sp_ref_mask_large256]),
            #                    cur_check_dir + f'{im_name_2}_with_{im_name_1}_sp_gen_ref_{step}.png', normalize=True,
            #                    nrow=sp_gen_im.shape[0])

            prev_im = gen_im.clone().detach()
            prev_seg = torch.argmax(down_seg.clone().detach(), dim=1).long()

        # if self.opts.save_all:
            # save_im = toPIL(gen_im.squeeze().cpu())
            # save_im.save(os.path.join(self.opts.save_dir, '1_warped_img.png'))
        if 'F' in mode:
            return latent_F_optimized, latent_W_optimized
        if self.opts.warped_seg:
            return latent_in, prev_seg
        else:
            return latent_in, None
        

    def vis_seg(self, pred):
        num_labels = 16
        
        color = np.array([[0, 0, 0],  ## 0
                        [102, 204, 255],  ## 1
                        [255, 204, 255],  ## 2
                        [255, 255, 153],  ## 3
                        [255, 255, 153],  ## 4
                        [255, 255, 102],  ## 5
                        [51, 255, 51],  ## 6
                        [0, 153, 255],  ## 7
                        [0, 255, 255],  ## 8
                        [0, 255, 255],  ## 9
                        [204, 102, 255],  ## 10
                        [0, 153, 255],  ## 11
                        [0, 255, 153],  ## 12
                        [0, 51, 0],
                        [102, 153, 255],  ## 14
                        [255, 153, 102],  ## 15
                        ])
        h, w = np.shape(pred)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        #     print(color.shape)
        for ii in range(num_labels):
            #         print(ii)
            mask = pred == ii
            rgb[mask, None] = color[ii, :]
        # Correct unk
        unk = pred == 255
        rgb[unk, None] = color[0, :]
        return rgb

    def save_vis_mask(self, img_path1, img_path2, mask, save_dir, count = 0):
        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
        vis_path = os.path.join(save_dir,f'{count}.png')
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().squeeze().cpu().numpy()
        vis_mask = self.vis_seg(mask)
        PIL.Image.fromarray(vis_mask).save(vis_path)
    def tensor_to_pil(self, gen_im):
        return toPIL(((gen_im.squeeze() + 1) / 2).detach().cpu().clamp(0, 1))
    def tensor_to_numpy(self, gen_im):
        return np.array(self.tensor_to_pil(gen_im))