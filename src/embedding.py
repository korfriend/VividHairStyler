import sys
sys.path.append('..')
sys.path.append('./Barbershop')
sys.path.append('../Barbershop')

import numpy as np
import torch
import os 
import cv2
from functools import partial
# from Barbershop.models.Net import Net
from Barbershop.models.stylegan2.model import Generator
from Barbershop.utils.bicubic import BicubicDownSample
from Barbershop.models.optimizer.ClampOptimizer import ClampOptimizer
from Barbershop.models.face_parsing.model import BiSeNet, seg_mean, seg_std
import torchvision.transforms as transforms
import tqdm
import torch.nn as nn
from Barbershop.losses import lpips, masked_lpips
from Barbershop.losses.style.style_loss import StyleLoss
from PIL import Image
import skimage.metrics as sm

class Embedding():
    toPIL = transforms.ToPILImage()
    opt_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'sgdm': partial(torch.optim.SGD, momentum=0.9),
        'adamax': torch.optim.Adamax
    }
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    def __init__(self, opts) -> None:
        self.opts = opts
        self.device = opts.device
        
        # self.net = Net()
        self.load_generator()
        self.load_segmentmodel()
        self.load_PCA_model()
        self.cal_layer_num()
        self.load_loss_fn()
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)
        self.downsample_512 = BicubicDownSample(factor=self.opts.size // 512)

    def load_generator(self):
        self.generator = Generator(self.opts.size, self.opts.latent, self.opts.n_mlp, channel_multiplier=self.opts.channel_multiplier).to(self.device)
        checkpoint = torch.load(self.opts.ckpt)
        self.generator.load_state_dict(checkpoint['g_ema'])
        self.latent_avg = checkpoint['latent_avg'].to(self.device)

    def load_segmentmodel(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        assert os.path.exists(self.opts.seg_ckpt), "seg ckpt 확인."

        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def preprocess_img(self, img_path):
        if isinstance(img_path, str):
            im = transforms.ToTensor()(Image.open(img_path))[:3].unsqueeze(0).to(self.opts.device)
        elif isinstance(img_path, np.ndarray):
            im = transforms.ToTensor()(Image.fromarray(img_path))[:3].unsqueeze(0).to(self.opts.device)
        elif isinstance(img_path, torch.Tensor):
            im = img_path
        else:
            raise("세그먼트 이미지 확인")
        im = (self.downsample_512(im).clamp(0, 1) - seg_mean) / seg_std
        return im

    def get_seg(self, image, target=10):
        im = self.preprocess_img(image)
        down_seg, _, _ = self.seg(im)
        if target is None:
            return down_seg
        seg_target = torch.argmax(down_seg, dim=1).squeeze()
        # 1, 512, 512
        return seg_target == target

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
            'l2': self.l2_loss_fn, 
            'percept': self.percept_loss_fn, 
            'mask_percept': self.mask_percep_loss_fn, 
            'style': self.style_loss_fn, 
            'psnr': self.psnr, 
            'ssim': self.ssim, 
        }

    def load_loss_fn(self):
        if not hasattr(self, '_percept_loss_fn'):
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
                'l2': self.l2_loss_fn, 
                'percept': self.percept_loss_fn, 
                'mask_percept': self.mask_percep_loss_fn, 
                'style': self.style_loss_fn, 
                'psnr': self.psnr, 
                'ssim': self.ssim, 
            }
        
    def l2_loss_fn(self, im1, im2, mask1, mask2=None):
        if mask1 is None:
            return self._l2_loss_fn(im1, im2) * self.opts.l2_lambda
        return self._l2_loss_fn(im1*mask1, im2*mask1) * self.opts.l2_lambda

    def percept_loss_fn(self, im1, im2, mask1, mask2):
        return self._percept_loss_fn(im1, im2).sum() * self.opts.percept_lambda
    
    def mask_percep_loss_fn(self, im1, im2, mask1, mask2):
        return self._mask_percep_loss_fn(im1, im2, mask1).sum() * self.opts.percept_lambda

    def style_loss_fn(self, im1, im2, mask1, mask2):
        return self._style_loss_fn(im1*mask1, im2*mask2, mask1, mask2) * self.opts.style_lambda

    def psnr(self, im1, im2, mask1, mask2=None):
        return sm.peak_signal_noise_ratio(self.tensor_to_numpy(im1*mask1), self.tensor_to_numpy(im2*mask1))
    
    def ssim(self, im1, im2, mask1, mask2=None):
        # im1 = ((im1*mask1).detach().cpu().numpy() * 255).astype(np.uint8)
        return sm.structural_similarity(self.tensor_to_numpy(im1*mask1), self.tensor_to_numpy(im2*mask1), win_size=3)

    def load_PCA_model(self):
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

        assert os.path.isfile(PCA_path), "PCA 모델 없습니다. 바버샵 한번 돌려주세요"

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)


    def p_norm_loss_fn(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss
    
    def cal_layer_num(self):
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14

        self.S_index = self.layer_num - 11

        return

    def load_latent(self, input, latent_F=None, latent_space='W+'):
        if isinstance(input, str):
            if latent_space == 'W+':
                input = torch.from_numpy(np.load(input)).to(self.device)
            elif latent_space == 'FS':
                saved_dict = np.load(input)
                input = torch.from_numpy(saved_dict['latent_F']).to(self.device)
                if latent_F is not None:
                    latent_F = torch.from_numpy(saved_dict['latent_in']).to(self.device).requires_grad_(True)
        elif isinstance(input, np.ndarray):
            if input.ndim == 3:
                assert "input should be latent variable. ex shape) (1, 18, 512)"
            else:
                input = torch.from_numpy(input).to(self.device)
        elif isinstance(input, torch.Tensor):
            input = input.clone().detach()
        if input is not None and input.shape == (18, 512):
            input = input.unsqueeze(0)
        # if input is None:
        #     self.net.latent_avg.clone().detach().to(self.devi)
        return input, latent_F
    
    def load_image(self, ref_im):
        if isinstance(ref_im, str):
            assert os.path.exists(ref_im)
            ref_im = cv2.imread(ref_im)
            ref_im = cv2.cvtColor(cv2.COLOR_BGR2RGB)        
        if ref_im is None:
            return None, None
        ref_im_H = self.image_transform(cv2.resize(ref_im, (1024, 1024))).unsqueeze(0).to(self.device)
        ref_im_L = self.image_transform(cv2.resize(ref_im,  (256,  256))).unsqueeze(0).to(self.device)
        return ref_im_H, ref_im_L
    
    def tensor_to_numpy(self, gen_im):
        return np.array(self.toPIL(((gen_im.squeeze() + 1) / 2).detach().cpu().clamp(0, 1)))

    def invert(
            self, 
            ref_im: np.ndarray, 
            input = None, 
            latent_F: torch.Tensor = None, 
            # input_type: str = 'path', 
            latent_space: str = 'W+', 
            epoch: int = 1100, 
            log_interval: int = 50, 
            train_layers = None, 
            mask = None, 
            res_im = None, 
            F_layer: int = 3, 
        ) -> None:
        
        # assert input in [None, 'path', 'numpy', 'tensor']
        assert latent_space in ['W+', 'FS']

        input, latent_F = self.load_latent(input, latent_F=latent_F, latent_space=latent_space)
        
        ref_im_H, ref_im_L = self.load_image(ref_im)
        res_im_H, res_im_L = self.load_image(res_im)
    
        if mask is not None:
            mask[mask>0] = 1
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask_256 = nn.functional.interpolate(mask, size=(256, 256), mode='nearest')
            
        # Optimizer
        latent = []
        for i in range(self.layer_num):
            if input is None:
                tmp = self.latent_avg.clone().detach().to(self.device)
            else:
                tmp = input[0, i, :].clone().detach()
            
            if (latent_space == 'W+') or (i > self.S_index):
                tmp.requires_grad = True
            
            if train_layers and i in train_layers:
                tmp.requires_grad = True
            # else:
            #     tmp.requires_grad = False
            latent.append(tmp)

        if latent_space == 'W+':
            optimizer = self.opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)
        elif latent_space == 'FS':
            if latent_F is None:
                latent_F, _ = self.generator([input], input_is_latent=True, return_latents=False, start_layer=0, end_layer=F_layer)
            latent_F = latent_F.detach().clone().requires_grad_(True)
            optimizer = self.opt_dict[self.opts.opt_name](latent[self.S_index:] + [latent_F], lr=self.opts.learning_rate)

        output_list = []
        for step in tqdm.tqdm(range(epoch), desc=f"{latent_space}", leave=False):
            optimizer.zero_grad()

            latent_in = torch.stack(latent).unsqueeze(0)
            if latent_space == 'W+':
                gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False)
            elif latent_space == 'FS':
                gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False,
                                               start_layer=F_layer+1, end_layer=8, layer_in=latent_F)

            if mask is None:
                l2_loss = self.l2_loss_fn(ref_im_H, gen_im, mask, mask)
                percept_loss = self.percept_loss_fn(ref_im_L, self.downsample_256(gen_im), mask, mask)
            else:
                if res_im is None:
                    l2_loss = self.l2_loss_fn(ref_im_H, gen_im, mask, mask)
                    percept_loss = self.mask_percep_loss_fn(self.downsample_256(gen_im), ref_im_L, mask_256, mask_256).sum()
                else:
                    l2_loss = self.l2_loss_fn(ref_im_H, gen_im, mask, mask)
                    # l2_loss2 = self.l2_loss_fn(res_im_H*(1-mask), gen_im*(1-mask))
                    # l2_loss = l2_loss1 + l2_loss2

                    # percept_loss1 = self.mask_percep_loss_fn(ref_im_L*mask_256, self.downsample_256(gen_im)*mask_256).sum()
                    percept_loss = self.mask_percep_loss_fn(self.downsample_256(gen_im), res_im_L, 1-mask_256, 1-mask_256).sum()
                    # percept_loss = percept_loss1 + percept_loss2

            l2_loss *= self.opts.l2_lambda
            percept_loss *= self.opts.percept_lambda

            p_norm_loss = self.p_norm_loss_fn(latent_in)*self.opts.p_norm_lambda
            loss = l2_loss + percept_loss + p_norm_loss
            
            if step % log_interval == 0 or step == epoch -1:
                image_rgb = self.tensor_to_numpy(gen_im)
                output_list.append({
                    'epoch': step, 
                    'gen_im': gen_im.clone().detach(), 
                    'image': image_rgb.copy(), 
                    'latent_in': latent_in, 
                    'latent_F': latent_F, 
                    'l2_loss': l2_loss.item(), 
                    'percept_loss': percept_loss.item(), 
                    'p_norm_loss': p_norm_loss.item(), 
                    'loss': loss.item(), 
                })

            loss.backward()
            optimizer.step()
        return output_list
    
    def color(
            self, 
            mask_list, 
            color_list, 
            input = None, 
            latent_F: torch.Tensor = None, 
            epoch: int = 1100, 
            log_interval: int = 50, 
            train_layers = range(0, 18), 
            F_layer: int = 3, 
        ) -> None:
        input, latent_F = self.load_latent(input, latent_F=latent_F, latent_space="FS")
        
        if mask is not None:
            mask[mask>0] = 1
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask_256 = nn.functional.interpolate(mask, size=(256, 256), mode='nearest')
            
        # Optimizer
        latent = []
        for i in range(self.layer_num):
            if input is None:
                tmp = self.latent_avg.clone().detach().to(self.device)
            else:
                tmp = input[0, i, :].clone().detach()

            if i in train_layers:
                tmp.requires_grad = True
            else:
                tmp.requires_grad = False
            latent.append(tmp)

        if latent_F is None:
            latent_F, _ = self.generator([input], input_is_latent=True, return_latents=False, start_layer=0, end_layer=F_layer)
        latent_F = latent_F.detach().clone().requires_grad_(True)
        optimizer = self.opt_dict[self.opts.opt_name](latent[self.S_index:] + [latent_F], lr=self.opts.learning_rate)

        output_list = []
        for step in tqdm.tqdm(range(epoch), desc=f"color", leave=False):
            optimizer.zero_grad()

            latent_in = torch.stack(latent).unsqueeze(0)
            gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False,
                                            start_layer=F_layer+1, end_layer=8, layer_in=latent_F)

            
            # l2_loss = self.l2_loss_fn(ref_im_H, gen_im, mask, mask)
            # percept_loss = self.mask_percep_loss_fn(self.downsample_256(gen_im), ref_im_L, mask_256, mask_256).sum()

            l2_loss *= self.opts.l2_lambda
            percept_loss *= self.opts.percept_lambda

            p_norm_loss = self.p_norm_loss_fn(latent_in)*self.opts.p_norm_lambda
            loss = l2_loss + percept_loss + p_norm_loss
            
            if step % log_interval == 0 or step == epoch -1:
                image_rgb = self.tensor_to_numpy(gen_im)
                output_list.append({
                    'epoch': step, 
                    'gen_im': gen_im.clone().detach(), 
                    'image': image_rgb.copy(), 
                    'latent_in': latent_in, 
                    'latent_F': latent_F, 
                    'l2_loss': l2_loss.item(), 
                    'percept_loss': percept_loss.item(), 
                    'p_norm_loss': p_norm_loss.item(), 
                    'loss': loss.item(), 
                })

            loss.backward()
            optimizer.step()
        return output_list

    def blend(
            self, 
            ref_im: np.ndarray, 
            res_im, 
            init_latent, 
            ref_latent, 
            res_latent, 
            latent_F: torch.Tensor, 
            mask, 
            epoch: int = 200, 
            log_interval: int = 50, 
            start_layer: int = 4, 
            hair_loss_list = ['mask_percept'], 
            face_loss_list = ['l2'], 
            train_layers = range(18), 
            latent_F_trainable = False,
        ) -> None:
        init_latent = init_latent.clone().detach()
        ref_latent, latent_F = self.load_latent(ref_latent, latent_F=latent_F, latent_space="W+")
        res_latent, _ = self.load_latent(res_latent, latent_F=None, latent_space="W+")

        ref_im_H, ref_im_L = self.load_image(ref_im)
        res_im_H, res_im_L = self.load_image(res_im)
    
        
        if mask is not None:
            mask[mask>0] = 1
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)#.clone().detach()
            elif len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
            mask_256 = nn.functional.interpolate(mask, size=(256, 256), mode='nearest')
         
        # Optimizer
        # interpolation_latent1 = torch.zeros((3, 18, 512), requires_grad=True, device=self.opts.device)
        
        # 각 부분의 가중치를 계산
        # weights = torch.tensor([1.0, 1.0, 1.0], device=self.opts.device, requires_grad=True)
        # optimizer = ClampOptimizer(torch.optim.Adam, [interpolation_latent1], lr=self.opts.learning_rate)

        # weight_gradients = {'init_latent': [], 'ref_latent': [], 'res_latent': []}

        #### 기존 바버샵
        # interpolation_latent = torch.zeros((18, 512), requires_grad=True, device=self.opts.device)
        # optimizer = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=self.opts.learning_rate)
            
        # Optimizer
        latent = []
        for i in range(self.layer_num):
            tmp = init_latent[0, i, :].clone().detach()

            if i in train_layers:
                tmp.requires_grad = True
            else:
                tmp.requires_grad = False
            latent.append(tmp)

        if(latent_F_trainable is True):
            optimizer = self.opt_dict[self.opts.opt_name](latent+[latent_F], lr=self.opts.learning_rate)
        else :  
            optimizer = self.opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)


        output_list = []
        for step in tqdm.tqdm(range(epoch), desc=f"Blend", leave=False):
            optimizer.zero_grad()

            #### 기본
            # latent_mixed = init_latent.detach().clone()

            #### # 가중치 합 1의 제약이 없을 때
            # sub_1 = interpolation_latent1[0].unsqueeze(0)
            # sub_2 = interpolation_latent1[1].unsqueeze(0)
            # sub_3 = interpolation_latent1[2].unsqueeze(0)
            # latent_mixed = sub_1*init_latent + sub_3*ref_latent + sub_2*res_latent

            #### 바버샵 블렌딩 
            # latent_mixed = res_latent + interpolation_latent.unsqueeze(0) * (ref_latent - res_latent)
            # gen_im, _ = self.generator([latent_mixed], input_is_latent=True, return_latents=False,
            #                                    start_layer=start_layer, end_layer=8, layer_in=latent_F)

            #### 가중치 합 1의 제약이 있을 떄?
            # Softmax를 적용하여 가중치의 합이 1이 되도록 정규화
            # normalized_weights = torch.softmax(weights, dim=0)
            # latent_weights = torch.softmax(interpolation_latent1, dim=0)
            # sub_1 = latent_weights[0].unsqueeze(0) * normalized_weights[0]
            # sub_2 = latent_weights[1].unsqueeze(0) * normalized_weights[1]
            # sub_3 = latent_weights[2].unsqueeze(0) * normalized_weights[2]
            # latent_mixed = sub_1*init_latent + sub_2*ref_latent + sub_3*res_latent

            # 가중치 합 1의 제약이 있을 때
            # weights = torch.softmax(interpolation_latent1, dim=0)  # 첫 번째 차원을 기준으로 softmax 적용
            # latent_mixed = weights[0] * init_latent + weights[1] * ref_latent + weights[2] * res_latent
            # print(weights[0])

            latent_mixed = torch.stack(latent).unsqueeze(0)
            gen_im, _ = self.generator([latent_mixed], input_is_latent=True, return_latents=False,
                                               start_layer=start_layer, end_layer=8, layer_in=latent_F)
           
            loss_dict = {
                'l2': torch.tensor([0], device=self.device, dtype=torch.float), 
                'percept': torch.tensor([0], device=self.device, dtype=torch.float), 
                'mask_percept': torch.tensor([0], device=self.device, dtype=torch.float), 
                'style': torch.tensor([0], device=self.device, dtype=torch.float), 
            }
            for loss_name in face_loss_list:
                if loss_name == 'l2':
                    loss = self.loss_dict[loss_name](res_im_H, gen_im , 1 -mask, None).sum()
                elif loss_name == 'percept':
                    loss = self.loss_dict[loss_name](res_im_H*(1 - mask), gen_im*(1 - mask), 1 - mask, None).sum()
                else:
                    loss = self.loss_dict[loss_name](res_im_L, self.downsample_256(gen_im), 1 - mask_256, 1 - mask_256)
                loss_dict[loss_name] += loss
            
            for loss_name in hair_loss_list:
                if loss_name == 'l2':
                    loss = self.loss_dict[loss_name](ref_im_H, gen_im, mask, None).sum()
                elif loss_name == 'percept':
                    loss = self.loss_dict[loss_name](ref_im_H*(mask), gen_im*(mask),  mask,  None).sum()
                else:
                    loss = self.loss_dict[loss_name](ref_im_L, self.downsample_256(gen_im), mask_256, mask_256)
                loss_dict[loss_name] += loss

            loss = loss_dict['l2'] + loss_dict['percept'] + loss_dict['mask_percept'] + loss_dict['style']
            
            if step % log_interval == 0 or step == epoch -1:
                image_rgb = self.tensor_to_numpy(gen_im)
                output_list.append({
                    'epoch': step, 
                    'gen_im': gen_im.clone().detach(), 
                    'image': image_rgb.copy(), 
                    'latent_in': latent_mixed, 
                    'latent_F': latent_F, 
                    'l2_loss': loss_dict['l2'].item(), 
                    'percept_loss': loss_dict['percept'].item() + loss_dict['mask_percept'].item(),  
                    'style': loss_dict['style'].item(), 
                    'loss': loss, 
                })

            loss.backward()

            # optimizer.step(min=0, max=1)
            optimizer.step()

        return output_list

    def eval_images(
        self, 
        target_im, 
        hair_im, 
        face_im, 
        hair_mask=None, 
        face_mask=None, 
        return_mask = False, 
        **kargv,
    ):
        target_H, target_L = self.load_image(target_im)
        face_H, face_L = self.load_image(face_im)
        hair_H, hair_L = self.load_image(hair_im)
        
        if hair_mask is None:
            hair_mask = self.get_seg(target_im).unsqueeze(0).float()
            print(f"hair_mask: {hair_mask.shape}")
            hair_mask_H = nn.functional.interpolate(hair_mask, size=(1024, 1024), mode='nearest')
            hair_mask_L = nn.functional.interpolate(hair_mask, size=(256, 256), mode='nearest')
        if face_mask is None:
            face_mask = 1 - self.get_seg(face_im).unsqueeze(0).float()
            face_mask_H = nn.functional.interpolate(face_mask, size=(1024, 1024), mode='nearest')
            face_mask_L = nn.functional.interpolate(face_mask, size=(256, 256), mode='nearest')
            
        # if isinstance(hair_mask, np.ndarray):
        #     hair_mask = torch.from_numpy(hair_mask).to(self.device)
        # if len(hair_mask.shape) == 2:
        #     hair_mask = hair_mask.unsqueeze(0).unsqueeze(0)#.clone().detach()
        # elif len(hair_mask.shape) == 1:
        #     hair_mask = hair_mask.unsqueeze(0)
        # hair_mask[hair_mask>0] = 1
        # hair_mask_256 = nn.functional.interpolate(hair_mask, size=(256, 256), mode='nearest')
        
        # if face_mask is not None:
        #     if isinstance(face_mask, np.ndarray):
        #         face_mask = torch.from_numpy(face_mask).to(self.device)
        #     if len(face_mask.shape) == 2:
        #         face_mask = face_mask.unsqueeze(0).unsqueeze(0)#.clone().detach()
        #     elif len(face_mask.shape) == 1:
        #         face_mask = face_mask.unsqueeze(0)
        #     face_mask[face_mask>0] = 1
        #     face_mask_256 = nn.functional.interpolate(face_mask, size=(256, 256), mode='nearest')
        
        mask_dict = {
            'hair_H': hair_mask_H, 
            'hair_L': hair_mask_L, 
            'face_H': face_mask_H, 
            'face_L': face_mask_L, 
        }
        ref_dict = {
            'hair_H': hair_H, 
            'hair_L': hair_L, 
            'face_H': face_H, 
            'face_L': face_L, 
        }
        
        score_dict = {}
        for target in ['hair', 'face']:
            mask_ = mask_dict[f"{target}_H"]
            mask_256_ = mask_dict[f"{target}_L"]
            ref_H = ref_dict[f"{target}_H"]
            ref_L = ref_dict[f"{target}_L"]
            for loss_name, score_fn in self.loss_dict.items():
                if loss_name == 'l2':
                    score = score_fn(target_H, ref_H, mask_).sum()
                elif loss_name in ['psnr', 'ssim']:
                    score = score_fn(target_H, ref_H, mask_)
                else:
                    score = score_fn(target_L, ref_L, mask_256_, mask_256_)
                
                if isinstance(score, torch.Tensor):
                    score = score.item()
                score_dict[f"{target}_{loss_name}"] = score
        if return_mask:
            return score_dict, hair_mask.detach().squeeze().cpu().numpy(), face_mask.detach().squeeze().cpu().numpy()
        return score_dict

    def blend_barbershop(
            self, 
            ref_im: np.ndarray, 
            res_im, 
            init_latent, 
            ref_latent, 
            res_latent, 
            latent_F: torch.Tensor, 
            mask, 
            epoch: int = 200, 
            log_interval: int = 50, 
            start_layer: int = 4
        ) -> None:
        init_latent = init_latent.clone().detach()
        ref_latent, latent_F = self.load_latent(ref_latent, latent_F=latent_F, latent_space="W+")
        res_latent, _ = self.load_latent(res_latent, latent_F=None, latent_space="W+")
        
        ref_im_H, ref_im_L = self.load_image(ref_im)
        res_im_H, res_im_L = self.load_image(res_im)
    
        if mask is not None:
            mask[mask>0] = 1
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(self.device)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)#.clone().detach()
            elif len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
            mask_256 = nn.functional.interpolate(mask, size=(256, 256), mode='nearest')
            
        # Optimizer
        interpolation_latent1 = torch.zeros((18, 512), requires_grad=True, device=self.opts.device)
        optimizer = ClampOptimizer(torch.optim.Adam, [interpolation_latent1], lr=self.opts.learning_rate)

        output_list = []
        for step in tqdm.tqdm(range(epoch), desc=f"Blend", leave=False):
            optimizer.zero_grad()
            # sub_1 = interpolation_latent1[:18, :].unsqueeze(0)
            # sub_2 = interpolation_latent1[18:, :].unsqueeze(0)
            latent_mixed = res_latent + interpolation_latent1.unsqueeze(0)*(ref_latent - res_latent)
            gen_im, _ = self.generator([latent_mixed], input_is_latent=True, return_latents=False,
                                               start_layer=start_layer, end_layer=8, layer_in=latent_F)
            
            loss_dict = {
                'l2': torch.tensor([0], device=self.device, dtype=torch.float), 
                'percept': torch.tensor([0], device=self.device, dtype=torch.float), 
                'mask_percept': torch.tensor([0], device=self.device, dtype=torch.float), 
                'style': torch.tensor([0], device=self.device, dtype=torch.float), 
            }

            loss_dict['l2'] += self.loss_dict['l2'](ref_im_H, gen_im, mask,1 -mask).sum()
            loss_dict['mask_percept'] += self.loss_dict['mask_percept'](self.downsample_256(gen_im), res_im_L, 1-mask_256, mask_256)

            loss = loss_dict['l2'] + loss_dict['mask_percept']

            # l2_loss1 = self.l2_loss_fn(ref_im_H, gen_im, mask,mask)
            # l2_loss2 = self.l2_loss_fn(res_im_H*(1-mask), gen_im*(1-mask))
            # l2_loss = l2_loss1# + l2_loss2
            # percept_loss = self.mask_percep_loss_fn(self.downsample_256(gen_im), res_im_L, 1-mask_256,1-mask_256).sum()

            # l2_loss *= self.opts.l2_lambda
            # percept_loss *= self.opts.percept_lambda
            # loss = l2_loss + percept_loss
            
            if step % log_interval == 0 or step == epoch -1:
                image_rgb = self.tensor_to_numpy(gen_im)
                output_list.append({
                    'epoch': step, 
                    'gen_im': gen_im.clone().detach(), 
                    'image': image_rgb.copy(), 
                    'latent_in': latent_mixed, 
                    'latent_F': latent_F, 
                    'l2_loss':  loss_dict['l2'].item(), 
                    'percept_loss': loss_dict['mask_percept'].item(), 
                    # 'p_norm_loss': p_norm_loss.item(), 
                    'loss': loss, 
                })

            loss.backward()
            optimizer.step(min=0, max=1)
        return output_list

    def color_inject(self, hair_im, res_im, sketch, hair_mask, res_mask, color_mask):
        # color_mask = sketch_segment(new_sketch_rgb, sketch_mask, hair_mask)

        """
        내부 옵티파이저 생성함수
        """
        def setup_optimizer(latent_S, latent_F):
            latent_F = latent_F.detach().clone()
            latent_S = latent_S.detach().clone()

            # Optimizer
            S_index = 9
            E_index = 13
            latent = []
            latent_F.requires_grad_(False)
            for i in range(18):
                tmp = latent_S[0, i, :].clone().detach()
                if i in range(S_index, E_index):
                    tmp.requires_grad = True
                else:
                    tmp.requires_grad = False
                latent.append(tmp)
            optimizer = torch.optim.Adam(latent[S_index:E_index], lr=self.opts.learning_rate)
            return latent, latent_F, optimizer

        color_result_list = []
        for color in tqdm.tqdm(color_list, leave=True):
            result_dict = {}
            target_color = image_transform(color.reshape(1, 1, -1)).to(device).reshape(1, -1)
        # latent_F = F7_style.detach().clone()
            # latent_S = rec_latent_W_tensor.detach().clone()
            latents, fixed_F, optimizer = setup_optimizer(rec_latent_W_tensor, F7_style)
            for step in tqdm.tqdm(range(30), leave=False):
                optimizer.zero_grad()
                latent_in = torch.stack(latents).unsqueeze(0)
                gen_im, _ = ii2s.generator([latent_in], input_is_latent=True, return_latents=False,
                                                        start_layer=4, end_layer=8, layer_in=fixed_F)
                loss1 = ii2s.l2_loss_fn(res_H, gen_im, 1-mask_1024, 1-mask_1024).sum()

                sum_color = torch.sum(torch.sum(gen_im*mask_1024, dim=-1), dim=-1)
                num_pixels = torch.sum(torch.sum(mask_1024, dim=-1), dim=-1)
                mean_color = sum_color / num_pixels
                
                loss2 = torch.abs(target_color - mean_color).mean()

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
            color_result_list.append({
                'gen_im': gen_im.detach().clone(), 
                'latent_in': latent_in.detach().clone(),
            })
            # break
        imshow([color.reshape(1, 1, -1) for color in color_list])
        imshow([ii2s.tensor_to_numpy(gen_im['gen_im']) for gen_im in color_result_list])

        return None
    

from scipy.spatial import KDTree


def sketch_segment(sketch, hair_mask, sketch_mask=None):
    if sketch_mask is None:
        sketch_mask = ~np.all(sketch == [0, 0, 0], axis=-1)

    # 색상 마스크 초기화 (검정색으로)
    color_mask = np.zeros_like(sketch)

    # 스케치 픽셀의 위치 찾기
    nonzero_x, nonzero_y = np.nonzero(sketch_mask)
    sketch_pixels = np.stack((nonzero_x, nonzero_y), axis=-1)

    # KDTree를 사용하여 가장 가까운 스케치 픽셀 찾기
    tree = KDTree(sketch_pixels)

    for i in tqdm.trange(sketch_mask.shape[0]):
        for j in range(sketch_mask.shape[1]):
            if hair_mask[i, j] == 0:
                continue  # 마스크가 0이면 검정색을 유지
            _, idx = tree.query([i, j])
            nearest_sketch_pixel = sketch_pixels[idx]

            # 스케치의 색상을 색상 마스크에 적용
            color_mask[i, j] = sketch[nearest_sketch_pixel[0], nearest_sketch_pixel[1]]
    # plt.imshow(color_mask)
    return color_mask

def sketch_sub_mask(sketch, color_mask):
    color_list = []
    sketch_mask_list = []
    unique_colors = np.unique(sketch.reshape(-1, sketch.shape[2]), axis=0)
    for unique_color in tqdm.tqdm(unique_colors):
        if np.sum(unique_color) == 0:
            continue
        sub_mask = np.all(color_mask == unique_color, axis=-1)
        color_list.append(unique_color)
        # 마스크를 uint8 형으로 변환 (0 또는 255)
        sketch_mask_list.append((sub_mask * 255).astype(np.uint8))
        # break
    # imshow(sketch_mask_list)
    return sketch_mask_list
# sketch_mask_list = sketch_sub_mask(new_sketch_rgb, color_mask)


from src.utils.args_utils import parse_yaml
if __name__ == "__main__":
    
    opts = parse_yaml('config.yml')
    ii2s = Embedding(opts)
    image = cv2.imread('test_images/0011_e0u0n_240104_184025/reconstructed.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    latent, gen_im = ii2s.invert(image, input=None, epoch=200)
    # latent, gen_im = ii2s.invert(image, input=latent, epoch=250, latent_space="FS")
    
    result = np.array(ii2s.toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1)))
    cv2.imwrite("__test1.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("__test2.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    

    # for score
    # opts = parse_yaml('configs/config.yml')
    # print(sys.argv)
