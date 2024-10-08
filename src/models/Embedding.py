import os
import torch
import numpy as np
from torch import nn
from functools import partial
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# Importing from local modules
from .Net import Net
from .face_parsing.model import BiSeNet, seg_mean, seg_std
from src.utils.data_utils import load_latent_W
from src.utils.model_utils import download_weight
from src.utils.bicubic import BicubicDownSample
from src.losses.embedding_loss import EmbeddingLossBuilder

toPIL = transforms.ToPILImage()

class Embedding(nn.Module):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image_transform_256 = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __init__(self, opts, net=None):
        super(Embedding, self).__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net
        self.generator = self.net.generator
        self.load_segmentmodel()
        self.load_downsampling()
        self.setup_embedding_loss_builder()
    
    def image_to_tensor(self, im, downsampling=False, device="cuda"):
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        elif isinstance(im, Image.Image):
            pass
        else:
            raise("이미지 타입 문제. 확인 바람.")
    
        if downsampling:
            result = self.image_transform_256(im)
        else:
            result = self.image_transform(im)

        return result.unsqueeze(0).to(device)
        
    def load_segmentmodel(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        # assert os.path.exists(self.opts.seg_ckpt), "seg ckpt 확인."

        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 256)
        self.downsample_512 = BicubicDownSample(factor=self.opts.size // 512)
        
    def setup_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder(self.opts)

    def preprocess_img(self, img_path):
        if isinstance(img_path, str):
            im = transforms.ToTensor()(Image.open(img_path))[:3].unsqueeze(0).to(self.opts.device)
        elif isinstance(img_path, np.ndarray):
            im = transforms.ToTensor()(Image.fromarray(img_path))[:3].unsqueeze(0).to(self.opts.device)
        elif isinstance(img_path, torch.Tensor):
            im = img_path
        elif isinstance(img_path, Image.Image):
            im = transforms.ToTensor()(img_path)[:3].unsqueeze(0).to(self.opts.device)
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

    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.net.cal_l_F(latent_F, F_init)
            loss_dic['l_F'] = l_F
            loss += l_F

        return loss, loss_dic 

    def setup_W_optimizer(self, init_latent=None):
        if init_latent is not None:
            if isinstance(init_latent, np.ndarray):
                init_latent = torch.tensor(init_latent.squeeze(), dtype=torch.float32).cuda()
            else:
                load_latent_W(init_latent)
            
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        latent = []
        for i in range(self.net.layer_num):
            if init_latent is None:
                tmp = self.net.latent_avg.clone().detach().cuda()
                print("제발로")
            else:
                tmp = init_latent[i, :]
            tmp.requires_grad = True
            latent.append(tmp)

        optimizer_W = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)

        return optimizer_W, latent

    def setup_FS_optimizer(self, latent_W, F_init):
        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        for i in range(self.net.layer_num):

            tmp = latent_W[0, i].clone()

            if i < self.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True

            latent_S.append(tmp)

        optimizer_FS = opt_dict[self.opts.opt_name](latent_S[self.net.S_index:] + [latent_F], lr=self.opts.learning_rate)

        return optimizer_FS, latent_F, latent_S

    def invert_image_in_W(self, image, init_latent=None, pbar=None):
        ref_im_H = self.image_transform(image)
        save_im = toPIL(((ref_im_H+ 1) / 2).detach().cpu().clamp(0, 1))
        save_im.save("test_w+.png")
        print("왜저래")
        ref_im_L = self.image_transform_256(image)
        device = self.opts.device
        optimizer_W, latent = self.setup_W_optimizer(init_latent)

        pbar = tqdm(range(self.opts.W_steps), desc='W+ Embedding', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            latent_in = torch.stack(latent).unsqueeze(0)

            gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
            im_dict = {
                'ref_im_H': ref_im_H.to(device),
                'ref_im_L': ref_im_L.to(device),
                'gen_im_H': gen_im,
                'gen_im_L': self.downsample(gen_im)
            }

            loss, _ = self.cal_loss(im_dict, latent_in)
            loss.backward()
            optimizer_W.step()
            pbar.set_postfix(step=step, loss=loss.item())
        print("저장햇냐?")
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_im.save("test_w+.png")

        return gen_im.detach().clone(), latent_in.detach().clone() # Return loss values and intermediate latents

    def invert_image_in_FS(self, image, W_init=None, F_init=None, pbar=None, text = '', max_steps=700):    
        ref_im_H = self.image_transform(image)
        ref_im_L = self.image_transform_256(image)

        device =self.opts.device 

        if W_init is None:
            _, latent_W = self.invert_image_in_W(image, pbar=pbar)
        else:
            latent_W = W_init.clone()
        
        if F_init is None:
            F_init, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

        optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)

        pbar = tqdm(range(max_steps), desc='FS Embedding', leave=False)
        for step in pbar:
            optimizer_FS.zero_grad()
            latent_in = torch.stack(latent_S).unsqueeze(0)
            gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                            start_layer=4, end_layer=8, layer_in=latent_F)
            im_dict = {
                'ref_im_H': ref_im_H.to(device),
                'ref_im_L': ref_im_L.to(device),
                'gen_im_H': gen_im,
                'gen_im_L': self.downsample(gen_im)
            }

            loss, _ = self.cal_loss(im_dict, latent_in)
            loss.backward()
            optimizer_FS.step()
            pbar.set_postfix(step=step, loss=loss.item())

        return gen_im.detach().clone(), latent_in.detach().clone(), latent_F.detach().clone() # Return both loss values

    def save_W_results(self, ref_name, gen_im, latent_in):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        output_dir = os.path.join(self.opts.output_dir, 'W+')
        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npy')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def save_W_intermediate_results(self, ref_name, gen_im, latent_in, step):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        intermediate_folder = os.path.join(self.opts.output_dir, 'W+', ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def save_FS_results(self, ref_name, gen_im, latent_in, latent_F):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        output_dir = os.path.join(self.opts.output_dir, 'FS')
        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f'{ref_name[0]}.npz')
        image_path = os.path.join(output_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(),
                 latent_F=latent_F.detach().cpu().numpy())

    def tensor_to_pil(self, gen_im):
        return toPIL(((gen_im.squeeze() + 1) / 2).detach().cpu().clamp(0, 1))
    
    def tensor_to_numpy(self, gen_im):
        return np.array(self.tensor_to_pil(gen_im))
    