import torch
from torch import nn
from .Net import Net
import numpy as np
import os
from functools import partial
from src.utils.bicubic import BicubicDownSample
from .face_parsing.model import BiSeNet, seg_mean, seg_std
from src.datasets.image_dataset import ImagesDataset
from src.losses.embedding_loss import EmbeddingLossBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
# import torchvision
import torchvision.transforms as transforms
from src.utils.data_utils import convert_npy_code
from src.utils.data_utils import load_image, load_latent_W
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
        # self._load_image_transform()
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

        assert os.path.exists(self.opts.seg_ckpt), "seg ckpt 확인."

        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 256)
        self.downsample_512 = BicubicDownSample(factor=self.opts.size // 512)
        
        

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
        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                if init_latent is None:
                    tmp = self.net.latent_avg.clone().detach().cuda()
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




    def setup_dataloader(self, image_path=None):

        self.dataset = ImagesDataset(opts=self.opts,image_path=image_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print("Number of images: {}".format(len(self.dataset)))

    def setup_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder(self.opts)

    def invert_images_in_W(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')
        for ref_im_H, ref_im_L, ref_name in ibar:
            optimizer_W, latent = self.setup_W_optimizer()
            pbar = tqdm(range(self.opts.W_steps), desc='Embedding', leave=False)
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

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_W.step()

                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))

                if self.opts.save_intermediate and step % self.opts.save_interval== 0:
                    self.save_W_intermediate_results(ref_name, gen_im, latent_in, step)

            self.save_W_results(ref_name, gen_im, latent_in)

    def invert_image_in_W(self, image_path, init_latent=None, pbar=None):
        image = load_image(image_path)
        ref_im_H = self.image_transform(image)
        ref_im_L = self.image_transform_256(image)
        device = self.opts.device
        optimizer_W, latent = self.setup_W_optimizer(init_latent)
        # pbar = tqdm(range(self.opts.W_steps), desc='Embedding', leave=False)
        for step in range(self.opts.W_steps):
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
            if pbar is not None:
                pbar.progress(int(step / self.opts.W_steps * 100), text=f'Embedding to W+ space ({step} / {self.opts.W_steps})')
        if pbar is not None:
            pbar.empty()
        return gen_im.detach().clone(), latent_in.detach().clone()


    def invert_image_in_W_without_path(self, image, init_latent=None, pbar=None):
        ref_im_H = self.image_transform(image)
        ref_im_L = self.image_transform_256(image)
        device = self.opts.device
        optimizer_W, latent = self.setup_W_optimizer(init_latent)
        if pbar is None:
            pbar = tqdm(range(100), desc='Embedding', leave=False)
        else:
            pbar.reset(total=self.opts.W_steps)
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

        pbar.close()
        return gen_im.detach().clone(), latent_in.detach().clone()


    def invert_images_in_FS(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        output_dir = self.opts.output_dir
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')
        for ref_im_H, ref_im_L, ref_name in ibar:

            latent_W_path = os.path.join(output_dir, 'W+', f'{ref_name[0]}.npy')
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(device)
            F_init, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
            optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)


            pbar = tqdm(range(self.opts.FS_steps), desc='Embedding', leave=False)
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

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_FS.step()

                if self.opts.verbose:
                    pbar.set_description(
                        'Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}, L_F loss: {:.3f}'
                        .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm'], loss_dic['l_F']))

            self.save_FS_results(ref_name, gen_im, latent_in, latent_F)

    def invert_image_in_FS(self, image_path, W_init=None, F_init = None, pbar=None):
        image = load_image(image_path)

        device = self.opts.device
        
        ref_im_H = self.image_transform(image)
        ref_im_L = self.image_transform_256(image)

        if W_init is None:
            _, latent_W = self.invert_image_in_W(image_path, pbar=pbar)
        else:
            latent_W = load_latent_W(W_init, device=device)
        
        if F_init is None:
            F_init, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

        optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)
        import streamlit as st
        for step in range(self.opts.FS_steps):
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
            # if step % 20 == 0:
            #     st.image(
            #         self.tensor_to_pil(gen_im)
            #     )
            if pbar is not None:
                pbar.progress(int(step / self.opts.FS_steps * 100), text=f'Embedding to FS space ({step} / {self.opts.FS_steps})')
        if pbar is not None:
            pbar.empty()
        return gen_im.detach().clone(), latent_in.detach().clone(), latent_F.detach().clone()

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


    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True

    def tensor_to_pil(self, gen_im):
        return toPIL(((gen_im.squeeze() + 1) / 2).detach().cpu().clamp(0, 1))
    
    def tensor_to_numpy(self, gen_im):
        return np.array(self.tensor_to_pil(gen_im))
    
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
        # target_H, target_L = load_image(target_im)
        # face_H, face_L = load_image(face_im)
        # hair_H, hair_L = load_image(hair_im)
        
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