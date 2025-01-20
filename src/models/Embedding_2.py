import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.models.Encoder import Encoder
from src.models.Net import Net
import numpy as np
import torchvision.transforms as transforms
from functools import partial
from src.utils.bicubic import BicubicDownSample
from src.losses.embedding_loss import EmbeddingLossBuilder
from src.utils.args_utils import parse_yaml
from PIL import Image
from tqdm import tqdm
import random

opts = parse_yaml('opts/config.yml')

toPIL = transforms.ToPILImage()
net = Net(opts)
encoder = Encoder(opts.e4e, decoder=net.generator)
device = opts.device
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
image_transform_256 = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS), 
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

downsample = BicubicDownSample(factor=1024 // 256)
downsample_512 = BicubicDownSample(factor=1024 // 512)
        
loss_builder = EmbeddingLossBuilder(opts)

def image_save(gen_im, image_path) :
    save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
    save_im.save(image_path)

def numpy_to_image_save(numpy_image, image_path):
    # If the numpy array is not in the range [0, 255], scale it
    if numpy_image.max() <= 1.0:
        numpy_image = (numpy_image * 255).astype(np.uint8)
    else:
        numpy_image = numpy_image.astype(np.uint8)

    # Convert the numpy array to a PIL image
    image = Image.fromarray(numpy_image)
    # Save the image
    image.save(image_path)

def cal_loss(im_dict, latent_in, latent_F=None, F_init=None):
    loss, loss_dic = loss_builder(**im_dict)
    p_norm_loss = net.cal_p_norm_loss(latent_in)
    loss_dic['p-norm'] = p_norm_loss
    loss += p_norm_loss

    if latent_F is not None and F_init is not None:
        l_F = net.cal_l_F(latent_F, F_init)
        loss_dic['l_F'] = l_F
        loss += l_F

    return loss, loss_dic

def setup_W_optimizer(init_latent=None):
    if init_latent is not None : 
        init_latent = torch.tensor(init_latent.squeeze(), dtype=torch.float32).cuda()
        
    opt_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'sgdm': partial(torch.optim.SGD, momentum=0.9),
        'adamax': torch.optim.Adamax
    }
    latent = []
    for i in range(18):
        if init_latent is None:
            tmp = net.latent_avg.clone().detach().cuda()
        else:
            tmp = init_latent[i, :]
        tmp.requires_grad = True
        latent.append(tmp)

    optimizer_W = opt_dict[opts.opt_name](latent, lr=opts.learning_rate)

    return optimizer_W, latent



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def invert_image_in_W(image, init_latent=None, pbar=None, max_steps=300, text=''):
    ref_im_H = image_transform(image)
    ref_im_L = image_transform_256(image)
    optimizer_W, latent = setup_W_optimizer(init_latent)

    loss_values_W = []  # Initialize a list to store loss values for W+ space
    intermediate_latents = {}  # Dictionary to store intermediate latents

    pbar = tqdm(range(max_steps), desc='W+', leave=False)
    for step in pbar:
        optimizer_W.zero_grad()
        latent_in = torch.stack(latent).unsqueeze(0)

        gen_im, _ = net.generator([latent_in], input_is_latent=True, return_latents=False)
        im_dict = {
            'ref_im_H': ref_im_H.to(device),
            'ref_im_L': ref_im_L.to(device),
            'gen_im_H': gen_im,
            'gen_im_L': downsample(gen_im)
        }

        loss, _ = cal_loss(im_dict, latent_in)
        loss.backward()
        optimizer_W.step()
        pbar.set_postfix(step=step, loss=loss.item())
        loss_values_W.append(loss.item())  # Store the loss value for W+ space

        # Save the generated image and latent every 100 steps
        # if step % 100 == 0 and step != 0:
        #     image_save(gen_im, f'{text}_{step}.png')
        #     intermediate_latents[step] = latent_in.detach().clone()
        #     print(f'Saved latent for iter {step}')  # Debug print


    image_save(gen_im, f'{text}_final.png')
    intermediate_latents[max_steps] = latent_in.detach().clone()  # Save the final latent

    return gen_im.detach().clone(), latent_in.detach().clone() # Return loss values and intermediate latents


def setup_FS_optimizer(latent_W, F_init):
    latent_F = F_init.clone().detach().requires_grad_(True)
    latent_S = []
    opt_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'sgdm': partial(torch.optim.SGD, momentum=0.9),
        'adamax': torch.optim.Adamax
    }
    for i in range(net.layer_num):

        tmp = latent_W[0, i].clone()

        if i < net.S_index:
            tmp.requires_grad = False
        else:
            tmp.requires_grad = True

        latent_S.append(tmp)

    optimizer_FS = opt_dict[opts.opt_name](latent_S[net.S_index:] + [latent_F], lr=opts.learning_rate)

    return optimizer_FS, latent_F, latent_S


def invert_image_in_FS(image, W_init=None, F_init=None, pbar=None, text = '', max_steps=700):    
    ref_im_H = image_transform(image)
    ref_im_L = image_transform_256(image)

    if W_init is None:
        _, latent_W = invert_image_in_W(image, pbar=pbar, text='FS_wo_encode.png', max_steps=300)
    else:
        latent_W = W_init.clone()
    
    if F_init is None:
        F_init, _ = net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)

    optimizer_FS, latent_F, latent_S = setup_FS_optimizer(latent_W, F_init)

    pbar = tqdm(range(max_steps), desc='FS', leave=False)
    for step in pbar:
        optimizer_FS.zero_grad()
        latent_in = torch.stack(latent_S).unsqueeze(0)
        gen_im, _ = net.generator([latent_in], input_is_latent=True, return_latents=False,
                                        start_layer=4, end_layer=8, layer_in=latent_F)
        im_dict = {
            'ref_im_H': ref_im_H.to(device),
            'ref_im_L': ref_im_L.to(device),
            'gen_im_H': gen_im,
            'gen_im_L': downsample(gen_im)
        }

        loss, _ = cal_loss(im_dict, latent_in)
        loss.backward()
        optimizer_FS.step()
        pbar.set_postfix(step=step, loss=loss.item())

    return gen_im.detach().clone(), latent_in.detach().clone(), latent_F.detach().clone() # Return both loss values

max_steps = 300

# gen_im1, latent_in, loss_values_W, intermediate_latents = invert_image_in_W(img, text='w_space_embedding', pbar=None, max_steps=max_steps)

# gen_im3, latent_in, latent_F, loss_values_FS = invert_image_in_FS(
#     img,
#     W_init=intermediate_latents[steps],
#     text='',
#     pbar=None,
#     max_steps=700
# )