"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import numpy as np
import torch
import cv2
import glob
import torchvision.transforms as transforms
from PIL import Image
from src.models.Trimap import Trimap

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def cuda_unsqueeze(li_variables=None, device='cuda'):

    if li_variables is None:
        return None

    cuda_variables = []

    for var in li_variables:
        if not var is None:
            var = var.to(device).unsqueeze(0)
        cuda_variables.append(var)

    return cuda_variables


def convert_npy_code(latent):

    if latent.shape == (18, 512):
        latent = np.reshape(latent, (1, 18, 512))

    if latent.shape == (512,) or latent.shape == (1, 512):
        latent = np.reshape(latent, (1, 1, 512)).repeat(18, axis=1)
    return latent



def load_FS_latent(latent_path, device = "cuda"):
    if isinstance(latent_path, str):
        path, ext = os.path.splitext(latent_path)
        if ext == '.npz':
            data_dict = np.load(latent_path)
            latent_in = torch.from_numpy(data_dict['latent_in']).to(device)
            latent_F = torch.from_numpy(data_dict['latent_F']).to(device)
        else:
            basename = os.path.basename(path)
            data_dict = find("FS", basename)
            latent_in = data_dict['latent_in']
            latent_F = data_dict['latent_F']
    elif isinstance(latent_path, dict):
        latent_in = latent_path['latent_in']
        latent_F = latent_path['latent_F']
        if isinstance(latent_in, np.ndarray):
            latent_in = torch.from_numpy(latent_in).to(device)
        if isinstance(latent_F, np.ndarray):
            latent_F = torch.from_numpy(latent_F).to(device)
    # elif isinstance(latent_path, torch.Tensor):

    else:
        raise(f"Error in loading FS latent. Input data type: {type(latent_path)}")
    return latent_in, latent_F

def load_image(data):
    # if isinstance(data, str):
    #     image = Image.open(data)
    if isinstance(data, np.ndarray):
        image = Image.fromarray(data)
    elif isinstance(data, Image.Image):
        image = data
    elif isinstance(data, str):
        image = Image.open(data)
    else:
        image = Image.open(data)
    return image

def load_latent_W(data, device="cpu", allow_pickle=False):
    print("data의 데이터 타입:", type(data))
    if isinstance(data, np.ndarray):
        latent = torch.from_numpy(data).to(device)
    elif isinstance(data, str):
        path, ext = os.path.splitext(data)
        if '.npy' == ext:
            latent = torch.from_numpy(np.load(data, allow_pickle=allow_pickle)).to(device)
        else:
            basename = os.path.basename(path)
            latent = find('W+', basename)['latent']
    elif isinstance(data, torch.Tensor):
        latent = data.detach().clone()
    else:
        raise(f"input data type is wrong: {type(data)}")
    return latent

def parsing(path, ffhq, device="cuda"):
    result_dict = {}
    for path in glob.glob(os.path.join(path, f"{ffhq}.*")):
        if '.png.npz' in path:
            path = path.replace(".png.npz", ".npz")
        ext = path.split(".")[1]

        # result_dict['path'] = path
        if ext == 'png':
            image = cv2.imread(path)
            result_dict['png_bgr'] = image = cv2.imread(path)
            result_dict['png'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result_dict['png_gray'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result_dict['png_path'] = path

        elif ext == 'npy':
            result_dict['npy'] = np.load(path)
            result_dict['latent'] = torch.from_numpy(result_dict['npy']).to(device)
            result_dict['npy_path'] = path
        elif ext == 'npz':
            result_dict['npz'] = np.load(path)
            result_dict['latent_in'] = torch.from_numpy(result_dict['npz']['latent_in']).to(device)
            result_dict['latent_F'] = torch.from_numpy(result_dict['npz']['latent_F']).to(device)
            result_dict['npz_path'] = path
        else:
            continue
    return result_dict

def find(name: str, target: str, device="cuda"):
    # 절대 경로 사용
    base_path = os.path.abspath('/home/diglab/workspace/sketch-project/database')
    
    if name in os.listdir(base_path):
        saved_path = os.path.join(base_path, name)
    else:
        raise ValueError("데이터베이스에 해당 폴더 없습니다.")
    
    data_dict = parsing(saved_path, target if '.' not in target else target.split('.')[0], device=device)
    return data_dict

def find_all(ffhq: str) -> dict:
    ffhq, ext = os.path.splitext(ffhq)
    result_dict = {'name': ffhq}
    for target in ['W+', 'FS', 'ffhq', 'mask', 'bald']:
        result_dict[target] = find(target, ffhq)
    result_dict.update(get_image_dict(result_dict['ffhq']['png_bgr'], rgb=False))
    result_dict['mask'] = get_mask_dict(result_dict['mask']['png_gray'])
    return result_dict

def get_image_dict(im_bgr, device="cuda", rgb=False):
    if isinstance(im_bgr, str):
        im_bgr = cv2.imread(im_bgr)
    
    if rgb:
        im_rgb = im_bgr
        im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2BGR)
    else:
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)        


        
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    d = {
        'rgb': {},
        'bgr': {},
        'tensor': {} 
    }
    for size in [32, 64, 128, 256, 512, 1024]:
        d['rgb'][size] = cv2.resize(im_rgb, (size, size))
        d['bgr'][size] = cv2.resize(im_bgr, (size, size))
        d['tensor'][size] = image_transform(d['rgb'][size]).to(device).unsqueeze(0)
    return d


# def get_mask_dict(mask=None, im=None, get_hair_mask=None, net=None, device="cuda", dilate_kernel=0):
def get_mask_dict(im, mask=None, embedding=None, device="cuda", kernel_size=50, dilate_kernel=10):
    if mask is None:
        M_512 = embedding.get_seg(im, target=10).cpu().numpy().astype(np.uint8)
        if dilate_kernel>0:
            M_512 = cv2.dilate(M_512, kernel=np.ones((dilate_kernel, dilate_kernel), np.uint8))
        # im_rgb_512 = cv2.resize(im_rgb, (512, 512))
        # M3_512 = get_hair_mask(img_path=im, net=net, include_hat=True, include_ear=False, dilate_kernel=dilate_kernel)
        # M_512 = cv2.cvtColor(M3_512, cv2.COLOR_RGB2GRAY)
    else:
        if mask.ndim == 3:
            M_512 = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            M_512 = mask.copy()

    M_dict = {
        'numpy': {}, 
        'tensor': {}, 
    }
    for size in [32, 64, 128, 256, 512, 1024]:
        M_numpy = cv2.resize(M_512, (size, size))
        M_dict['numpy'][size] = M_numpy

        M_tensor = torch.from_numpy(M_numpy/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
        M_dict['tensor'][size] = M_tensor # torch.from_numpy(M_trimap).unsqueeze(0).unsqueeze(0).float().to(device)

    return M_dict

def get_mask_dict_backup(mask=None, im=None, get_hair_mask=None, net=None, device="cuda", kernel_size=0):
    trimap = Trimap()
    if mask is None:
        # im_rgb_512 = cv2.resize(im_rgb, (512, 512))
        M3_512 = get_hair_mask(img_path=im, net=net, include_hat=True, include_ear=False, dilate_kernel=dilate_kernel)
        M_512 = cv2.cvtColor(M3_512, cv2.COLOR_RGB2GRAY)
    else:
        if mask.ndim == 3:
            M_512 = cv2.cvtColor(M3_512, cv2.COLOR_RGB2GRAY)
        else:
            M_512 = mask.copy()
    M_1024 = cv2.resize(M_512, (1024, 1024))
    _, _, tirmap_result = trimap.mask_to_trimap(im, M_1024, kernel_size=kernel_size)
    # print(np.max(tirmap_result)) # 1.0
    tirmap_result = (tirmap_result*1.5*255).clip(0, 255).astype(np.uint8)
    del trimap

    M_dict = {
        'numpy': {}, 
        'tensor': {}, 
        'trimap': {}
    }
    for size in [32, 64, 128, 256, 512, 1024]:
        M_numpy = cv2.resize(M_512, (size, size))
        M_dict['numpy'][size] = M_numpy

        M_trimap = cv2.resize(tirmap_result, (size, size))
        M_dict['trimap'][size] = M_trimap

        M_tensor = torch.from_numpy(M_numpy/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
        M_dict['tensor'][size] = M_tensor # torch.from_numpy(M_trimap).unsqueeze(0).unsqueeze(0).float().to(device)

    return M_dict

