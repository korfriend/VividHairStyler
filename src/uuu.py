import yaml
from argparse import Namespace
import matplotlib.pyplot as plt
import os
import cv2
import numpy as  np
import torch
import glob
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def open_yaml(path):
    base_name = '_base'
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    if base_name in data.keys() and os.path.exists(data[base_name]):
        base_data = open_yaml(data[base_name])
        base_data.update(data)
        return base_data
    
    return data

def parse_yaml(yml_path: 'str'):
    return Namespace(**open_yaml(yml_path))


def imshow(image_list, title_list=None, rgb_list=None, figsize=(10,6)):
    n_images = len(image_list)
    plt.figure(figsize=figsize)
    for i in range(n_images):
        image = image_list[i]
        if rgb_list is not None:
            if rgb_list[i] == 'bgr':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n_images, i+1)
        plt.imshow(image)
        plt.title(title_list[i] if title_list is not None else f"image {i+1}")
        plt.axis('off')
    plt.show()

def MapperPreprocessing(mask_path, dilate_kernel_size, blur_kernel_size, input_is_numpy=False):
    if input_is_numpy:
        hair_mask = mask_path
    else:
        hair_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_dilate = cv2.dilate(hair_mask,
                                kernel=np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8))
    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(blur_kernel_size, blur_kernel_size))
    mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)
    face_mask = 255 - mask_dilate_blur
    return face_mask, hair_mask

def blend(
    image, 
    edited_image, 
    face_mask, 
):
    index = np.where(face_mask > 0)
    cy = (np.min(index[0]) + np.max(index[0])) // 2
    cx = (np.min(index[1]) + np.max(index[1])) // 2
    center = (cx, cy)
    mixed_clone = cv2.seamlessClone(image, edited_image, face_mask, center, cv2.NORMAL_CLONE)
    return mixed_clone

def pencil_sketch(img, size=None):
    if size is None:
        size = np.random.randint(3, 31)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), size) # 밝은 곳은 더 밝게 어두운 곳은 더 어둡게 해야 스케치스러운 느낌을 받을 수 있다
    dst = cv2.divide(gray, blr, scale=255) # 흑백영상을 블러로 나눈 값을 255로 곱함.
    return dst

def to_sketch(from_im_rgb, size=3, threshold=255):
    pencil_gray = pencil_sketch(from_im_rgb, size)
    pencil_mask = pencil_gray < threshold
    pencil_mask_3 = np.dstack([pencil_mask,pencil_mask,pencil_mask])
    pencil_rgb = np.where(pencil_mask_3, from_im_rgb, 255)
    return pencil_rgb

def rgb_to_lab(rgb):
    # RGB에서 sRGB로 (정규화된 값을 0-1로 스케일 조정)
    rgb = (rgb + 1.0) / 2.0

    # sRGB를 XYZ로 변환
    def srgb_to_xyz(srgb):
        mask = srgb > 0.04045
        srgb[mask] = ((srgb[mask] + 0.055) / 1.055) ** 2.4
        srgb[~mask] /= 12.92
        srgb *= 100.0
        xyz = torch.matmul(srgb, torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                               [0.2126729, 0.7151522, 0.0721750],
                                               [0.0193339, 0.1191920, 0.9503041]]).to(srgb.device))
        return xyz

    # XYZ를 LAB로 변환
    def xyz_to_lab(xyz):
        xyz /= torch.tensor([95.047, 100.000, 108.883]).to(xyz.device)
        mask = xyz > 0.008856
        xyz[mask] = torch.pow(xyz[mask], 1/3)
        xyz[~mask] = (7.787 * xyz[~mask]) + (16/116)
        # LAB 변환 수행
        L = 116 * xyz[..., 1] - 16
        a = 500 * (xyz[..., 0] - xyz[..., 1])
        b = 200 * (xyz[..., 1] - xyz[..., 2])
        lab = torch.stack([L, a, b], dim=-1)  # 마지막 차원을 기준으로 쌓음
        return lab

    # sRGB로 변환
    rgb = rgb.permute(0, 2, 3, 1)  # BCHW -> BHWC
    xyz = srgb_to_xyz(rgb)
    # print(f"xyz: {xyz.shape}")
    lab = xyz_to_lab(xyz)
    # print(f"lab: {lab.shape}")
    lab = lab.permute(0, 3, 1, 2)  # BHWC -> BCHW
    return lab

# rgb_to_lab(gen_im).shape

    

def rgb_to_grayscale(rgb_image):
    """
    RGB 이미지를 그레이스케일로 변환하는 함수
    :param rgb_image: (N, C, H, W) 형태의 텐서. C는 3이어야 합니다(RGB).
    :return: (N, 1, H, W) 형태의 그레이스케일 이미지
    """
    if rgb_image.size(1) != 3:
        raise ValueError("입력 이미지는 RGB 채널을 가져야 합니다.")
    
    # RGB 채널에 대한 가중치
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=rgb_image.device)
    weights = weights.view(1, 3, 1, 1)  # 연산을 위한 차원 조정
    
    # 가중 평균을 계산하여 그레이스케일 이미지 생성
    grayscale_image = torch.sum(rgb_image * weights, dim=1, keepdim=True)
    rgb_image = grayscale_image.repeat(1, 3, 1, 1)

    return rgb_image


def parsing(path, ffhq, device="cuda"):
    result_dict = {}
    # st.markdown(os.path.join(path, f"{ffhq}*"))
    for path in glob.glob(os.path.join(path, f"{ffhq}.*")):
        # import streamlit as st
        # st.markdown(f"path: {path}")
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

def find(name: str, target: str):
    if name in os.listdir('database'):
        saved_path = os.path.join("database", name)
        # target = 
    else:
        assert "데이터베이스에 해당 폴더 없습니다."
    return parsing(saved_path, target)

def find_all(ffhq: str) -> dict:
    ffhq, ext = os.path.splitext(ffhq)
    result_dict = {'name': ffhq}
    for target in ['W+', 'FS', 'ffhq', 'mask', 'bald']:
        result_dict[target] = find(target, ffhq)
    result_dict.update(get_image_dict(result_dict['ffhq']['png_bgr'], rgb=False))
    result_dict['mask'] = get_mask_dict(result_dict['mask']['png_gray'])
    return result_dict


def parse_json(json_data):
    # JSON 데이터 로드
    drawing_data = json_data # json.loads(json_data)
    paths = drawing_data['objects']  # 'objects' 키에 선의 정보가 저장되어 있음
    img = Image.new('RGB', (512, 512), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    new_sketch_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    
    for i, path in enumerate(paths):
        if path['type'] == 'path':  # 선의 타입 확인
            # 배경을 검은색으로 설정한 새 이미지 생성

            # 선의 색상과 두께를 추출
            stroke_color = path.get('stroke', '#000000')  # 기본값은 검은색
            stroke_width = path.get('strokeWidth', 5)  # 기본값은 3
            # 선의 좌표를 추출하고 그림
            points = path['path']  # 선의 좌표
            for j in range(1, len(points)):
                start_point = (points[j-1][1], points[j-1][2])
                end_point = (points[j][1], points[j][2])
                draw.line([start_point, end_point], fill=stroke_color, width=stroke_width)
    sketch = np.array(img)
    # sketch_mask = ~np.all(sketch == [0, 0, 0], axis=-1)
    # sketch_mask3 = np.dstack([sketch_mask,sketch_mask,sketch_mask])
    # new_sketch_rgb[sketch_mask3] = sketch[sketch_mask3]
    return sketch

def get_unique_colors(im, except_black=False):
    unique_colors = np.unique(im.reshape(-1, im.shape[2]), axis=0)
    if except_black:
        unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # 검정색 제거
    return unique_colors

def F_blending(ii2s, L1, L2, mask, start_layer=0 , end_layer=3):
    with torch.no_grad():
        if len(L1.shape) != 4:
            L1, _ = ii2s.generator([L1], input_is_latent=True, return_latents=False,
                                                            start_layer=start_layer, end_layer=end_layer)
        if len(L2.shape) != 4:
            L2, _ = ii2s.generator([L2], input_is_latent=True, return_latents=False,
                                                            start_layer=start_layer, end_layer=end_layer)
    return L1*mask + L2*(1-mask)

import torchvision.transforms as transforms

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


def get_mask_dict(mask=None, im=None, get_hair_mask=None, net=None, device="cuda", dilate_kernel=0):
    # trimap = Trimap()
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
    # _, _, tirmap_result = trimap.mask_to_trimap(im, M_1024, kernel_size=im.shape[0]//8)
    # print(np.max(tirmap_result)) # 1.0
    # tirmap_result = (tirmap_result*1.5*255).clip(0, 255).astype(np.uint8)
    M_dict = {
        'numpy': {}, 
        'tensor': {}, 
        # 'trimap': {}
    }
    for size in [32, 64, 128, 256, 512, 1024]:
        M_numpy = cv2.resize(M_512, (size, size))
        M_dict['numpy'][size] = M_numpy

        # M_trimap = cv2.resize(tirmap_result, (size, size))
        # M_dict['trimap'][size] = M_trimap

        M_tensor = torch.from_numpy(M_numpy/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
        M_dict['tensor'][size] = M_tensor # torch.from_numpy(M_trimap).unsqueeze(0).unsqueeze(0).float().to(device)

    return M_dict