import sys
sys.path.insert(0, 'src')

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import os
import numpy as np
import cv2
import torch

import tqdm
import time
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


from pathlib import Path
from src.utils.parse_face import parse_face
from src.utils.args_utils import parse_yaml
from src.utils.color_utils import rgb_to_lab
from src.utils.data_utils import find, get_mask_dict
from src.utils.sketch_utils import parse_json
from src.models.Sketch import SketchHairSalonModule 
from src.models.Net import Net
from src.models.Encoder import Encoder
from src.models.Embedding import Embedding
from src.models.Alignment import Alignment
from src.models.Alignment2 import Alignment as Alignment2
from src.models.Blending import Blending
from src.losses.blend_loss import BlendLossBuilder
from src.models.face_parsing.model import seg_mean, seg_std
torch.cuda.empty_cache()

args = parse_yaml('configs/config.yml')
device = args.device
args.save_dir = "_temps"
args.output_dir = "_temps"
st.set_page_config(layout="wide")

formatted_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
os.makedirs("temp_images", exist_ok=True)
n_saved_image = len(os.listdir('now_images'))


#region Function


def process_filename(filename, output_dir, ffhq_dir):
    if ".png" in filename:
        name, _ = os.path.splitext(filename)
        image_path = os.path.join(ffhq_dir, filename)
        background_path = image_path
    else:
        filename_split = filename.split('_')
        name = filename_split[1]
        if len(filename_split) > 4:
            for next_word in filename_split[2:-2]:
                name = f"{name}_{next_word}"
        background_path = os.path.join(output_dir, filename, 'background.png')
        filename = f"{name}.png"
        image_path = os.path.join(ffhq_dir, filename)
    return name, filename, image_path, background_path

def BGR2RGB(image_path):
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise FileNotFoundError(f"File not found or unable to read: {image_path}")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def save_tensor_as_image(tensor, filename):
    image = transforms.ToPILImage()(tensor.squeeze(0))
    image.save(filename)

# @st.cache
# def cache_parse_face(image):

@st.cache_data
def cache_encode_image(image):
    result, latent = encoder.encode(np.array(image), return_is_tensor=True)
    return result, latent.detach().cpu().numpy()

@st.cache_data
def cache_embedding(img):
    _, W_init = cache_encode_image(img)
    gen_im, latent_S, latent_F = ii2s.invert_image_in_FS(img, W_init=torch.from_numpy(W_init).to(device))
    return gen_im, latent_S, latent_F.detach().cpu().numpy()

# Define functions
def display_image_with_caption(columns, filepaths, captions, keys, indices):
    
    # pbar.progress(66)
    
    images = []
    texts = []  # 리스트를 만들어 텍스트를 저장합니다.
    selected_filenames = []
    latents = []
    Fs = []
    for col, filepath, cap, key, index in zip(columns, filepaths, captions, keys, indices):
        col.header(cap)
        
        # Initialize the image variable
        img = None
        
        # Display the image first
        img_placeholder = col.empty()
        
        # Selectbox and file uploader for each image
        selected_filename = col.selectbox(f"Select {cap} image", filepath, index=index, key=f"{key}_select")
        uploaded_image = col.file_uploader(f"Upload {cap} image", type=["png", "jpg", "jpeg"], key=f"{key}_upload")

        run_embedding = False
        if uploaded_image is not None:
            img_pil = Image.open(uploaded_image)
            img_pil.save(os.path.join(args.cache_dir, "uploaded_image.png"))
            img = np.array(img_pil)
            
            run_embedding = True
        else:
            img_path = os.path.join(ffhq_dir, selected_filename)
            img = BGR2RGB(img_path)
        
        # Add text input box for 'shape' and 'color' only
        if cap != 'Original':
            text = col.text_input(f"Enter text for {cap}", key=f"{key}_text")
            if text != "":
                run_embedding = True
                img = text_to_image(text)
        if run_embedding:
            # st.header(f"img: {img.shape}")
            
            # path = Path(os.path.join(args.cache_dir, "uploaded_imrage.png"))
            images = parse_face(os.path.join(args.cache_dir, "uploaded_image.png"))
            img = np.array(images[0])
            if img.shape[2] == 4:  # RGBA -> BGRA
                img = img[:, :, :3]
            _, latent_S, latent_F = cache_embedding(img)
            latent_F = torch.from_numpy(latent_F).to(device)
            # result, latent = encoder.encode(img, return_is_tensor=True)
            # gen_im, latent_S, latent_F = ii2s.invert_image_in_FS(img, W_init=latent)
            # selected_filename = None
        else:
            data_dict = find("FS", selected_filename)
            latent_S = data_dict['latent_in']
            latent_F = data_dict['latent_F']
        latents.append(latent_S)
        Fs.append(latent_F)
        # Display the image after selectbox and file uploader
        img_placeholder.image(img)
        selected_filenames.append(selected_filename)

        # Append the loaded image to the images list
        images.append(img)
    
    return images, selected_filenames, latents, Fs

def text_to_image(prompt, num_inference_steps=25):
    repo_id = "stabilityai/stable-diffusion-2-base"
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    try:
        image = pipe(f"{prompt}, portrait, realistic photo, real human, gray background, looking straight ahead, one person", num_inference_steps=num_inference_steps).images[0]
        result_image = cv2.resize(np.array(image).astype(np.uint8), (1024, 1024))
    except:
        result_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    finally:
        del pipe
        torch.cuda.empty_cache()
    return result_image

#endregion Function

# Data path
root = 'database'
ffhq_dir = os.path.join(root, 'ffhq')
bald_dir = os.path.join(root, 'bald')
baldFS_dir = os.path.join(root, 'baldFS')
ffhq_wp_dir = os.path.join(root, 'W+')
ffhq_f_dir = os.path.join(root, 'FS')
masked_hair_dir = os.path.join(root,'masked')
output_dir = os.path.join('test_images')
os.makedirs(output_dir, exist_ok=True)

# Side bar
st.sidebar.header("Sketch")
stroke_width = st.sidebar.slider("Stroke width: ", 3, 75, 50)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#E00")
user_sketch = st.sidebar.checkbox("use user sketch", key='use_user_sketch')
if stroke_color == "#000":
    stroke_color = "#111"

l2_lambda = 1.0
percept_lambda = 1.0
p_norm_lambda = 1.0
style_lambda = 100.0
epoch = 150
log_interval = 15

args.l2_lambda = l2_lambda
args.percept_lambda = percept_lambda
args.p_norm_lambda = p_norm_lambda / 10
args.style_lambda = style_lambda / 10




filepath_list = sorted(os.listdir(ffhq_dir))
image_keys = ['original', 'shape', 'color']
initial_indices = [0, 16, 20]  # Indices for 'original', 'shape', 'color' respectively

# Display title and separator
st.title("Hair Transfer")

# Display images with selectbox and file uploader
images, selected_filenames, Ws, Fs = display_image_with_caption(
    st.columns(3), 
    [filepath_list, filepath_list, filepath_list], 
    ['Original', 'Shape', 'Color'], 
    image_keys,
    initial_indices
)
images_ = []
for im in images:
    images_.append(np.array(im))
images = images_

FSs = []
for W, FF in zip(Ws, Fs):
    FSs.append({
        'latent_in': W, 
        'latent_F': FF
    })
color = np.array([
    [0, 0, 0],  ## 0
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
    [0, 51, 0],     ## 13
    [102, 153, 255],  ## 14
    [255, 153, 102]   ## 15
])
def rgb_to_hex(rgb_array):
    return ['#%02x%02x%02x' % tuple(rgb) for rgb in rgb_array]

def hex_to_rgb(hex_color):
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')
    
    # Convert 3 character hex to 6 character hex
    if len(hex_color) == 3:
        hex_color = ''.join([char * 2 for char in hex_color])
    
    # Convert hex to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_selected_checkboxes():
    selected_indices = []
    for i in range(16):
        if st.sidebar.checkbox(f"Checkbox {i}", key=f"cb{i}"):
            selected_indices.append(i)
    return selected_indices




### User hair mask test

st.header("Mask test")
can1, can2 = st.columns(2)
with can1:
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#eee",
        background_image=Image.fromarray(images[1]),
        update_streamlit=False,
        width=512,
        height=512,
        drawing_mode='freedraw',
        point_display_radius=0,
        key="canvas",
    )    
if canvas_result is None:
    # user_mask = np.zeros((512, 512), dtype=np.uint8)
    user_mask = None
else:
    # user_mask = parse_json(canvas_result.json_data)
    user_mask = canvas_result.image_data
    # st.markdown(f"user_mask:{user_mask.shape}")
    user_mask = user_mask[:,:,:3]
    user_mask = ~np.all(user_mask == [0, 0, 0], axis=-1)
    # user_mask = ((user_mask > 0)*255).astype(np.uint8)
    kernel_size = 10  # 스트로크 두께를 더 두껍게 하려면 이 값을 증가시키세요.
    # user_mask = cv2.dilate(user_mask, kernel=np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    can2.image(user_mask)

# run = st.button("소녀시대가 부릅니다! run devil run! \nyou better run~ run~ run~ run~ run~")
# if not run:
#     sys.exit()


run = st.button("이미지 바뀔 경우 눌러주세요")
if run:
    I_glign_1_2 = None
    I_G_blend1 = None
cs = st.columns(2)
cs[0].header("Align results")
cs[1].header("Blend results")

I_glign_1_2 = None
if I_glign_1_2 is None or run:

    progress_text = "Loading models..."
    pbar = st.progress(0, text=progress_text)
    net = Net(args)
    pbar.progress(33, progress_text)
    ii2s = Embedding(args, net=net)
    pbar.progress(66, progress_text)
    # encoder = Encoder(args.e4e, decoder = net.generator)
    # pbar.progress(60, progress_text)
    # align1 = Alignment(args, embedding=ii2s)
    align2 = Alignment2(args, embedding=ii2s)
    # pbar.progress(80, progress_text)
    # blend = Blending(args, embedding = ii2s)
    pbar.empty()
    st.markdown("---")

    if user_sketch:
        can1.image(
            Image.open("./_temp/r_result.png")
        )

    I_glign_1_2, F7_blend_1_2, HM_1_2 = align2.M2H_test(
            None, 
            img_path1 = images[0], 
            img_path2 = images[1], 
            save_dir = args.save_dir, 
            latent_FS_path_1 = FSs[0], 
            latent_FS_path_2 = FSs[1], 
            latent_W_path_1 = Ws[0], 
            latent_W_path_2 = Ws[1], 
            sign=args.sign, 
            align_more_region=False, 
            smooth=args.smooth, 
            save_intermediate=False, 
            user_mask = user_mask, 
            user_sketch = user_sketch,
            pbar = pbar)
        # return ii2s.tensor_to_numpy(I_glign_1_2), F7_blend_1_2.detach().clone().cpu().numpy(), HM_1_2.detach().clone().cpu().numpy()

    @st.cache_data
    def cache_align_image(I_glign_1_2):
        return I_glign_1_2
        
    # I_glign_1_2, F7_blend_1_2, HM_1_2 = get_align_image()
    # I_glign_1_2, F7_blend_1_2, HM_1_2 = cache_align_image(I_glign_1_2, F7_blend_1_2, HM_1_2)
    cs[0].image(cache_align_image(ii2s.tensor_to_numpy(I_glign_1_2)), caption='M2H_test, w/ sketch')
    del align2
    del ii2s
    del net

I_G_blend1 = None
if I_G_blend1 is None or run:
    progress_text = "Loading models..."
    pbar = st.progress(0, text=progress_text)
    net = Net(args)
    pbar.progress(33, progress_text)
    ii2s = Embedding(args, net=net)
    pbar.progress(66, progress_text)
    # encoder = Encoder(args.e4e, decoder = net.generator)
    # pbar.progress(60, progress_text)
    # align1 = Alignment(args, embedding=ii2s)
    # align2 = Alignment2(args, embedding=ii2s)
    # pbar.progress(80, progress_text)
    blend = Blending(args, embedding = ii2s)
    pbar.empty()
    st.markdown("---")


    M_ori = get_mask_dict(im=images_[0], mask=None, embedding=ii2s)
    M_shape = get_mask_dict(im=images[1], mask=None, embedding=ii2s)
    M_color = get_mask_dict(im=images[2], mask=None, embedding=ii2s)

    # @st.cache_data
    def get_blend_images():
        I_G_blend1 = blend.blend_images(
            images=images, 
            Ws = Ws, 
            Fs = Fs,
            F7_blend_1_2=F7_blend_1_2, 
            HM_1_2=HM_1_2, 
            m1_tensor_1024=M_ori['tensor'][1024], 
            m3_tensor_1024=M_color['tensor'][1024], 
            pbar=None
        )
        return ii2s.tensor_to_numpy(I_G_blend1)

    I_G_blend1 = get_blend_images()
    cs[1].image(I_G_blend1)

    del blend
    del ii2s
    del net
st.markdown('---')






cs = st.columns(2)
selected_indices = get_selected_checkboxes()
if len(selected_indices) == 0:
    stroke_color = "#000"
else:
    stroke_color = rgb_to_hex(color)[selected_indices[0]]

cs = st.columns(2)
sketch_bg = Image.open('./_temp/seg.png')
with cs[0]:
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#eee",
        background_image=sketch_bg,
        update_streamlit=True,
        width=512,
        height=512,
        drawing_mode='freedraw',
        point_display_radius=0,
        key="segment-canvas",
    )    
if canvas_result is None:
    # user_mask = np.zeros((512, 512), dtype=np.uint8)
    user_mask = None
else:
    sketch = parse_json(canvas_result.json_data, bg=sketch_bg)
    # sketch_mask = ~np.all(sketch == [0, 0, 0], axis=-1)
    cs[1].image(sketch)

print(formatted_time)
# del blend, align1, align2, ii2s, encoder
# del net
sys.exit()