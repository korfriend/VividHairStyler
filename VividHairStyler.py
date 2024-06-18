import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tqdm import tqdm
from src.utils.args_utils import parse_yaml
from src.utils.color_utils import rgb_to_lab
from src.utils.data_utils import find, get_mask_dict, load_latent_W, load_FS_latent
from src.utils.sketch_utils import parse_json
from src.utils.seg_utils import vis_seg_reverse, vis_seg
from src.models.Sketch import SketchHairSalonModule
from src.models.face_parsing.model import seg_mean, seg_std
from src.models.Net import Net
from src.models.Encoder import Encoder
from src.models.Embedding import Embedding
from src.models.Alignment import Alignment
from src.models.Blending import Blending
from src.losses.blend_loss import BlendLossBuilder
from src.mapper.Bald import Bald

#region Configurations and Constants
args = parse_yaml('opts/config.yml')
device = args.device
st.set_page_config(layout="wide")

root = './database'
ffhq_dir = os.path.join(root, 'ffhq')
ffhq_wp_dir = os.path.join(root, 'W+')
ffhq_f_dir = os.path.join(root, 'FS')
output_dir = os.path.join('Output')
Result_FS_path = os.path.join(output_dir, 'Result_FS.npz')
Result_Image_path = os.path.join(output_dir, "Result_image.png")

args.output_dir = output_dir
os.makedirs(ffhq_wp_dir, exist_ok=True)
os.makedirs(ffhq_f_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

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
#endregion

#region Function Definitions
def initialize_model(args):
    net = Net(args)
    ii2s = Embedding(args, net=net)
    encoder = Encoder(args.e4e, decoder=net.generator)
    return net, ii2s, encoder

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

def set_background_image(background_choice, I_bg_rgb):
    if background_choice == "Result Image" and 'I_G_blend' in st.session_state and st.session_state.I_G_blend is not None:
        st.session_state.canvas_background = cv2.resize(st.session_state.I_G_blend, (512, 512))
        st.session_state.canvas_background2, st.session_state.canvas_background2_mask = make_hair_line_background(st.session_state.I_G_blend)
        return st.session_state.I_G_blend
    else:
        st.session_state.canvas_background = cv2.resize(I_bg_rgb, (512, 512))
        st.session_state.canvas_background2, st.session_state.canvas_background2_mask = make_hair_line_background(I_bg_rgb)
        return I_bg_rgb
    
def make_hair_line_background(img, alpha = 0.5):
    mask = ii2s.get_seg(cv2.resize(img, (1024, 1024)), target=10).detach().cpu().squeeze().numpy().astype(np.uint8) * 255
    # mask_ = mask.copy()
    mask3 = np.dstack([mask,mask,mask])
    # st.markdown(f"mask3: {mask3.shape}")
    # st.markdown(f"img: {img.shape}")
    bg = cv2.addWeighted(cv2.resize(img, (512, 512)), alpha, mask3, 1-alpha ,0)
    # bg_pil = Image.fromarray(bg)
    return bg, mask

def bald_blending(ii2s, W, F7_ori, M, layer_range=(0, 3)):
    with torch.no_grad():
        F7_bald, _ = ii2s.generator([W], input_is_latent=True, return_latents=False,
                                    start_layer=layer_range[0], end_layer=layer_range[1])
        M_32 = M
        F7_bald = F7_bald * M_32 + F7_ori * (1 - M_32)
        G_baldFS, _ = ii2s.generator([W], input_is_latent=True, return_latents=False,
                                     start_layer=layer_range[1] + 1, end_layer=8, layer_in=F7_bald)
    return F7_bald, G_baldFS

# @st.cache_data
def cache_embedding(img, encoder, ii2s):
    embedd_progress = st.progress(20, text="Embedding in progress...")
    _ , W_init = encoder.encode(np.array(img), return_is_tensor=True)
    embedd_progress.progress(60, text=f"Embedding in progress... ")
    gen_im, latent_S, latent_F = ii2s.invert_image_in_FS(img, W_init=W_init)
    embedd_progress.progress(100, text=f"Embedding in progress... ")
    embedd_progress.empty()
    return gen_im, latent_S, latent_F

def display_image_with_caption(columns, filepaths, captions, keys, indices):
    images = []
    selected_filenames = []
    latents = []
    Fs = []
    for col, filepath, cap, key, index in zip(columns, filepaths, captions, keys, indices):
        col.header(cap)
        img = None
        img_placeholder = col.empty()
        selected_filename = col.selectbox(f"Select {cap} image", filepath, index=index, key=f"{key}_select")
        uploaded_image = col.file_uploader(f"Upload {cap} image", type=["png", "jpg", "jpeg"], key=f"{key}_upload")
        run_embedding = False
        if uploaded_image is not None:
            img = np.array(Image.open(uploaded_image))
            run_embedding = True
        else:
            img_path = os.path.join(ffhq_dir, selected_filename)
            img = BGR2RGB(img_path)
        if cap != 'Source':
            text_key = f"{key}_text"
            if text_key not in st.session_state:
                st.session_state[text_key] = ""
            text = col.text_input(f"Enter text for {cap}", key=text_key)
            if text != "":
                img = text_to_image(text)
                run_embedding = True
                st.session_state[f"{key}_run_embedding"] = True  # 플래그 설정
            if f"{key}_run_embedding" in st.session_state and st.session_state[f"{key}_run_embedding"]:
                run_embedding = True
                st.session_state[f"{key}_run_embedding"] = False  # 플래그 초기화
        if run_embedding:
            net = Net(args)
            ii2s = Embedding(args, net=net)
            encoder = Encoder(args.e4e, decoder=net.generator)
            _, latent_S, latent_F = cache_embedding(img, encoder=encoder, ii2s=ii2s)
            del net, ii2s, encoder
        else:
            data_dict = find("FS", selected_filename)
            latent_S = data_dict['latent_in']
            latent_F = data_dict['latent_F']
        latents.append(latent_S)
        Fs.append(latent_F)
        img_placeholder.image(img)
        selected_filenames.append(selected_filename)
        images.append(img)
    return images[0], images[1], images[2], selected_filenames, latents, Fs

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


def process_and_save_latents(images, latents, Fs, output_dir, names, ii2s):
    masks = []
    for img, latent, fs, name in zip(images, latents, Fs, names):
        mask = get_mask_dict(im=img, mask=None, embedding=ii2s)
        masks.append(mask)
        latent_dir = os.path.join(output_dir, f'FS_{name}')
        np.savez(latent_dir, latent_in=latent.detach().cpu().numpy(), latent_F=fs.detach().cpu().numpy())
    return masks
#endregion

#region Main Streamlit Interface
torch.cuda.empty_cache()
model_progress = st.progress(0, text="Loading models...")
model_progress.progress(50, text="Loading models...")

if 'net' not in st.session_state or 'ii2s' not in st.session_state or 'encoder' not in st.session_state:
    net, ii2s, encoder = initialize_model(args)
    st.session_state.net = net
    st.session_state.ii2s = ii2s
    st.session_state.encoder = encoder
else:
    net = st.session_state.net
    ii2s = st.session_state.ii2s
    encoder = st.session_state.encoder

model_progress.progress(100, text="Loading models...")
model_progress.empty()

########### Configuring the Hair Transfer interface ###########
st.title("Hair Transfer")
run_opt = st.button('All decided')

filepath_list = sorted(os.listdir(ffhq_dir))
image_keys = ['Source', 'Structure', 'Appearance']
initial_indices = [1,0, 2]


# Initialize session state variables
if 'images' not in st.session_state:
    st.session_state.images = [None, None, None]
if 'latents' not in st.session_state:
    st.session_state.latents = [None, None, None]
if 'Fs' not in st.session_state:
    st.session_state.Fs = [None, None, None]
if 'selected_filenames' not in st.session_state:
    st.session_state.selected_filenames = ["", "", ""]


def process_image(img, key):
    _, latent_S, latent_F = cache_embedding(img, encoder=encoder, ii2s=ii2s)
    latent_dir = os.path.join(output_dir, f'FS_{key}.npz')
    np.savez(latent_dir, latent_in=latent_S.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())
    return img, latent_S, latent_F

def run_embedding(img,key, i) :
    img, latent_S, latent_F = process_image(img, key)
    st.session_state.images[i] = img
    st.session_state.latents[i] = latent_S
    st.session_state.Fs[i] = latent_F


for i, (col, filepath, key, index) in enumerate(zip(st.columns(3), [filepath_list, filepath_list, filepath_list], image_keys, initial_indices)):
    col.header(f"{key} image")
    img_placeholder = col.empty()
    selected_filename = col.selectbox(f"Select {key} image", filepath, index=index, key=f"{key}_select")
    uploaded_image = col.file_uploader(f"Upload {key} image", type=["png", "jpg", "jpeg"], key=f"{key}_upload")
    text = col.text_input(f"Enter text for {key}", key=f"{key}_text")
    
    # Determine if any input has changed

    if uploaded_image is not None and f'{uploaded_image.name}' != st.session_state.selected_filenames[i]:
        img = np.array(Image.open(uploaded_image))
        run_embedding(img, key, i)
        st.session_state.selected_filenames[i] = f'{uploaded_image.name}'
        
    elif text != "" and text != st.session_state.selected_filenames[i]:
        img = text_to_image(text)
        run_embedding(img, key, i)
        st.session_state.selected_filenames[i] = text
        
    elif uploaded_image is None and text == "" and selected_filename != st.session_state.selected_filenames[i]:
        img_path = os.path.join(ffhq_dir, selected_filename)
        img = BGR2RGB(img_path)
        run_embedding(img, key, i)
        st.session_state.selected_filenames[i] = selected_filename


    img_placeholder.image(st.session_state.images[i])

images = st.session_state.images
latents = st.session_state.latents
Fs = st.session_state.Fs
selected_filenames = st.session_state.selected_filenames
I_src_rgb, I_sref_rgb, I_aref_rgb = images
I_src_rgb_temp = I_src_rgb

M_src = get_mask_dict(im=images[0], mask=None, embedding=ii2s)
M_sref = get_mask_dict(im=images[1], mask=None, embedding=ii2s)
M_aref = get_mask_dict(im=images[2], mask=None, embedding=ii2s)

########### Configuring the Hair Editing interface ###########
st.title("Hair Editing")
sketch_completed = st.button('Sketch Completed')
st.sidebar.header("Sketch")

edit_mode = st.radio(
    "Choose editing mode:", 
    ("Hair Line Editing", "Hair Strain Editing"),
    key="edit_mode"
)

background_choice = st.radio(
    "Choose canvas background:", 
    ("Source Image", "Result Image") if 'I_G_blend' in st.session_state and st.session_state.I_G_blend is not None else ("Source Image",),
    index=1 if 'I_G_blend' in st.session_state and st.session_state.I_G_blend is not None else 0
)

I_src_rgb = set_background_image(background_choice, I_src_rgb)

can1, can2 = st.columns(2)
if edit_mode == "Hair Line Editing":
    stroke_width = st.sidebar.slider("Stroke width (line): ", 1, 100, 50, key="line_editing_stroke_width")
    stroke_color = "#000001"  # black
    eraser_mode = st.sidebar.checkbox("eraser mode", False, key='mode1', help="지우기모드를 사용합니다.")
    
    can1.header("Canvas")
    with can1:
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="#111" if eraser_mode else "#EEEEEE",
            background_color="#eee",
            background_image=Image.fromarray(st.session_state.canvas_background2),
            update_streamlit=True,
            width=512,
            height=512,
            drawing_mode='freedraw',
            point_display_radius=0,
            key="canvas2",
        )
        mask1 = canvas_result.image_data[:,:,0] > 127
        mask2 = canvas_result.image_data[:,:,0] != 0
        mask2 = np.logical_xor(mask1, mask2)

        user_mask = np.where(mask1, 255, st.session_state.canvas_background2_mask)
        user_mask = np.where(mask2, 0, user_mask)
        user_mask = user_mask > 0
        Image.fromarray(user_mask).save(os.path.join(output_dir,"user_mask.png"))

elif edit_mode == "Hair Strain Editing":
    stroke_width = st.sidebar.slider("Stroke width (structure): ", 1, 100, 5, key="structure_editing_stroke_width")
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#A52A2A", key="structure_editing_stroke_color")  # 갈색으로 설정

    can1.header("Canvas")
    with can1:
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#eee",
            background_image=Image.fromarray(st.session_state.canvas_background),
            update_streamlit=True,
            width=st.session_state.canvas_background.shape[1],
            height=st.session_state.canvas_background.shape[0],
            drawing_mode='freedraw',
            point_display_radius=0,
            key="canvas",
        )

    if canvas_result:
        user_mask = ~np.all(canvas_result.image_data[:, :, :3] == [0, 0, 0], axis=-1)

if stroke_color == "#000":
    stroke_color = "#111"

can2.header("Result Image")
if 'I_G_blend' in st.session_state and st.session_state.I_G_blend is not None:
    can2.image(cv2.resize(st.session_state.I_G_blend, (512, 512)))

if not run_opt and not sketch_completed:
    sys.exit()
#endregion
W_src, F7_src = latents[0].clone(), Fs[0].clone()
W_sref, F7_sref = latents[1].clone(), Fs[1].clone()
W_aref, F7_aref = latents[2].clone(), Fs[2].clone()


align = Alignment(args, embedding=ii2s)


if run_opt:
    I_src_rgb = I_src_rgb_temp
    I_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(I_src_rgb).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
    I_3 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(I_aref_rgb).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)

    src_kp_hm = align.kp_extractor.face_alignment_net(I_1)

    #region Alignment
    # HM_img = BGR2RGB("/home/diglab/workspace/sketch-project/now_images/20240531-1069_240606_193348_00090/new_target_mask.png")
    # generated_image = torch.from_numpy(vis_seg_reverse(HM_img)).to(device)
    # warped_latent_2 = load_latent_W("/home/diglab/workspace/sketch-project/now_images/20240531-1069_240606_193348_00090/warped_latent_2.npy").to(device)

    generated_image = None
    warped_latent_2 = None

    # I_glign_1_2, F7_blend_1_2, HM_1_2, M_hole, M_hair, M_src, bald_seg_target1, target_mask, warped_latent_2, seg_target2, inpaint_seg, bald_target1 = align.M2H_test(
    #     generated_image, warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir, baldFS_dir, bald_dir, save_dir, 
    #     all_inpainting=False, init_align=True, sign=args.sign, smooth=args.smooth)

    bald_module = Bald(args.bald_model_path)
    W_src_bald = bald_module.make_bald(W_src)
    del bald_module
    
    I_glign_1_2, F7_blend_1_2, warped_latent_2, bald_seg_target1, target_mask, HM_1_2 =  align.align_images(I_src_rgb, I_sref_rgb, F7_src, W_src, W_sref, W_src_bald, smooth=args.smooth)
    st.image(ii2s.tensor_to_numpy(I_glign_1_2))
    #endregion  

    # mask
    HM_1D, _ = align.dilate_erosion(M_src['tensor'][1024], device)
    HM_3D, HM_3E = align.dilate_erosion(M_aref['tensor'][1024], device)
    HM_1_2 = HM_1_2.unsqueeze(0).unsqueeze(0).cpu()
    target_hairmask = (HM_1_2 == 10) * 1.0
    target_hairmask = target_hairmask.float()
    HM_1_2D, HM_1_2E = align.dilate_erosion(target_hairmask, device)
    downsampled_hair_mask = F.interpolate(HM_1_2E, size=(256, 256), mode='area')
    upsampled_hair_mask = F.interpolate(HM_1_2E, size=(1024, 1024), mode='area')
    

    with torch.no_grad():
        I_X, _ = ii2s.generator([W_src], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
        I_X_0_1 = (I_X + 1) / 2
        IM = (align.downsample(I_X_0_1) - seg_mean) / seg_std
        down_seg, _, _ = ii2s.seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().cpu().float()
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        HM_X = F.interpolate(HM_X.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()
        HM_XD, _ = align.cuda_unsqueeze(align.dilate_erosion_mask_tensor(HM_X), device)
        target_mask = (1 - HM_1D) * (1 - HM_1_2D) * (1 - HM_XD)

    F_mixed = F7_blend_1_2
    F_hair = F7_aref
    interpolation_latent = ((W_src + W_aref) / 2).detach().clone().requires_grad_(True)

    im_dict = {
        'im_1': I_1,
        'im_3': I_3,
        'mask_face': target_mask,
        'mask_hair': HM_3E,
        'mask_2_hair': downsampled_hair_mask,
    }


elif sketch_completed and user_mask is not None :
    if edit_mode == "Hair Line Editing" :
        bald_module = Bald(args.bald_model_path)
        W_src_bald = bald_module.make_bald(W_src)

        del bald_module
        F7_bald, G_baldFS = bald_blending(ii2s, W_src_bald, F7_src, M_src['tensor'][32]*255, layer_range=(0, 3))

        bald_seg_mask = ii2s.get_seg(G_baldFS, target=None)
        bald_target1 = torch.argmax(bald_seg_mask, dim=1).long()
        bald_target1 = bald_target1[0].byte()
        new_bald_seg = torch.where(torch.from_numpy(user_mask).to(device)==1, 10 * torch.ones_like(bald_target1), bald_target1)
        
        if background_choice == "Result Image" and st.session_state.I_G_blend is not None :
            F7_path = 'Result_FS.npz'
            I_glign_line, F7_blend_line, HM_line, new_hair_mask = align.Hair_Line_editing(new_bald_seg, F7_bald, os.path.join(output_dir, F7_path), smooth=args.smooth)
            I_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(ii2s.tensor_to_numpy(I_glign_line)).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
            F_mixed = F7_blend_line
            
            result_S, result_F = load_FS_latent(Result_FS_path)

            result_img = BGR2RGB(Result_Image_path)
            I_3 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(result_img).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)

            result_seg_mask = ii2s.get_seg(Result_Image_path, target=None)
            result_target1 = torch.argmax(result_seg_mask, dim=1).long()
            result_target1 = result_target1[0].byte()
            result_hair_mask = torch.where(result_target1==10, torch.ones_like(result_target1),torch.zeros_like(result_target1))
            HM_1D, _ = align.dilate_erosion(result_hair_mask.unsqueeze(0).unsqueeze(0), device)

            F_hair = result_F
            interpolation_latent = result_S.detach().clone().requires_grad_(True)

        else : 
            F7_path = 'FS_Source.npz'
            I_glign_line, F7_blend_line, HM_line, new_hair_mask = align.Hair_Line_editing(new_bald_seg, F7_bald, os.path.join(output_dir, F7_path), smooth=args.smooth)
            I_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(ii2s.tensor_to_numpy(I_glign_line)).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
            F_mixed = F7_blend_line

            I_3 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(I_src_rgb).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
            HM_1D, _ = align.dilate_erosion(M_src['tensor'][1024], device)

            F_hair = F7_src
            interpolation_latent = W_src.detach().clone().requires_grad_(True)
            
        st.image(ii2s.tensor_to_numpy(I_glign_line))

        with torch.no_grad():
            HM_newD, HM_newE = align.dilate_erosion(new_hair_mask, device)
            downsampled_hair_mask = F.interpolate(HM_newE, size=(256, 256), mode='bilinear', align_corners=False)  
            I_X, _ = ii2s.generator([interpolation_latent], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_line)
            I_X_0_1 = (I_X + 1) / 2
            IM = (align.downsample(I_X_0_1) - seg_mean) / seg_std
            down_seg, _, _ = ii2s.seg(IM)
            current_mask = torch.argmax(down_seg, dim=1).long().cpu().float()
            HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
            HM_X = F.interpolate(HM_X.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()
            HM_XD, _ = align.cuda_unsqueeze(align.dilate_erosion_mask_tensor(HM_X), device)
            target_mask = (1 - HM_1D) * (1 - HM_newD) * (1 - HM_XD)
        
        
        im_dict = {
            'im_1': I_1,
            'im_3': I_3,
            'mask_face': target_mask,
            'mask_hair': HM_1D,
            'mask_2_hair': downsampled_hair_mask,
        }


    elif edit_mode == "Hair Strain Editing" :
        sketch_rgb = parse_json(canvas_result.json_data)
        SHS = SketchHairSalonModule(args.S2M, args.S2I)
        matte_512, sketch_rgb_new = SHS.get_matte_and_image(sketch_rgb, background=I_src_rgb)
        del SHS

        binary_sketch_mask = (matte_512 >= 128).astype(np.uint8) * 255
        image, latent = encoder.encode(sketch_rgb_new)
        pil_new = Image.fromarray(sketch_rgb_new)
        resized_pil_new = pil_new.resize((1024, 1024), Image.LANCZOS)
        I_new, W_new = ii2s.invert_image_in_W_without_path(resized_pil_new, init_latent=latent)

        sketch_seg_mask = ii2s.get_seg(I_new, target=None)
        sketch_target1 = torch.argmax(sketch_seg_mask, dim=1).long()
        sketch_target1 = sketch_target1[0].byte()
        sketch_hair_mask = torch.where(sketch_target1 == 10, torch.ones_like(sketch_target1), torch.zeros_like(sketch_target1))
        new_hair_mask = torch.from_numpy(binary_sketch_mask).cpu() * sketch_hair_mask.cpu()
        new_hair_mask, _ = align.dilate_erosion(new_hair_mask.unsqueeze(0).unsqueeze(0), device)
        new_hair_mask_down_32 = F.interpolate(new_hair_mask.float(), size=(32, 32), mode='area')[0]

        F_sketch, _ = ii2s.generator([W_new], input_is_latent=True, return_latents=False,
                                        start_layer=0, end_layer=3)
        
        I_3 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(sketch_rgb_new).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
        
        if background_choice == "Result Image" and st.session_state.I_G_blend is not None :
            result_img = BGR2RGB(Result_Image_path)
            I_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(result_img).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)

            result_seg_mask = ii2s.get_seg(Result_Image_path, target=None)
            result_target1 = torch.argmax(result_seg_mask, dim=1).long()
            result_target1 = result_target1[0].byte()
            result_hair_mask = torch.where(result_target1==10, torch.ones_like(result_target1),torch.zeros_like(result_target1))
            HM_1D, _ = align.dilate_erosion(result_hair_mask.unsqueeze(0).unsqueeze(0), device)

            result_S, result_F = load_FS_latent(Result_FS_path)
            F_mixed = result_F + new_hair_mask_down_32*(F_sketch-result_F)
            interpolation_latent = result_S.detach().clone().requires_grad_(True)
        else :  
            I_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(I_src_rgb).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
            F_mixed = F7_src + new_hair_mask_down_32*(F_sketch-F7_src)
            HM_1D, _ = align.dilate_erosion(M_src['tensor'][1024], device)
            interpolation_latent = W_src.detach().clone().requires_grad_(True)

        hair_mask = HM_1D + new_hair_mask
        HM_newD, HM_newE = align.dilate_erosion(hair_mask.float(), device)
        downsampled_hair_mask = F.interpolate(HM_newE, size=(256, 256), mode='bilinear', align_corners=False)

        im_dict = {
            'im_1': I_1,
            'im_3': I_3,
            'mask_face': 1- HM_newD,
            'mask_hair': HM_newD,
            'mask_2_hair': downsampled_hair_mask,
        }
        
#region Blending
del align
blend = Blending(args, embedding=ii2s)
loss_builder = BlendLossBuilder(args)
opt_blend = torch.optim.Adam([interpolation_latent], lr=ii2s.opts.learning_rate)

pbar = tqdm(range(150), desc='Blend', leave=False)
blend_progress = st.progress(0, text="Blending in progress...")
for step in pbar:
    opt_blend.zero_grad()
    latent_mixed = interpolation_latent
    
    I_G, _ = ii2s.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F_mixed)
    if run_opt or edit_mode == "Hair Line Editing":
        I_G_color, _ = ii2s.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F_hair)
    
    im_dict['gen_im'] = blend.downsample_256(I_G)

    total_loss = 0

    face_loss = loss_builder._loss_face_percept(im_dict['gen_im'], im_dict['im_1'], im_dict['mask_face']) 
    H1_region = im_dict['gen_im'] * im_dict['mask_2_hair'] 
    H2_region = im_dict['im_3'] * im_dict['mask_hair']
    style_loss = loss_builder.style_loss(H2_region, H1_region, im_dict['mask_hair'], im_dict['mask_2_hair'])

    if run_opt or edit_mode == "Hair Line Editing":
        hair_loss = loss_builder._loss_hair_percept(blend.downsample_256(I_G_color), im_dict['im_3'], im_dict['mask_hair'])
    else :
        hair_loss = loss_builder._loss_hair_percept(im_dict['gen_im'], im_dict['im_3'], im_dict['mask_hair'])

    total_loss += face_loss + hair_loss + 5000 * style_loss
    opt_blend.zero_grad()
    total_loss.backward(retain_graph=True)
    opt_blend.step()
        
    blend_progress.progress((step + 1) / 150, text=f"Blending in progress... ({step + 1}/150)")

blend_progress.empty()

# st.image(ii2s.tensor_to_numpy(im_dict['mask_hair']))
# st.image(ii2s.tensor_to_numpy(im_dict['mask_2_hair']))
# st.image(ii2s.tensor_to_numpy(I_G_color))

I_G_blend, _ = ii2s.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F_mixed)
np.savez(Result_FS_path, latent_in=latent_mixed.detach().cpu().numpy(),
            latent_F=F_mixed.detach().cpu().numpy())
Image.fromarray(ii2s.tensor_to_numpy(I_G_blend)).save(Result_Image_path)

st.session_state.I_G_blend = ii2s.tensor_to_numpy(I_G_blend)
st.session_state.canvas_background = cv2.resize(st.session_state.I_G_blend, (512, 512))
st.session_state.canvas_background2, st.session_state.canvas_background2_mask = make_hair_line_background(st.session_state.I_G_blend)
#endregion


del blend, loss_builder, ii2s
st.experimental_rerun()

print('-----')
#endregion
