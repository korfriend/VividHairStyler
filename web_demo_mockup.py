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

from src.utils.args_utils import parse_yaml
from src.utils.color_utils import rgb_to_lab
from src.utils.data_utils import find, get_mask_dict
from src.utils.sketch_utils import parse_json
from src.models.Sketch import SketchHairSalonModule 
from src.models.Net import Net
from src.models.Encoder import Encoder
from src.models.Embedding import Embedding
from src.models.Alignment import Alignment
from src.models.Blending import Blending
from src.losses.blend_loss import BlendLossBuilder
from src.models.face_parsing.model import seg_mean, seg_std
from src.models.optimizer.ClampOptimizer import ClampOptimizer

args = parse_yaml('opts/config.yml')
device = args.device
st.set_page_config(layout="wide")

formatted_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
os.makedirs("Final_images", exist_ok=True)
n_saved_image = len(os.listdir('Final_images'))


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

@st.cache_data
def cache_encode_image(image):
    result, latent = encoder.encode(np.array(image), return_is_tensor=True)
    return result, latent

@st.cache_data
def cache_embedding(img):
    _, W_init = cache_encode_image(img)
    gen_im, latent_S, latent_F = ii2s.invert_image_in_FS(img, W_init=W_init)
    return gen_im, latent_S, latent_F

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
            img = np.array(Image.open(uploaded_image))
            run_embedding = True
        else:
            img_path = os.path.join(ffhq_dir, selected_filename)
            img = BGR2RGB(img_path)
        
        # Add text input box for 'shape' and 'color' only
        if cap != 'Original':
            text = col.text_input(f"Enter text for {cap}", key=f"{key}_text")
            if text != "":
                run_embedding = True
                # img = text_to_image(text)
        if run_embedding:
            assert("빠른 실험을 위해 모델 로드를 나중에 합니다. 활성화 필요시 연락 바랍니다.")
            _, latent_S, latent_F = cache_embedding(img)
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
stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 50)
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
initial_indices = [0, 5, 20]  # Indices for 'original', 'shape', 'color' respectively

# Display title and separator
st.title("Hair Transfer")

# Display images with selectbox and file uploader
images, selected_filenames, latents, Fs = display_image_with_caption(
    st.columns(3), 
    [filepath_list, filepath_list, filepath_list], 
    ['Original', 'Shape', 'Color'], 
    image_keys,
    initial_indices
)
# 메모리 해제
# del pipe, encoder


# Assign updated images to respective variables
I_ori_rgb, I_shape_rgb, I_color_rgb = images

# # Display the entered texts
# # st.write("Entered texts for Shape and Color:", texts[1:])  # Only display texts for 'Shape' and 'Color'
ffhq_name, original_filename, original_image_path, background_image_path = process_filename(selected_filenames[0], output_dir, ffhq_dir)
I_baldFS_rgb = BGR2RGB(os.path.join(baldFS_dir, original_filename))
I_bg_rgb = BGR2RGB(background_image_path)
I_bg_rgb = cv2.resize(I_bg_rgb, (512, 512))

# # Shape 이미지 처리
shape_name, shape_filename, shape_image_path, background_shape_image_path = process_filename(selected_filenames[1], output_dir, ffhq_dir)
I_baldFS_shape_rgb = BGR2RGB(os.path.join(baldFS_dir, shape_filename))

# # Color 이미지 처리
color_name, color_filename, color_image_path, background_color_image_path = process_filename(selected_filenames[2], output_dir, ffhq_dir)
I_baldFS_color_rgb = BGR2RGB(os.path.join(baldFS_dir, color_filename))


st1, st2, st3 = st.columns(3)
with st1 :
    run_opt = st.button('All decided')
#endregion Get a Mask

st.title("Hair Editing")
btn = st.columns(1)


# ii2s = Embedding(args, net=net)


#region Set a Canvas
can1, can2= st.columns(2)
can1.header('Canvas')
st.markdown('---')
can2.header('Result')


if 'use_I_G_blend_as_background' not in st.session_state:
    st.session_state.use_I_G_blend_as_background = False
if 'use_I_ori_rgb_as_background' not in st.session_state:
    st.session_state.use_I_ori_rgb_as_background = True
if 'canvas_background' not in st.session_state:
    st.session_state.canvas_background = I_bg_rgb
if 'I_G_blend' not in st.session_state:
    st.session_state.I_G_blend = None  # 초기화

# 배경 이미지를 설정하는 함수
def set_background_image():
    if st.session_state.use_I_G_blend_as_background and st.session_state.I_G_blend is not None:
        st.session_state.canvas_background = cv2.resize(st.session_state.I_G_blend, (512, 512))
    else:
        st.session_state.canvas_background = cv2.resize(I_bg_rgb, (512, 512))

# 초기 배경 이미지 설정
set_background_image()

# 캔버스 설정
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



# ## 캔버스를 안그렸으면 더이상 진행 할 필요 없다.
# if canvas_result.json_data is not None:
#     if len(canvas_result.json_data["objects"]) == 0:
#         sys.exit()

# sketch_ = canvas_result.image_data
# if sketch_ is None:
#     sys.exit()
if user_sketch:
    can1.image(
        Image.open("./_temp2/r_result.png")
    )


if canvas_result is None:
    user_mask = None
else:
    user_mask = canvas_result.image_data
    user_mask = user_mask[:,:,:3]
    user_mask = ~np.all(user_mask == [0, 0, 0], axis=-1)
    # kernel_size = 10  # 스트로크 두께를 더 두껍게 하려면 이 값을 증가시키세요.
    st.image((user_mask*255).astype(np.uint8))
    # st.image(sketch_rgb)


sketch_rgb = parse_json(canvas_result.json_data)
# sketch_rgb = ((canvas_result.image_data > 0)*255).astype(np.uint8)
# st.image(sketch_rgb)


SHS = SketchHairSalonModule(args.S2M, args.S2I)
matte_512, baldFS_sketch_rgb_new = SHS.get_matte_and_image(sketch_rgb, background=I_ori_rgb)
del SHS

test1, test2 = st.columns(2)
# test1.image(matte_512)
# test2.image(baldFS_sketch_rgb_new)

if not run_opt:
    sys.exit()

progress_text = "Loading models..."
pbar = st.progress(0, text=progress_text)
net = Net(args)
pbar.progress(33, progress_text)
ii2s = Embedding(args, net=net)
pbar.progress(66, progress_text)
encoder = Encoder(args.e4e, decoder = net.generator)
pbar.progress(100, progress_text)
pbar.empty()
st.markdown("---")

M_ori = get_mask_dict(im=images[0], mask=None, embedding=ii2s)
M_shape = get_mask_dict(im=images[1], mask=None, embedding=ii2s)
M_color = get_mask_dict(im=images[2], mask=None, embedding=ii2s)


#region Make a F7_bald

torch.cuda.empty_cache()


save_dir = os.path.join("Final_images", f"{str(n_saved_image+1).zfill(4)}_{formatted_time}_{ffhq_name}")
os.makedirs(save_dir, exist_ok=True)
args.save_dir = save_dir
args.output_dir =save_dir

st.write(save_dir)
#endregion



def image_transform(I_rgb, I_baldFS_rgb):
    I_ori = ii2s.image_transform(I_rgb).to(device).unsqueeze(0)
    I_baldFS = ii2s.image_transform(I_baldFS_rgb).to(device).unsqueeze(0)
    return I_ori, I_baldFS

I_ori, I_baldFS = image_transform(I_ori_rgb, I_baldFS_rgb)
I_shape, I_baldFS_shape = image_transform(I_shape_rgb, I_baldFS_shape_rgb)
I_color, I_baldFS_color = image_transform(I_color_rgb, I_baldFS_color_rgb)

def load_latent(ffhq_f_dir, name):
    saved_dict = np.load(os.path.join(ffhq_f_dir, f"{name}.npz"))
    latent_W = torch.from_numpy(saved_dict['latent_in']).to(device)
    latent_F7 = torch.from_numpy(saved_dict['latent_F']).to(device)
    return latent_W, latent_F7

W_ori, F7_ori = latents[0], Fs[0]
W_shape, F7_shape = latents[1], Fs[2]
W_color, F7_color = latents[2], Fs[2]

# F7_bald
W_bald_numpy = np.load(os.path.join(bald_dir, f"{ffhq_name}.npy"))
W_bald = torch.from_numpy(W_bald_numpy).to(device)
W_bald_color_numpy = np.load(os.path.join(bald_dir, f"{color_name}.npy"))
W_bald_color = torch.from_numpy(W_bald_color_numpy).to(device)

# Hair Patch W+
# Hair_patch_output_list = ii2s.invert(baldFS_sketch_rgb_new, input=None, epoch=1100, latent_space="W+")
# # st.write(Hair_patch_output_list)
# can3.image(Hair_patch_output_list[-1]["image"])


# 대머리 F7
def bald_blending(W, F7_ori, M, layer_range=(0, 3)):
    with torch.no_grad():
        F7_bald, _ = ii2s.generator([W], input_is_latent=True, return_latents=False,
                                    start_layer=layer_range[0], end_layer=layer_range[1])
        M_32 = M
        F7_bald = F7_bald * M_32 + F7_ori * (1 - M_32)

        G_baldFS, _ = ii2s.generator([W], input_is_latent=True, return_latents=False,
                                    start_layer=layer_range[1] + 1, end_layer=8, layer_in=F7_bald)
    return F7_bald, G_baldFS

F7_bald, G_baldFS = bald_blending(W_bald, F7_ori, M_ori['tensor'][32])
F7_bald_color, G_baldFS_color = bald_blending(W_bald_color, F7_color, M_color['tensor'][32])
#endregion

# Normalize 필수
I_1 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(I_ori_rgb).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)
I_3 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.ToTensor()(Image.fromarray(I_color_rgb).resize((256, 256), Image.LANCZOS))).to(device).unsqueeze(0)

#region Make a F7_blend and HM
align = Alignment(args, embedding=ii2s)

# im1_for_kp = F.interpolate(I_1, size=(256, 256))
im1_for_kp = ((I_1 + 1) / 2).clamp(0, 1) # [0, 1] 사이로
src_kp_hm = align.kp_extractor.face_alignment_net(im1_for_kp)
bald_image_path = os.path.join(baldFS_dir, ffhq_name)
bald_image_path = bald_image_path + '.png'

# # test prior
from src.utils.seg_utils import vis_seg_reverse
from src.utils.data_utils import load_FS_latent
from src.utils.data_utils import load_latent_W
HM_img = BGR2RGB("/home/diglab/workspace/sketch-project/now_images/20240531-1069_240606_193348_00090/new_target_mask.png")
generated_image = torch.from_numpy(vis_seg_reverse(HM_img)).to(device)
warped_latent_2 = load_latent_W("/home/diglab/workspace/sketch-project/now_images/20240531-1069_240606_193348_00090/warped_latent_2.npy").to(device)
generated_image = None
warped_latent_2 = None
# _  , F7_blend_1_2 = load_FS_latent("/home/diglab/workspace/sketch-project/now_images/20240531-0756_240605_104614_65690/65690_00780.npz")
# I_glign_1_2, _ = ii2s.generator([W_ori], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)

from src.utils.seg_utils import vis_seg

# target_mask, hair_mask_target, _, _, warped_latent_2 = align.create_target_segmentation_mask_test(bald_image_path, shape_image_path, 5)
# target = target_mask.squeeze(0).squeeze(0).to(device)
# st.image(vis_seg(target_mask.cpu()))

# bald test
# I_glign_1_2, F7_blend_1_2, HM_1_2 = target.M2H_test(None,warped_latent_2, bald_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir, baldFS_dir, bald_dir,save_dir, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)

# test opt
# I_glign_1_2, F7_blend_1_2, HM_1_2= align.M2H_test(None,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir, None, None,save_dir, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)

I_glign_1_2, F7_blend_1_2, HM_1_2, M_hole, M_hair, M_src,bald_seg_target1, target_mask, warped_latent_2, seg_target2, inpaint_seg, bald_target1 = align.M2H_test(
    generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir,baldFS_dir, bald_dir,save_dir, all_inpainting = True, init_align = False, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False, 
    user_mask=user_mask, user_sketch=user_sketch)

# I_glign_1_2, F7_blend_1_2, HM_1_2, M_hole, M_hair, M_src,bald_seg_target1, target_mask, warped_latent_2, seg_target2, inpaint_seg, bald_target1,new_hair_mask, sketch_target1  = align.M2H_test(
#     generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir,baldFS_dir, bald_dir,save_dir, all_inpainting = True, init_align = False, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False, 
#     user_mask=None, user_sketch=None, latent_sketch=None, sketch_mask=None)
#region visual mask
generated_image = target_mask.to(device)
warped_latent_2 = warped_latent_2

M_hair_rgb_np = M_hair.permute(1, 2, 0).numpy().astype(np.uint8)
M_hole_rgb_np = M_hole.permute(1, 2, 0).numpy().astype(np.uint8)
M_src_rgb_np = M_src.permute(1, 2, 0).numpy().astype(np.uint8)

# 마스크 생성 (M_hole이 0이 아닌 부분)
mask_hole = (M_hole.permute(1, 2, 0).numpy() != 0)

combined_image = M_src_rgb_np.copy()
combined_image[mask_hole[:, :, 0]] = [0, 0, 0]
combined_image[mask_hole] = M_hole_rgb_np[mask_hole]

# M_hair에 M_hole의 픽셀을 덮어쓰기
combined_image1 = M_hair_rgb_np.copy()
combined_image1[mask_hole[:, :, 0]] = [0, 0, 0]
combined_image1[mask_hole] = M_hole_rgb_np[mask_hole]

ti1, ti2, ti3, ti4,ti5= st.columns(5)
seg_mask = ii2s.get_seg(original_image_path, target=None)
seg_target1 = torch.argmax(seg_mask, dim=1).long()
seg_target1 = seg_target1[0].byte().cpu().detach()
seg1_hairmask = (seg_target1 == 10) * 1.0
ti1.image(vis_seg(seg_target1.cpu()),  caption ="source seg mask")

# seg_mask = ii2s.get_seg(shape_image_path, target=None)
# seg_target2 = torch.argmax(seg_mask, dim=1).long()
# seg_target2 = seg_target2[0].byte().cpu().detach()
# seg2_hairmask = (seg_target2 == 10) * 1.0
ti2.image(vis_seg(seg_target2.cpu()),  caption ="align seg mask")
ti4.image(M_hole_rgb_np)
ti3.image(M_src_rgb_np)

# ti4.image(vis_seg(masked_bald_down_seg.cpu()), caption ="inpainting region")
ti5.image(combined_image1, caption ="combine region")
ti5.image(combined_image, caption ="combine region")


t1, t2, t3 = st.columns(3)
t1.image(vis_seg(bald_seg_target1.cpu()),  caption ="bald seg mask")
t2.image(vis_seg(inpaint_seg), caption ="inpainting region")
t3.image(vis_seg(target_mask), caption ="final seg mask")

# target_mask = BGR2RGB("/home/diglab/workspace/sketch-project/now_images/20240531-1069_240606_193348_00090/target_mask.png")
# t2.image(target_mask, caption ="inpainting region w/o bald mask")

# t3.image(vis_seg(new_target_mask.cpu()), caption ="inpainting region w/ bald mask")
#endregion

####### test #######
# st.header("All inapting region update")
tt1, tt2, tt3 = st.columns(3)
tt1.image(ii2s.tensor_to_numpy(I_glign_1_2),  caption ="Init : W_align")
tt1.image(vis_seg(HM_1_2.cpu()), caption ="optimized mask")
# I_glign_1_2, _, HM_1_2, _, _, _,_, _, _, _, _, _ = align.M2H_test(
#     generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir,baldFS_dir, bald_dir,save_dir, all_inpainting = True, init_align = False, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False, 
#     user_mask=user_mask, user_sketch=user_sketch)
# tt2.image(ii2s.tensor_to_numpy(I_glign_1_2),  caption ="Init : W_src")
# tt2.image(vis_seg(HM_1_2.cpu()), caption ="optimized mask")
# I_glign_1_2, _, HM_1_2, _, _, _,_, _, _, _, _, _ = align.M2H_test(
#     generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, bald_dir,baldFS_dir, bald_dir,save_dir, all_inpainting = True, init_align = False, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False, 
#     user_mask=user_mask, user_sketch=user_sketch)
# tt3.image(ii2s.tensor_to_numpy(I_glign_1_2),  caption ="Init : W_bald")
# tt3.image(vis_seg(HM_1_2.cpu()), caption ="optimized mask")

# st.header("Only inapting region update")
# ttt1, ttt2, ttt3 = st.columns(3)
# I_glign_1_2, _ , HM_1_2, _, _, _,_, _, _, _, _ = align.M2H_test(generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir,baldFS_dir, bald_dir,save_dir,all_inpainting = False, init_align = False, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)
# ttt1.image(ii2s.tensor_to_numpy(I_glign_1_2),  caption ="Init : W_src")
# ttt1.image(vis_seg(HM_1_2.cpu()), caption ="optimized mask")
# I_glign_1_2, _ , HM_1_2, _, _, _,_, _, _, _, _ = align.M2H_test(generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, ffhq_wp_dir,baldFS_dir, bald_dir,save_dir,all_inpainting = False, init_align = True, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)
# ttt2.image(ii2s.tensor_to_numpy(I_glign_1_2),  caption ="Init : W_align")
# ttt2.image(vis_seg(HM_1_2.cpu()), caption ="optimized mask")
# I_glign_1_2, _ , HM_1_2, _, _, _,_, _, _, _, _ = align.M2H_test(generated_image,warped_latent_2, original_image_path, shape_image_path, ffhq_f_dir, bald_dir,baldFS_dir, bald_dir,save_dir,all_inpainting = False, init_align = False, sign=args.sign, align_more_region=False, smooth=args.smooth, save_intermediate=False)
# ttt3.image(ii2s.tensor_to_numpy(I_glign_1_2),  caption ="Init : W_bald")
# ttt3.image(vis_seg(HM_1_2.cpu()), caption ="optimized mask")


I_1_bald = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(I_baldFS)
I_1_bald = F.interpolate(I_1_bald, size=(256, 256), mode='nearest').squeeze()
I_1_2 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(I_glign_1_2)
I_1_2 = F.interpolate(I_glign_1_2, size=(256, 256), mode='nearest').squeeze()
I_3_lab = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(rgb_to_lab(I_3))

# mask
HM_1D, _ = align.dilate_erosion(M_ori['tensor'][1024], device)
HM_3D, HM_3E = align.dilate_erosion(M_color['tensor'][1024], device)
HM_1_2 = HM_1_2.unsqueeze(0).unsqueeze(0).cpu()

target_hairmask = (HM_1_2 == 10) * 1.0
target_hairmask = target_hairmask.float()
HM_1_2D, HM_1_2E = align.dilate_erosion(target_hairmask, device)

#endregion

r5col1, r5col2, r5col3 = st.columns(3)

#region Make a S_blend
blend = Blending(args, embedding = ii2s)
loss_builder = BlendLossBuilder(args)

downsampled_hair_mask = F.interpolate(HM_1_2E, size=(256, 256), mode='bilinear', align_corners=False)
upsampled_hair_mask = F.interpolate(HM_1_2E, size=(1024, 1024), mode='bilinear', align_corners=False)
# 원래
# interpolation_latent = torch.zeros((18, 512), requires_grad=True, device=device)
# 평균시작
# for i in range(4):
#     st.header("W_color시작")
num_temp_layer = 18
interpolation_latent = ((W_ori + W_color) / 2).detach().clone().requires_grad_(True)
optimizable_part = interpolation_latent[:, 18-num_temp_layer:18, :].detach().clone().requires_grad_(True)

# w_ori 시작
# interpolation_latent = W_ori.detach().clone().requires_grad_(True)
# w_color 시작
# interpolation_latent = W_color.detach().clone().requires_grad_(True)
# opt_blend = ClampOptimizer(torch.optim.Adam, [interpolation_latent], lr=ii2s.opts.learning_rate)
opt_blend = torch.optim.Adam([interpolation_latent], lr=ii2s.opts.learning_rate)
with torch.no_grad():
    I_X, _ = ii2s.generator([W_ori], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
    I_X_0_1 = (I_X + 1) / 2
    IM = (align.downsample(I_X_0_1) - seg_mean) / seg_std
    down_seg, _, _ = ii2s.seg(IM)
    current_mask = torch.argmax(down_seg, dim=1).long().cpu().float()
    HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
    HM_X = F.interpolate(HM_X.unsqueeze(0), size=(256, 256), mode='nearest').squeeze()
    HM_XD, _ = align.cuda_unsqueeze(align.dilate_erosion_mask_tensor(HM_X), device)
    target_mask = (1 - HM_1D) * (1 - HM_1_2D) * (1 - HM_XD)

pbar = tqdm.tqdm(range(150), desc='Blend', leave=False)
for step in pbar:
    opt_blend.zero_grad()
    # latent_mixed = W_ori + interpolation_latent.unsqueeze(0) * (W_color - W_ori)
    # interpolation_latent[:, 18-num_temp_layer:18, :] = optimizable_part
    latent_mixed = interpolation_latent
    I_G, _ = ii2s.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_color)
    I_G_1_2, _ = ii2s.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
    
    G_lab_ori = rgb_to_lab(I_G)
    G_lab = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(rgb_to_lab(I_G_1_2))

    im_dict = {
        'gen_im': blend.downsample_256(I_G),
        'im_1': I_1,
        'im_3': I_3,
        'mask_face': target_mask,
        'mask_hair': HM_3E,
        'mask_2_hair': downsampled_hair_mask,
    }

    total_loss = 0
    face_loss = loss_builder._loss_face_percept(blend.downsample_256(I_G_1_2), im_dict['im_1'], im_dict['mask_face'])
    hair_loss = loss_builder._loss_hair_percept(im_dict['gen_im'], im_dict['im_3'], im_dict['mask_hair'])
    # hair_lab_loss = loss_builder._loss_hair_percept(blend.downsample_256(G_lab_ori), I_3_lab, im_dict['mask_hair'])

    H1_region = blend.downsample_256(I_G_1_2) * im_dict['mask_2_hair']
    H2_region = im_dict['im_3'] * im_dict['mask_hair']
    style_loss = loss_builder.style_loss(H2_region, H1_region, im_dict['mask_hair'], im_dict['mask_2_hair'])
    
    # H1_region_lab = blend.downsample_256(G_lab) * im_dict['mask_2_hair']
    # H2_region_lab = I_3_lab * im_dict['mask_hair']
    # style_lab_loss = loss_builder.style_loss(H2_region_lab, H1_region_lab, im_dict['mask_hair'], im_dict['mask_2_hair'])
    
    total_loss += face_loss+ hair_loss + 1000*style_loss
    opt_blend.zero_grad()
    total_loss.backward(retain_graph=True)
    opt_blend.step()

I_G_blend1, _ = ii2s.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
# can2.image(ii2s.tensor_to_numpy(I_G))
can2.image(ii2s.tensor_to_numpy(I_G_blend1))

result_image_path = os.path.join(save_dir, "result_image.png")
Image.fromarray(ii2s.tensor_to_numpy(I_G_blend1)).save(result_image_path)
np.save(os.path.join(save_dir,'result_w'), latent_mixed.detach().cpu().numpy())
np.savez(os.path.join(save_dir,'result_FS.npz') , latent_in=latent_mixed.detach().cpu().numpy(), latent_F=F7_blend_1_2.detach().cpu().numpy())


del align, blend, loss_builder, ii2s


sys.exit()

interpolation_latent_2 = torch.zeros((18, 512), requires_grad=True, device=device)
opt_blend_2 = torch.optim.Adam([interpolation_latent_2], lr=ii2s.opts.learning_rate)
pbar = tqdm.tqdm(range(400), desc='Blend_2', leave=False)
for step in pbar:
    opt_blend_2.zero_grad()
    latent_mixed_new = W_ori + interpolation_latent_2.unsqueeze(0) * (latent_mixed - W_ori)
    I_G_blend, _ = ii2s.generator([latent_mixed_new], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
    
    im_dict = {
        'gen_im': blend.downsample_256(I_G_blend),
        'im_1': I_1,
        'im_3': blend.downsample_256(I_G_blend1),
        'mask_face': target_mask,
        'mask_hair': HM_1_2E,
        'mask_2_hair': downsampled_hair_mask,
    }

    total_loss = 0
    face_loss = loss_builder._loss_face_percept(im_dict['gen_im'], im_dict['im_1'], im_dict['mask_face'])
    hair_loss = loss_builder._loss_hair_percept(im_dict['gen_im'], im_dict['im_3'], im_dict['mask_hair'])
    total_loss += face_loss + hair_loss
    opt_blend_2.zero_grad()
    total_loss.backward(retain_graph=True)
    opt_blend_2.step()

can2.image(ii2s.tensor_to_numpy(I_G_blend))

I_G_shape_wo, _ = ii2s.generator([W_color], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_2)
# I_G_wo, _ = ii2s.generator([W_color], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F7_blend_1_3)

#endregion
st.session_state.canvas_background = ii2s.tensor_to_numpy(I_G_blend) 
st.session_state.I_G_blend = ii2s.tensor_to_numpy(I_G_blend)  # I_G_blend를 세션 상태에 저장
st.session_state.use_I_G_blend_as_background = True
set_background_image()


#region Results and Save

result_image_path = os.path.join(save_dir, "result_image.png")
full_image_path = os.path.join(save_dir, original_filename)

Image.fromarray(ii2s.tensor_to_numpy(I_G_blend)).save(result_image_path)

# Make sub image 
large_width = 1024
small_width = int(large_width / 3)
small_size = (small_width, small_width)
large_size = (large_width, large_width)

I_ori_rgb_small = Image.fromarray(cv2.resize(I_ori_rgb, small_size))
I_shape_rgb_small = Image.fromarray(cv2.resize(I_shape_rgb, small_size))
I_color_rgb_small = Image.fromarray(cv2.resize(I_color_rgb, small_size))
small_images = [I_ori_rgb_small, I_shape_rgb_small, I_color_rgb_small]

I_G_rgb_large = F.interpolate(I_G_blend, size=large_size, mode='bilinear', align_corners=False)
I_G_rgb_large_pil = Image.fromarray(ii2s.tensor_to_numpy(I_G_rgb_large))

new_image = Image.new('RGB', (small_width + large_width, large_width))
for i, img in enumerate(small_images):
    new_image.paste(img, (0, i * small_width, small_width, (i + 1) * small_width))
new_image.paste(I_G_rgb_large_pil, (small_width, 0, small_width + large_width, large_width))
r5 = st.columns(1)
r5[0].image(new_image)
new_image.save(full_image_path)

r5_2 = st.columns(1)
r5_2[0].text(f"face percep: {face_loss.item()} | hair percep: {hair_loss.item()}")
r5_2[0].text(f"mask style: {style_loss.item()} | mask style lab: {style_lab_loss.item()}")
r5_2[0].text(f"hair lab percep: {hair_lab_loss.item()}")
r5_2[0].text(f"Total Loss: {total_loss.item()}")

del ii2s
print('-----')
# 블렌딩이 끝난 후 배경 이미지를 선택하는 라디오 버튼 생성
# if 'background_option' not in st.session_state:
#     st.session_state.background_option = "Use blended image as canvas background"

# background_option = btn[0].radio(
#     "Select canvas background",
#     ("Use Original image as canvas background", "Use blended image as canvas background"),
#     index=1 if st.session_state.use_I_G_blend_as_background else 0
# )


#endregion
