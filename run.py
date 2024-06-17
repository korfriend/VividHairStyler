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

args = parse_yaml('opts/config.yml')
device = args.device
st.set_page_config(layout="wide")

formatted_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
os.makedirs(args.output_dir, exist_ok=True)
n_saved_image = len(os.listdir(args.output_dir))


filepath_list = sorted(os.listdir(os.path.join(args.data_dir, "ffhq")))


# @st.cache_data
def setup_data(ffhq, uploaded_image):
    if uploaded_image:
        raise "아직입니다."
            # TODO: 
            # add face parsing
            # add invert module
            # img = uploaded_image
            # image_placeholder.image(uploaded_image)
    else:
        # found = 
        img = cv2.imread(os.path.join(args.data_dir, "ffhq", ffhq))# find("ffhq", ffhq, root=args.data_dir)['png']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # st.markdown(os.path.join(args.data_dir, "W+", f"{ffhq[:5]}.npy"))
        W = np.load(os.path.join(args.data_dir, "W+", f"{ffhq[:5]}.npy"))# find("W+", ffhq, root=args.data_dir)['latent']
        found = np.load(os.path.join(args.data_dir, "FS", f"{ffhq[:5]}.npz"))# find("FS", ffhq, root=args.data_dir)
        FS = {
            'latent_in': torch.from_numpy(found['latent_in']).to(device),
            'latent_F': torch.from_numpy(found['latent_F']).to(device),
        }
    return img, FS, W

ffhqs = []
images = []
Ws = []
FSs = []
idxs = [20, 50, 70]
for idx, st_, cap in zip(
        range(3), 
        st.columns(3), 
        ['original', 'shape', 'color']
    ):
    st_.header(f"{cap}")
    selected_image = st_.selectbox(f"Select from test", filepath_list, index = idxs[idx], key=f"{idx}_select")
    uploaded_image = st_.file_uploader(f"Upload", type=["png", "jpg", "jpeg"], key=f"{idx}_upload")
    
    img, FS, W = setup_data(selected_image, uploaded_image)
    st_.image(img)

    ffhqs.append(os.path.basename(selected_image))
    images.append(img)
    Ws.append(W)
    FSs.append(FS)


progress_text = "Loading models..."
pbar = st.progress(0, text=progress_text)
net = Net(args)
pbar.progress(33, progress_text)
ii2s = Embedding(args, net=net)
pbar.progress(66, progress_text)
# encoder = Encoder(args.e4e, decoder = net.generator)
align1 = Alignment(args, embedding=ii2s)
align2 = Alignment2(args, embedding=ii2s)
pbar.progress(100, progress_text)
pbar.empty()
st.markdown("---")

can1, can2 = st.columns(2)
with can1:
    canvas_mask_result = st_canvas(
        stroke_width=75,
        stroke_color="#000",
        background_color="#eee",
        background_image=Image.fromarray(st.session_state.canvas_background),
        update_streamlit=True,
        width=st.session_state.canvas_background.shape[1],
        height=st.session_state.canvas_background.shape[0],
        drawing_mode='freedraw',
        point_display_radius=0,
        key="canvas1",
    )
can1.image(canvas_mask_result.image_data)


st1, st2 = st.columns(2)
# I_glign_1_2, F7_blend_1_2, HM_1_2, M_hole, \
# M_hair, M_src,bald_seg_target1, target_mask, \
# warped_latent_2, seg_target2, inpaint_seg, bald_target1 = align1.M2H_test(
#     None, 
#     Ws[1], 
#     os.path.join(args.data_dir, 'ffhq', ffhqs[0]), # find('ffhq', ffhqs[0], root=args.data_dir)['png_path'], 
#     os.path.join(args.data_dir, 'ffhq', ffhqs[1]), # find('ffhq', ffhqs[1], root=args.data_dir)['png_path'], 
#     os.path.join(args.data_dir, "FS"), 
#     os.path.join(args.data_dir, "W+"), 
#     os.path.join(args.data_dir, "bladFS"), 
#     os.path.join(args.data_dir, "bald"), 
#     args.save_dir, 
#     all_inpainting = True, 
#     init_align = False, 
#     sign=args.sign, 
#     # align_more_region=False, 
#     smooth=args.smooth, 
#     # save_intermediate=False, 
#     user_mask=None, 
#     user_sketch=False
# )


# I_glign_1_2, F7_blend_1_2, HM_1_2 = align2.M2H_test(
#     None, 
#     img_path1 = images[0], 
#     img_path2 = images[1], 
#     save_dir = args.save_dir, 
#     latent_FS_path_1 = FSs[0], 
#     latent_FS_path_2 = FSs[1], 
#     latent_W_path_1 = Ws[0], 
#     latent_W_path_2 = Ws[1], 
#     sign=args.sign, 
#     align_more_region=False, 
#     smooth=args.smooth, 
#     save_intermediate=False, 
#     user_mask = None, 
#     user_sketch = False,
#     pbar = pbar
# )
# st2.image(ii2s.tensor_to_numpy(I_glign_1_2))

del align2
del ii2s
del net