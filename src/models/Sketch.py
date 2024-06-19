import numpy as np
import cv2
import torch
from scipy.spatial import KDTree
import tqdm

import sys

from .models2.Unet_At_Bg import UnetAtGenerator, UnetAtBgGenerator
import torchvision.transforms.functional as tf


class Sk2Matte:
    def __init__(self,load_path="./checkpoints/S2M/200_net_G.pth"):
        self.model = UnetAtGenerator(1,1,8,64,use_dropout=True)
        self.device = torch.device('cuda:0')
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        model_name = '/'.join(load_path.split('/')[2:])
        # print("Model Sk2Matte (%s) is loaded."%model_name)
    
    def getResult(self,inputs,img=None):
        inputs_tensor = tf.to_tensor(inputs[:,:,np.newaxis])*2.0-1.0
        inputs_tensor = inputs_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            result_tensor = self.model(inputs_tensor)
            result = ((result_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")[...,0]
            # result = util.tensor2im(result_tensor)
            result[result>250] = 255

        return result

class Sk2Image:
    def __init__(self,load_path="./checkpoints/S2I_unbraid/200_net_G.pth"):
        self.model = UnetAtBgGenerator(3,3,8,64,use_dropout=True)
        self.device = torch.device('cuda:0')
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        model_name = '/'.join(load_path.split('/')[2:])
        # print("Model Sk2Image (%s) is loaded."%model_name)
    
    def getResult(self,inputs,img,matte, thred=1):
            
        h,w = img.shape[:2]
        noise = self.generate_noise(w,h)

        N = tf.to_tensor(noise)*2.0-1.0
        N = N.unsqueeze(0).to(self.device)

        img_tensor = tf.to_tensor(img)*2.0-1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        M = tf.to_tensor(matte).unsqueeze(0).to(self.device)

        inputs = tf.to_tensor(inputs)*2.0-1.0
        inputs = inputs.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            # result_tensor, hair_matte_tensor = self.model(inputs,img_tensor, M,N, thred=thred)
            result_tensor = self.model(inputs,img_tensor, M,N)
            result = ((result_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")
            # hair_matte = ((hair_matte_tensor[0]+1)/2*255).cpu().numpy().transpose(1,2,0).astype("uint8")
        return result#, hair_matte
    
    def generate_noise(self, width, height):
        weight = 1.0
        weightSum = 0.0
        noise = np.zeros((height, width, 3)).astype(np.float32)
        while width >= 8 and height >= 8:
            noise += cv2.resize(np.random.normal(loc = 0.5, scale = 0.25, size = (int(height), int(width), 3)), dsize = (noise.shape[0], noise.shape[1])) * weight
            weightSum += weight
            width //= 2
            height //= 2
        return noise / weightSum

class SketchHairSalonModule():
    def __init__(
            self, 
            S2M_path ="./checkpoints/S2M/200_net_G.pth", 
            S2I_path = "./checkpoints/S2I_unbraid/200_net_G.pth"):
        # self.args = args
        self.S2I = Sk2Image(S2I_path)
        self.S2M = Sk2Matte(S2M_path)

    def get_matte(self, im, input_is_gray=True):
        self.sketch1_gray =  im
        if not input_is_gray:
            self.sketch_mask = ~np.all(im == [0, 0, 0], axis=-1)
            sketch1_gray = np.full((512, 512), 127, dtype=np.uint8)
            self.sketch1_gray = np.where(self.sketch_mask, 255, sketch1_gray)
        self.matte =  self.S2M.getResult(self.sketch1_gray)
        return self.matte

    def get_image(self, sketch2_rgb, original, matte):
        if sketch2_rgb.shape[1] != 512:
            sketch2_rgb = cv2.resize(sketch2_rgb, (512, 512))
        if original.shape[1] != 512:
            original = cv2.resize(original, (512, 512))
        if len(matte.shape) == 2:
            matte = cv2.cvtColor(matte, cv2.COLOR_GRAY2BGR)
        # sketch2_rgb = cv2.cvtColor(self.matte_512, cv2.COLOR_GRAY2RGB)
        self.result = self.S2I.getResult(sketch2_rgb, original, matte)
        return self.result

    def get_matte_and_image(self, sketch, background):
        sketch = cv2.resize(sketch, (512, 512))
        sketch_mask = ~np.all(sketch == [0, 0, 0], axis=-1)
        sketch1_gray = np.full((512, 512), 127, dtype=np.uint8)
        sketch1_gray = np.where(sketch_mask, 255, sketch1_gray)
        matte = self.get_matte(sketch1_gray, input_is_gray=True)
        
        # new_sketch_rgb = cv2.resize(sketch, (512, 512))
        sketch2m_rgb = np.array(cv2.cvtColor(matte, cv2.COLOR_GRAY2RGB))
        sketch2m_rgb[sketch_mask] = sketch[sketch_mask]

        result = self.get_image(sketch2m_rgb, background, matte)
        return matte, result
    # def S2I(self, )


# S2M = Sk2Matte()
# S2I = Sk2Image()

