import os
import sys
import importlib
import torch
from torchvision import transforms
from PIL import Image
from CycleGAN.options.test_options import TestOptions
from CycleGAN.models.base_model import BaseModel

# 경로 추가
def add_relative_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
    if absolute_path not in sys.path:
        sys.path.append(absolute_path)

add_relative_path('../e0u0n-workspace/HairStyler/CycleGAN')
add_relative_path('../e0u0n-workspace/HairStyler/CycleGAN/models')

# find_model_using_name 및 create_model 정의
def find_model_using_name(model_name):
    model_filename = "CycleGAN.models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(f"In {model_filename}.py, there should be a subclass of BaseModel with class name that matches {target_model_name} in lowercase.")
        exit(0)

    return model

def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model [{type(instance).__name__}] was created")
    return instance

# 이미지 로드 및 저장 함수
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    image = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(image)
    image.save(path)

def load_single_image(image_path, transform):
    try:
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = transform(img)
            return img
    except Exception as e:
        print(f"Could not load image {image_path}: {e}")
        return None

# 모델 테스트 함수
def test_model_with_single_image(img_path, opt):
    opt.isTrain = False

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    # Load single image
    img = load_single_image(img_path, transform)
    if img is None:
        return None

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    # Process the image
    data = {'A': img.unsqueeze(0), 'B': img.unsqueeze(0), 'A_paths': img_path, 'B_paths': img_path}  # Ensure both A and B are set
    model.set_input(data)  # Create a fake data dictionary
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image results
    print(f'processing {img_path}')

    return visuals

# CustomTestOptions 클래스
class CustomTestOptions:
    def __init__(self):
        self.name = 'new_hair2sketch'
        self.gpu_ids = [0]
        self.checkpoints_dir = './CycleGAN/checkpoints'
        self.model = 'cycle_gan'
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.norm = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = True
        self.dataset_mode = 'single'
        self.direction = 'AtoB'
        self.serial_batches = True
        self.num_threads = 4
        self.batch_size = 1
        self.load_size = 512
        self.crop_size = 512
        self.max_dataset_size = float("inf")
        self.preprocess = 'resize_and_crop'
        self.no_flip = True
        self.display_winsize = 512
        self.epoch = 'latest'
        self.load_iter = 0
        self.verbose = False
        self.suffix = ''
        self.use_wandb = False
        self.wandb_project_name = 'CycleGAN-and-pix2pix'
        self.img_path_manual = 'database/ffhq/00090.png'
        self.results_dir = './results/'
        self.aspect_ratio = 1.0
        self.phase = 'test'
        self.eval = True
        self.num_test = 50
