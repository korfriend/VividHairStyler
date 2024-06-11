import importlib
import os
import sys
import torch
from torchvision import transforms
from PIL import Image

# 필요한 상대 경로 추가
def add_relative_path(relative_path):
    """현재 파일의 경로를 기준으로 상대 경로를 절대 경로로 변환하여 sys.path에 추가"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
    if absolute_path not in sys.path:
        sys.path.append(absolute_path)

add_relative_path('../e0u0n-workspace/HairStyler/CycleGAN')
add_relative_path('../e0u0n-workspace/HairStyler/CycleGAN/models')

from options.test_options import TestOptions
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

class BaseModel:
    def __init__(self, opt):
        self.opt = opt

def find_model_using_name(model_name):
    """Import the module 'models/[model_name]_model.py'."""
    model_filename = "models." + model_name + "_model"
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

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print(f"model [{type(instance).__name__}] was created")
    return instance

def test_model_with_single_image(image_path, opt):
    """Function to test the model with a single image and return the visuals."""
    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to the input size expected by the model
        transforms.ToTensor()  # Convert image to a tensor
    ])

    # Load and transform the image
    img = Image.open(image_path).convert('RGB')
    img = transform(img)

    # Create a model instance and set it up with the options
    model = create_model(opt)
    model.setup(opt)
    
    # If eval mode is on, switch the model to evaluation mode
    if opt.eval:
        model.eval()

    # Run inference on the image
    model.set_input({'A': img.unsqueeze(0), 'A_paths': image_path})
    model.test()
    
    # Get the visuals from the model
    visuals = model.get_current_visuals()

    # Return the visuals
    return visuals

class CustomTestOptions:
    def __init__(self):
        self.name = 'test_model'
        self.eval = True
        self.model = 'cycle_gan'
        self.gpu_ids = [0]
        # 필요한 다른 옵션들 추가

'/home/diglab/workspace/sketch-project/checkpoints/latest_net_G.pth'  # 모델 경로 추가
        