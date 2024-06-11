import sys
sys.path.append('./HairMapper/encoder4editing/')

import torch
import numpy as np
from argparse import Namespace
from PIL import Image
from models.psp import pSp
import torchvision.transforms as transforms

class Encoder():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    resize_dims = (256, 256)

    def __init__(
        self,
        model_path = "./HairMapper/ckpts/e4e_ffhq_encode.pt", 
        decoder = None
    ) -> None:
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        # pprint.pprint(opts)  # Display full options used
        # update the training options
        opts['checkpoint_path'] = model_path
        self.opts= Namespace(**opts)
        self.net = pSp(self.opts, decoder)
        self.net.eval()
        self.net.cuda()
        print('Model successfully loaded!')
        del ckpt, opts

    def run_on_batch(
            self, 
            inputs: torch.Tensor, 
            net,
        ):
        images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True,resize=False)
        return images, latents
    
    def encode(self, path, input_is_array=False, return_is_tensor=False):
        # image = path
        if isinstance(path, str):
            image = Image.open(path)
        elif isinstance(path, np.ndarray):
            image = Image.fromarray(path)
        elif isinstance(path, (Image)):
            image = path
        else:
            raise(f"input must be file path or numpy image, but got {type(path)}")
            
        transformed_image = self.transform(image)
        with torch.no_grad():
            images, latents = self.run_on_batch(transformed_image.unsqueeze(0), self.net)
        latent = latents.detach().cpu().numpy()
        image = images.detach().cpu().squeeze().transpose(0, 2).transpose(0, 1).numpy()
        image = ((image +1)/2)
        image[image < 0] = 0
        image[image > 1] = 1
        image = (image*255).astype(np.uint8)
        if return_is_tensor:
            return image, latents.detach().clone()
        return image, latent
    
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoder')
    parser.add_argument('--im_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    encoder = Encoder()
    image, latent = encoder.encode(args.im_path)
    os.makedirs(args.output_dir, exist_ok=True)
    image = Image.fromarray(image)
    image.save(os.path.join(args.output_dir, "result.png"))
    np.save(os.path.join(args.output_dir, "result.png"), latent)
    print("done!!")