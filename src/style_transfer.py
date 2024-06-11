import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
import cv2
import os

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from datetime import datetime

import copy



class StyleTransfer:

    def __init__(self, opts, net=None):
        super(StyleTransfer, self).__init__()
        self.opts = opts
        self.device = opts.device
        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define normalization mean and std
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


    @staticmethod
    class ContentLoss(nn.Module):
        def __init__(self, target):
            super(StyleTransfer.ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleTransfer.StyleLoss, self).__init__()
            self.target = self.gram_matrix(target_feature).detach()

        def gram_matrix(self, input):
            a, b, c, d = input.size()
            features = input.view(a * b, c * d)
            G = torch.mm(features, features.t())
            return G.div(a * b * c * d)

        def forward(self, input):
            G = self.gram_matrix(input)
            self.loss = nn.functional.mse_loss(G, self.target)
            return input

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(StyleTransfer.Normalization, self).__init__()
            self.mean = mean.view(-1, 1, 1)
            self.std = std.view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std
        
    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers):
        
        # normalization module
        normalization = StyleTransfer.Normalization(normalization_mean, normalization_std)
        content_layers = self.content_layers_default
        style_layers = self.style_layers_default
        style_img = style_img.to(self.device)  
        
        # just in order to have an iterable access to or list of content/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = StyleTransfer.ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleTransfer.StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], StyleTransfer.ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses
    
    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer
    
    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        model.eval()
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    output_img = input_img.detach().cpu().squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255
                    Image.fromarray(output_img.astype('uint8')).save(os.path.join(OUTPUT_DIR_PATH, '%s.png' % run[0]))

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img