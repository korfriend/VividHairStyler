import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Barbershop.losses import masked_lpips
from Barbershop.losses.style.style_loss import StyleLoss
from Barbershop.losses.style.custom_loss import prepare_mask
from Barbershop.losses.style.vgg_activations import VGG16_Activations


class BlendLossBuilder(nn.Module):
    def __init__(self, opt):
        super(BlendLossBuilder, self).__init__()

        self.opt = opt
        self.use_gpu = opt.device == 'cuda'

        self.masked_mse_loss_module = MaskedMSELoss().to(opt.device)

        # Perceptual Losses for face and hair
        self.face_percept = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=self.use_gpu
        )
        self.face_percept.eval()

        self.hair_percept = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=self.use_gpu
        )
        self.hair_percept.eval()

        # Style loss setup
        self.style_loss_module = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(opt.device)
        self.style_loss_module.eval()

        self.content_loss_module = ContentLoss(VGG16_ACTIVATIONS_LIST=[21], normalize=False).to(opt.device)
        self.content_loss_module.eval()

    def mse_loss(self, ref_im, gen_im, mask, device) : 
        return self.masked_mse_loss_module(ref_im, gen_im, mask, device)

    def style_loss(self, im1, im2, mask1, mask2):
        return self.style_loss_module(im1 * mask1, im2 * mask2, mask1=mask1, mask2=mask2)

    def content_loss(self, im1, im2, mask):
        return self.content_loss_module(im1, im2, mask)

    def _loss_face_percept(self, gen_im, ref_im, mask, **kwargs):
        return self.face_percept(gen_im, ref_im, mask=mask)

    def _loss_hair_percept(self, gen_im, ref_im, mask, **kwargs):
        return self.hair_percept(gen_im, ref_im, mask=mask)

    def compute_losses(self, gen_im, im_1, im_3, mask_face, mask_hair):
        losses = {}
        total_loss = 0

        # Face and Hair Perceptual Loss
        face_loss = self._loss_face_percept(gen_im, im_1, mask_face)
        hair_loss = self._loss_hair_percept(gen_im, im_3, mask_hair)
        losses['face'] = face_loss
        losses['hair'] = hair_loss
        total_loss += face_loss + hair_loss

        # Additional Style Loss
        style_loss = self.style_loss(im_1, im_3, mask_face, mask_hair)
        losses['style'] = style_loss
        total_loss += style_loss

        # Content Loss Calculation
        content_loss = self.content_loss(gen_im, im_1, mask_hair)  # Assuming the mask is not needed or integrated within the module
        losses['content'] = content_loss
        total_loss += content_loss

        return total_loss, losses

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask, device):
        # Ensure the mask is on the correct device and is the correct data type
        if mask is not None:
            mask = mask.to(dtype=input.dtype, device=device)

        # Apply the mask efficiently
        if mask is not None:
            # Assume mask is [B, 1, H, W] and input is [B, C, H, W]
            # This avoids expanding mask explicitly and uses broadcasting
            masked_input = input * mask
            masked_target = target * mask
        else:
            masked_input = input
            masked_target = target

        # Compute the MSE loss on the masked regions
        # Instead of squaring before mean, use mean of squares minus square of means for numerical stability
        diff = masked_input - masked_target
        mse = (diff * diff).mean()

        return mse


class ContentLoss(nn.Module):
    def __init__(self, VGG16_ACTIVATIONS_LIST=[21], normalize=False):
        super(ContentLoss, self).__init__()
        self.vgg16_act = VGG16_Activations(VGG16_ACTIVATIONS_LIST)
        self.vgg16_act.eval()
        self.normalize = normalize

    def get_features(self, model, x):
        return model(x)

    def mask_features(self, x, mask):
        mask = prepare_mask(x, mask)
        return x * mask

    def forward(self, x, x_hat, mask=None):
        x = x.cuda()
        x_hat = x_hat.cuda()

        # resize images to 256px resolution
        N, C, H, W = x.shape
        upsample2d = nn.Upsample(
            scale_factor=256 / H, mode="bilinear", align_corners=True
        )

        x = upsample2d(x)
        x_hat = upsample2d(x_hat)

        # Get features from the model for x and x_hat
        with torch.no_grad():
            act_x = self.get_features(self.vgg16_act, x)
        for layer in range(0, len(act_x)):
            act_x[layer].detach_()

        act_x_hat = self.get_features(self.vgg16_act, x_hat)

        # Only use a specific layer for content loss, typically a deeper layer
        if mask is not None:
            feat_x = self.mask_features(act_x[0], mask)
            feat_x_hat = self.mask_features(act_x_hat[0], mask)
        else:
            feat_x = act_x[0]
            feat_x_hat = act_x_hat[0]

        # Calculate content loss using MSE
        content_loss = F.mse_loss(feat_x, feat_x_hat)

        return content_loss
