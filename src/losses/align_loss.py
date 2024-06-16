import torch
from src.losses.style.style_loss import StyleLoss
from src.losses import masked_lpips

class AlignLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(AlignLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep']]
        if opt.device == 'cuda':
            self.use_gpu = True
        else:
            self.use_gpu = False

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(opt.device)
        self.style.eval()


        tmp = torch.zeros(16).to(opt.device)
        tmp[0] = 1
        self.cross_entropy_wo_background = torch.nn.CrossEntropyLoss(weight=1 - tmp)
        self.cross_entropy_only_background = torch.nn.CrossEntropyLoss(weight=tmp)

        weight_tmp = torch.zeros(16).cuda()
        weight_tmp[10] = 1
        weight_tmp[1] = 1
        weight_tmp[6] = 1
        weight_tmp[0] = 1

        self.weight_cross_entropy = torch.nn.CrossEntropyLoss(weight=weight_tmp).cuda()

        self.hair_percept = masked_lpips.PerceptualLoss(
            model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=self.use_gpu
        )
        self.hair_percept.eval()


    def cross_entropy_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy(down_seg, target_mask)
        return loss
    def weight_cross_entropy_loss(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy(down_seg, target_mask)
        return loss


    def style_loss(self, im1, im2, mask1, mask2):
        loss = self.opt.style_lambda * self.style(im1 * mask1, im2 * mask2, mask1=mask1, mask2=mask2)
        return loss


    def cross_entropy_loss_wo_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_wo_background(down_seg, target_mask)
        return loss

    def cross_entropy_loss_only_background(self, down_seg, target_mask):
        loss = self.opt.ce_lambda * self.cross_entropy_only_background(down_seg, target_mask)
        return loss
    
    def _loss_hair_percept(self, gen_im, ref_im, mask, **kwargs):
        return self.hair_percept(gen_im, ref_im, mask=mask)