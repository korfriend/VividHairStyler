
import numpy as np
import os
import PIL
import torch
def vis_seg(pred):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().squeeze().cpu().numpy()
    num_labels = 16

    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 255, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7
                      [0, 255, 255],  ## 8
                      [0, 255, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      ])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]
    return rgb

def vis_seg_reverse(rgb):
    # Define the color to label mapping
    color_to_label = {
        (0, 0, 0): 0,
        (102, 204, 255): 1,
        (255, 204, 255): 2,
        (255, 255, 153): 3,
        (255, 255, 102): 5,
        (51, 255, 51): 6,
        (0, 153, 255): 7,
        (0, 255, 255): 8,
        (204, 102, 255): 10,
        (0, 255, 153): 12,
        (0, 51, 0): 13,
        (102, 153, 255): 14,
        (255, 153, 102): 15
    }
    
    # Initialize the output label array
    h, w, _ = rgb.shape
    label = np.zeros((h, w), dtype=np.uint8)
    
    # Iterate over the color_to_label dictionary and create masks for each color
    for color, label_value in color_to_label.items():
        mask = (rgb == color).all(axis=-1)
        label[mask] = label_value
    
    # Handle unknown colors (those not in the color_to_label mapping)
    unknown_mask = np.ones((h, w), dtype=bool)
    for color in color_to_label.keys():
        unknown_mask &= ~(rgb == color).all(axis=-1)
    label[unknown_mask] = 255
    
    return label

def save_vis_mask(img_path1, img_path2, sign, output_dir, mask):
    im_name_1 = os.path.splitext(os.path.basename(img_path1))[0]
    im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]
    vis_path = os.path.join(output_dir, 'vis_mask_{}_{}_{}.png'.format(im_name_1, im_name_2, sign))
    vis_mask = vis_seg(mask)
    PIL.Image.fromarray(vis_mask).save(vis_path)
