# FS Code Style Transfer: Improved Hair Transfer for Effective Sketch Hair Editing

> **Abstract** Recent advances in deep generative models have enabled realistic hairstyle editing. However, hair editing remains a challenging problem because it requires a convenient and intuitive interface that accurately reflects the user's preference, and the capability to precisely reconstruct the complex features of hair. Hair transfer, blending a reference image's hairstyle into a source image, is widely used for its simplicity. Nevertheless, due to semantic misalignment and spatial feature discrepancies between the reference and source images, the detailed features of the reference hairstyle, e.g., hair color and texture, are often not accurately reflected in the source image. Sketch tools allow users to intuitively depict the desired hairstyles on specific areas of the source image, free from this issue, but they impose a significant design burden on users for representing the overall hairstyle according to the user's intent. We propose a unified hair editing system that utilizes our improved hair transfer model and sketch operations by leveraging the capabilities of latent space blending. First, to improve the hair transfer, we apply a set of masked version of perceptual and style losses for hair and propose an optimal architecture where the losses can effectively influence the latent space optimization. Additionally, hair mask manipulation and hair patch generation from user-sketch operations allows for local fine-grained hairstyle editing in conjunction with our enhanced hair transfer system. Quantitative and qualitative evaluations, including user preference studies, demonstrate that our hairstyle editing system outperforms current state-of-the-art techniques in both hairstyle generation quality and usability.

## Description
Official Implementation of "FS Code Style Transfer". **KEEP UPDATING! Please Git Pull the latest version.**

## Updates
`2024/06/07` All source codes have been uploaded

## Installation
- Clone the repository:
``` 
git clone https://github.com/korfriend/VividHairStyler.git
cd VividHairStyler
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment.yaml`.

## Download sample images
Please download the [FFHQ](https://drive.google.com/drive/folders/1RxzbNcKb3bPDKccyo300YXCJ8EvZSaIL) 

## Getting Started  
Preparing...

## Web UI

You can try out our online web demo at 

## Acknowledgments
This code borrows heavily from [BARBERSHOP](https://github.com/ZPdesu/Barbershop).

