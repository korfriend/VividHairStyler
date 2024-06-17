# FS Code Style Transfer: Improved Hair Transfer for Effective Sketch Hair Editing
(new title: Improved Hairstyle Transfer: Latent Code Optimization for Vivid Hair Representation and Sketch Hair Editing)

> **Abstract** Recent advances in deep generative models have enabled realistic hairstyle editing. However, hair editing remains a challenging problem because it requires a convenient and intuitive interface that accurately reflects the user's preference, and the capability to precisely reconstruct the complex features of hair. Hair transfer, applying a hairstyle from a reference image to a source image, is widely used for its simplicity. Nevertheless, semantic misalignment and spatial feature discrepancies between the reference and source images lead to the detailed features of the reference hairstyle, such as hair color and strand texture, often not being accurately reflected in the source image. Free from this issue, sketch tools allow users to intuitively depict the desired hairstyles on specific areas of the source image, but they impose a significant design burden on users and present a technical challenge in generating natural-looking hair that seamlessly incorporates the sketch details into the source image. In this paper, we present an improved hair transfer system that utilizes latent space optimizations with masked perceptual and style losses. Our system effectively captures detailed hair features, including vibrant hair colors and strain textures, resulting in more realistic and visually compelling hair transfers. Additionally, we introduce user-controllable components used in our hair transfer process, empowering users to refine the desired hairstyle. Our sketch interfaces can efficiently manipulate these components, providing enhanced editing effects through our improved hair transfer capabilities. Quantitative and qualitative evaluations, including user preference studies, demonstrate that our hairstyle editing system outperforms current state-of-the-art techniques in both hairstyle generation quality and usability.

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

## Download additional files
Please download the additional `.pth` files required for the project from the following link: <a href="https://drive.google.com/drive/folders/1R-Af1g8kkjkiQVGGiODhBNPQiJw6t32Y?usp=sharing" target="_blank">Additional Files</a>.


## Getting Started  
Preparing...

## Web UI

You can use the web UI by running the following command in the `/VividHairStyler` directory:
```
streamlit run VividHairStyler.py
```


## Acknowledgments
This code borrows heavily from [BARBERSHOP](https://github.com/ZPdesu/Barbershop).

