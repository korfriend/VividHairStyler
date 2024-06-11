import torch

def rgb_to_lab(rgb):
    # RGB에서 sRGB로 (정규화된 값을 0-1로 스케일 조정)
    rgb = (rgb + 1.0) / 2.0

    # sRGB를 XYZ로 변환
    def srgb_to_xyz(srgb):
        mask = srgb > 0.04045
        srgb[mask] = ((srgb[mask] + 0.055) / 1.055) ** 2.4
        srgb[~mask] /= 12.92
        srgb *= 100.0
        xyz = torch.matmul(srgb, torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                               [0.2126729, 0.7151522, 0.0721750],
                                               [0.0193339, 0.1191920, 0.9503041]]).to(srgb.device))
        return xyz

    # XYZ를 LAB로 변환
    def xyz_to_lab(xyz):
        xyz /= torch.tensor([95.047, 100.000, 108.883]).to(xyz.device)
        mask = xyz > 0.008856
        xyz[mask] = torch.pow(xyz[mask], 1/3)
        xyz[~mask] = (7.787 * xyz[~mask]) + (16/116)
        # LAB 변환 수행
        L = 116 * xyz[..., 1] - 16
        a = 500 * (xyz[..., 0] - xyz[..., 1])
        b = 200 * (xyz[..., 1] - xyz[..., 2])
        lab = torch.stack([L, a, b], dim=-1)  # 마지막 차원을 기준으로 쌓음
        return lab

    # sRGB로 변환
    rgb = rgb.permute(0, 2, 3, 1)  # BCHW -> BHWC
    xyz = srgb_to_xyz(rgb)
    # print(f"xyz: {xyz.shape}")
    lab = xyz_to_lab(xyz)
    # print(f"lab: {lab.shape}")
    lab = lab.permute(0, 3, 1, 2)  # BHWC -> BCHW
    return lab


def rgb_to_grayscale(rgb_image):
    """
    RGB 이미지를 그레이스케일로 변환하는 함수
    :param rgb_image: (N, C, H, W) 형태의 텐서. C는 3이어야 합니다(RGB).
    :return: (N, 1, H, W) 형태의 그레이스케일 이미지
    """
    if rgb_image.size(1) != 3:
        raise ValueError("입력 이미지는 RGB 채널을 가져야 합니다.")
    
    # RGB 채널에 대한 가중치
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32, device=rgb_image.device)
    weights = weights.view(1, 3, 1, 1)  # 연산을 위한 차원 조정
    
    # 가중 평균을 계산하여 그레이스케일 이미지 생성
    grayscale_image = torch.sum(rgb_image * weights, dim=1, keepdim=True)
    rgb_image = grayscale_image.repeat(1, 3, 1, 1)

    return rgb_image