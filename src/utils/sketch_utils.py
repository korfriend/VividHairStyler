import cv2
import tqdm
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import KDTree

def parse_json(json_data, bg=None):
    # JSON 데이터 로드
    drawing_data = json_data # json.loads(json_data)
    paths = drawing_data['objects']  # 'objects' 키에 선의 정보가 저장되어 있음
    if bg is None:
        img = Image.new('RGB', (512, 512), (0, 0, 0))
    else:
        img = bg.convert("RGB")
    draw = ImageDraw.Draw(img)
    new_sketch_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    
    for i, path in enumerate(paths):
        if path['type'] == 'path':  # 선의 타입 확인
            # 배경을 검은색으로 설정한 새 이미지 생성

            # 선의 색상과 두께를 추출
            stroke_color = path.get('stroke', '#000000')  # 기본값은 검은색
            stroke_width = path.get('strokeWidth', 25)  # 기본값은 3
            # 선의 좌표를 추출하고 그림
            points = path['path']  # 선의 좌표
            for j in range(1, len(points)):
                start_point = (points[j-1][1], points[j-1][2])
                end_point = (points[j][1], points[j][2])
                draw.line([start_point, end_point], fill=stroke_color, width=stroke_width)
    sketch = np.array(img)
    # sketch_mask = ~np.all(sketch == [0, 0, 0], axis=-1)
    # sketch_mask3 = np.dstack([sketch_mask,sketch_mask,sketch_mask])
    # new_sketch_rgb[sketch_mask3] = sketch[sketch_mask3]
    return sketch

def get_unique_colors(im, except_black=False):
    unique_colors = np.unique(im.reshape(-1, im.shape[2]), axis=0)
    if except_black:
        unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # 검정색 제거
    return unique_colors

def sketch_segment(sketch, hair_mask, sketch_mask=None):
    if sketch_mask is None:
        sketch_mask = ~np.all(sketch == [0, 0, 0], axis=-1)

    # 색상 마스크 초기화 (검정색으로)
    color_mask = np.zeros_like(sketch)

    # 스케치 픽셀의 위치 찾기
    nonzero_x, nonzero_y = np.nonzero(sketch_mask)
    sketch_pixels = np.stack((nonzero_x, nonzero_y), axis=-1)

    # KDTree를 사용하여 가장 가까운 스케치 픽셀 찾기
    tree = KDTree(sketch_pixels)

    for i in tqdm.trange(sketch_mask.shape[0]):
        for j in range(sketch_mask.shape[1]):
            if hair_mask[i, j] == 0:
                continue  # 마스크가 0이면 검정색을 유지
            _, idx = tree.query([i, j])
            nearest_sketch_pixel = sketch_pixels[idx]

            # 스케치의 색상을 색상 마스크에 적용
            color_mask[i, j] = sketch[nearest_sketch_pixel[0], nearest_sketch_pixel[1]]
    return color_mask

def sketch_sub_mask(color_mask, color_mask_dilation = 10):
    sketch_mask = ~np.all(color_mask == [0, 0, 0], axis=-1)
    # color_list = []
    sketch_mask_list = []
    unique_colors = np.unique(color_mask.reshape(-1, color_mask.shape[2]), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # 검정색 제거
    for unique_color in tqdm.tqdm(unique_colors):
        if np.sum(unique_color) == 0:
            continue
        sub_mask = np.all(color_mask == unique_color, axis=-1)
        if color_mask_dilation > 0:
            # sub_mask_uint8 = ~np.all(sketch == [0, 0, 0], axis=-1)
            # sub_mask_dilate = cv2.dilate(sub_mask.astype(np.uint8), kernel=np.ones((7, 7), np.uint8), iterations=color_mask_dilation)
            sub_mask_dilate = cv2.GaussianBlur(sub_mask.astype(np.uint8)*255, (0, 0), sigmaX=color_mask_dilation, sigmaY=color_mask_dilation)/255
            sub_mask = np.where(
                np.logical_and(sub_mask_dilate>0, sketch_mask), 
                sub_mask_dilate, 
                0
            )

        # color_list.append(unique_color)
        # 마스크를 uint8 형으로 변환 (0 또는 255)
        sketch_mask_list.append((sub_mask * 255).astype(np.uint8))
        # break
    
    return sketch_mask_list