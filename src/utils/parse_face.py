import dlib
from pathlib import Path
import torchvision
from .shape_predictor import align_face
from .drive import open_url
from typing import Union

def parse_face(
    path: Union[str, Path], 
    cache_dir: Union[str, Path] = Path("cache"), 
    model_weights: str = "https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", 
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    # path = Path(os.path.join(cache_dir, "uploaded_image.png"))
    # im = Image.open(uploaded_image)
    # im.save(path)

    f=open_url(model_weights, cache_dir=cache_dir, return_path=True)
    predictor = dlib.shape_predictor(f)

    faces = align_face(str(path), predictor)
    images = []
    for i,face in enumerate(faces):
        face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
        face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
        face = torchvision.transforms.ToPILImage()(face_tensor_lr)
        # face.save(cache_dir / "image.png")
    images.append(face)
    return images
    # if len(faces) > 1:
    #     face.save(Path(cache_dir) / (path.stem+f"_{i}.png"))
    # else:
    #     face.save(Path(cache_dir) / (path.stem + f".png"))