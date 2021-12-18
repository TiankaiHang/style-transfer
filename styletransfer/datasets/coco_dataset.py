import os
from PIL import Image
import torch

import torch.utils.data as data
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, 
    Normalize, RandomResizedCrop)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _to_tensor(x):
    return torch.from_numpy(x)

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def _transform(n_px):
    return Compose([
        RandomResizedCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), 
                  (0.229, 0.224, 0.225)),
    ])


class COCODataset(data.Dataset):
    def __init__(self, data_dir='datasets', is_train=True, img_size=256,
                 style_path=None) -> None:
        super().__init__()
        self.data_dir = data_dir

        self.file_list = os.listdir(self.data_dir)
        self.file_list = [os.path.join(self.data_dir, _f) for _f in self.file_list]

        if is_train:
            self.file_list = self.file_list[:-16]
        else:
            self.file_list = self.file_list[-16:]

        self.transform = _transform(img_size)

        self.style_path = style_path

    def __getitem__(self, index: int):
        img = Image.open(self.file_list[index])
        img = self.transform(img)

        if self.style_path is not None:
            assert os.path.isfile(self.style_path), \
                f"Style Image {self.style_path} does not exist!"
            style_image = Image.open(self.style_path)
            style_image = self.transform(style_image)
            return img, style_image

        return img

    def __len__(self) -> int:
        return len(self.file_list)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cocodataset = COCODataset(data_dir="/data/tiankai/datasets/MSCOCO/train2014")
    to_show = cocodataset[3]
    to_show = _normalize(to_show)
    plt.imshow(to_show.permute(1, 2, 0).numpy())
    # plt.show()
    plt.savefig("/home/v-tiahang/.cache/test.png")