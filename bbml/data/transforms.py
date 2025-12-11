
from typing import Any, Literal

from PIL import Image
import torch
from torchvision.transforms import v2 as T

from bbml.core.data_transform import DataTransform


class ImageDataTransform(DataTransform):
    def __init__(
        self,
        size:int|None=None,
        crop:Literal["center","random"]|None=None,
    ):
        self.size = size
        self.crop = crop

        augs = []
        if self.size is not None:
            if isinstance(self.size, int):
                # do resize when size is int (resize resizes it to shorter side)
                # if [int, int] just crop is good
                augs.append(T.Resize(self.size))
            if self.crop == "center":
                augs.append(T.CenterCrop(self.size))
            elif self.crop == "random":
                augs.append(T.RandomCrop(self.size))

        self.tv_transforms = T.Compose([
            T.ToImage(),
            T.RGB(),
            *augs,
            T.ToDtype(torch.float, scale=True),
        ])


    def transform(self, inp: Any) -> torch.Tensor:
        return self.tv_transforms(inp)
    
    def batch_transform(self, inp: list[torch.Tensor]):
        return torch.stack(inp).to(memory_format=torch.contiguous_format)


class IdentityDataTransform(DataTransform):
    def transform(self, inp: Any) -> Any:
        return inp
    
    def batch_transform(self, inp: list) -> list:
        return inp

class UnsqueezeDataTransform(DataTransform):
    def transform(self, inp: Any) -> Any:
        return inp
    
    def batch_transform(self, inp: list) -> list:
        assert len(inp) == 1
        return inp[0]
