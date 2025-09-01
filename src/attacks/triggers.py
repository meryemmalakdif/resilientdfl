from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import Tuple
import torchvision.transforms.functional as TF
import torch

class Trigger(ABC):
    @abstractmethod
    def apply(self, img):  # img can be PIL Image or Tensor
        pass

class PatchTrigger(Trigger):
    def __init__(self, patch_size: int = 3, value: int = 255, position: Tuple[int,int]=None):
        self.patch_size = patch_size
        self.value = value
        self.position = position

    def _apply_pil(self, pil_img: Image.Image):
        arr = np.array(pil_img.convert("RGB"))
        h, w = arr.shape[:2]
        ps = self.patch_size
        if self.position is None:
            x = h - ps - 1
            y = w - ps - 1
        else:
            x, y = self.position
        arr[x:x+ps, y:y+ps] = self.value
        return Image.fromarray(arr)

    def apply(self, img):
        # support both PIL and tensors
        if isinstance(img, Image.Image):
            return self._apply_pil(img)
        if isinstance(img, torch.Tensor):
            pil = TF.to_pil_image(img)
            out = self._apply_pil(pil)
            return TF.to_tensor(out)
        raise TypeError("Unsupported image type")

class BlendedTrigger(Trigger):
    """Blend a semi-transparent watermark."""
    def __init__(self, mask: np.ndarray, strength: float = 0.2):
        self.mask = mask  # same HxW or smaller
        self.strength = strength
    def apply(self, img):
        # implement blending for PIL or Tensor as needed
        raise NotImplementedError
