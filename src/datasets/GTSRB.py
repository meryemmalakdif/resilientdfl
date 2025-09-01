from .adapter import DatasetAdapter
from torchvision.datasets import GTSRB
from .transforms import get_transforms
from typing import Optional

class GTSRBAdapter(DatasetAdapter):
    def __init__(self, root: str = "data", train: bool = True, download: bool = True, transform=None):
        # default transform if not provided
        if transform is None:
            transform = get_transforms("gtsrb", train=train)
        self.transform = transform
        super().__init__(root=root, train=train, download=download, transform=transform)

    def load_dataset(self) -> None:
        self._dataset = GTSRB(root=self.root, train=self.train, transform=self.transform, download=self.download)