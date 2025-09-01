from torchvision import transforms
from typing import Tuple




def get_transforms(dataset_name: str, image_size: Tuple[int,int]=None, train: bool=True):
    """Return torchvision transforms for common datasets.


    - CIFAR-style: 32x32
    - MNIST: 28x28
    """
    if dataset_name.lower() in ('cifar10','cifar100'):
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])


    if dataset_name.lower() == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


    # fallback: basic transform (resize if requested)
    t = [transforms.ToTensor()]
    if image_size is not None:
        t.insert(0, transforms.Resize(image_size))
    return transforms.Compose(t)