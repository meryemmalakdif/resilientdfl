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
    
    if dataset_name.lower() == 'gtsrb':
        if train:
            return transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
            ])
        else:
            return  transforms.Compose([
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ])
        
    if dataset_name.lower() == 'femnist':
        if train:
            return transforms.Compose([
                    transforms.Resize((28, 28)),             # FEMNIST usually 28x28
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
                    transforms.ToTensor(),                   # -> shape (1, H, W)
                    transforms.Normalize((0.1307,), (0.3081,))  # single-channel mean/std
                ])
        else:
            return transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])



    # fallback: basic transform (resize if requested)
    t = [transforms.ToTensor()]
    if image_size is not None:
        t.insert(0, transforms.Resize(image_size))
    return transforms.Compose(t)