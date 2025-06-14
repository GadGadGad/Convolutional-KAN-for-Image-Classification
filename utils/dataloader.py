import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100
import argparse
import logging
logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_RESIZE_SIZE = 256
IMAGENET_CROP_SIZE = 224

def get_dataloader(dataset: str,
                   batch_size: int,
                   data_path: str,
                   num_workers: int,
                   imagenet_preprocessing: bool=False):
    if imagenet_preprocessing:
        logger.info(f"Getting dataloader for {dataset}. ImageNet preprocessing: {imagenet_preprocessing}")
    else:
        logger.info(f"Getting dataloader for {dataset}.")

    # transform = transforms.Compose([transforms.ToTensor()])
    SVHN_CLASSES = [str(i) for i in range(10)]
    if imagenet_preprocessing:

        if dataset == 'MNIST':
                transform_train = transforms.Compose([
                    transforms.Resize(IMAGENET_CROP_SIZE),
                    transforms.Grayscale(num_output_channels=3), 
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
                transform_test = transforms.Compose([
                    transforms.Resize(IMAGENET_CROP_SIZE),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
        else: 
                transform_train = transforms.Compose([
                    transforms.Resize(IMAGENET_RESIZE_SIZE),
                    transforms.RandomResizedCrop(IMAGENET_CROP_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
                transform_test = transforms.Compose([
                    transforms.Resize(IMAGENET_RESIZE_SIZE),
                    transforms.CenterCrop(IMAGENET_CROP_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
    else:
        if dataset == 'MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transform_train
        elif dataset == 'SVHN':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
            transform_test = transform_train 
        elif dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        elif dataset == 'CIFAR100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15), 
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

    dataset_root = data_path 
    if dataset == 'MNIST':
        train_set = MNIST(root=dataset_root, train=True, download=True, transform=transform_train)
        test_set = MNIST(root=dataset_root, train=False, download=True, transform=transform_test)
        classes = SVHN_CLASSES
    elif dataset == 'SVHN':
        train_set = SVHN(root=dataset_root, split='train', download=True, transform=transform_train)
        test_set = SVHN(root=dataset_root, split='test', download=True, transform=transform_test)
        classes = SVHN_CLASSES
    elif dataset == 'CIFAR10':
        train_set = CIFAR10(root=dataset_root, train=True, download=True, transform=transform_train)
        test_set = CIFAR10(root=dataset_root, train=False, download=True, transform=transform_test)
        classes = train_set.classes
    elif dataset == 'CIFAR100':
        train_set = CIFAR100(root=dataset_root, train=True, download=True, transform=transform_train)
        test_set = CIFAR100(root=dataset_root, train=False, download=True, transform=transform_test)
        classes = train_set.classes


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_set, test_set, classes


def main():
    parser = argparse.ArgumentParser(description="Dataloader for experiments")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100'], help='Choose the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_path', type=str, default="./data", help='Dataset root path')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--imagenet_preprocessing', action='store_true', help='Use ImageNet preprocessing (resize/normalize)')

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO) 
    
    train_loader, test_loader, train_set, test_set, classes = get_dataloader(
        args.dataset, 
        args.batch_size, 
        args.data_path,
        args.num_workers,
        args.imagenet_preprocessing
    )
    
    logger.info(f"\n--- Dataset Information ({args.dataset}) ---")
    logger.info(f"Using ImageNet Preprocessing: {args.imagenet_preprocessing}")
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Test set size: {len(test_set)}")
    logger.info(f"Number of classes: {len(classes)}")
    logger.info(f"Classes: {classes}") 

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    logger.info("\n--- Sample Batch Info ---")
    logger.info(f"Image batch shape: {images.shape}")
    logger.info(f"Label batch shape: {labels.shape}")
    logger.info(f"Image data type: {images.dtype}")
    logger.info(f"Label data type: {labels.dtype}")
    logger.info(f"Image min value: {images.min():.4f}")
    logger.info(f"Image max value: {images.max():.4f}")

if __name__ == "__main__":
    main()