import os

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader


def load(dataset='mnist', dataset_root='./datasets', batch_size=32, image_size=None):
    # Define default sizes for datasets
    default_sizes = {
        'mnist': 28,
        'emnist': 28,
        'cifar': 32,
        'imagenette': 128
    }
    dataset = dataset.lower()
    if image_size is None:
        image_size = default_sizes.get(dataset, 128)  # Default to 128 if dataset unknown

    
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(2/ image_size, 2 / image_size)),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform_test)
    elif dataset == 'emnist':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(2/ image_size, 2 / image_size)),
            transforms.ToTensor(),
            lambda x: x.permute(0, 2, 1)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            lambda x: x.permute(0, 2, 1)
        ])
        train_dataset = torchvision.datasets.EMNIST(root=dataset_root, split='letters', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.EMNIST(root=dataset_root, split='letters', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(image_size, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform_test)
    elif dataset == 'imagenette':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        if not os.path.exists(f'{dataset_root}/imagenette2'):
            _ = torchvision.datasets.Imagenette(dataset_root, split="train", download=True, transform=transform_train)
        train_dataset = torchvision.datasets.Imagenette(dataset_root, split="train", download=False, transform=transform_train)
        test_dataset = torchvision.datasets.Imagenette(dataset_root, split="val", download=False, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def plot_sample_grid(data_loader, grid_size=(5, 5), fig_size=None, verbose=0):
    CLASS_LENGTH_LIMIT = 15

    dataset = data_loader.dataset
    num_samples = grid_size[0] * grid_size[1]

    if fig_size is None:
        fig_size = (grid_size[1] * 1.5, grid_size[0] * 1.5)  # Default fig size based on grid size

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=fig_size)
    axes = axes.flatten()

    used_classes = set()
    class_names = getattr(dataset, 'classes', None)

    max_class_length = max((len(str(name)) for name in class_names), default=0) if class_names else -1

    for i in range(num_samples):
        # getting image as nparray
        img, label = dataset[np.random.randint(len(dataset))]
        img = img.numpy() if not isinstance(img, np.ndarray) else img
        # plotting image
        if img.shape[0] == 3:  # RGB Image
            img = np.transpose(img, (1, 2, 0))
            axes[i].imshow(img)
        elif img.shape[0] == 1:  # Grayscale Image
            axes[i].imshow(img[0], cmap='gray')
        else:
            raise RuntimeError(f'Incorrect number of channels: {img.shape[0]}')
        # adding label
        used_classes.add(label)
        if max_class_length > CLASS_LENGTH_LIMIT or max_class_length < 0:
            axes[i].set_title(f"Class #{label}") # can't print class name on figure
        else:
            axes[i].set_title(f"{class_names[label]}")

        axes[i].axis("off")

    fig.tight_layout()

    if verbose and max_class_length > CLASS_LENGTH_LIMIT:
        print("Class legend:")
        for label in sorted(used_classes):
            class_name = class_names[label] if class_names else f"Class #{label}"
            print(f"{label}: {class_name}")

    return fig, axes