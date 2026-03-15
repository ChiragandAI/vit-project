import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_tinyimagenet_dataloaders(data_dir, batch_size=64, num_workers=4):
    """
    Creates TinyImageNet training and validation dataloaders.

    Parameters
    ----------
    data_dir : str
        Root directory of TinyImageNet dataset.
        Expected structure:

        data_dir/
            train/
                class_1/
                    images...
                class_2/
                    images...
            val/
                class_1/
                    images...
                class_2/
                    images...

    batch_size : int
        Number of samples per batch.

    num_workers : int
        Number of parallel workers used for loading data.

    Returns
    -------
    train_loader : DataLoader
        Training dataloader.

    val_loader : DataLoader
        Validation dataloader.
    """

    # -----------------------------
    # BASIC PREPROCESSING TRANSFORMS
    # -----------------------------
    # These are NOT augmentations.
    # These are required steps so the data can be used by PyTorch models.

    base_transform = transforms.Compose([

        # Convert PIL image to PyTorch tensor
        # Output shape becomes (C, H, W)
        transforms.ToTensor(),

        # Normalize pixel values
        # Helps stabilize training by keeping feature magnitudes consistent
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # -----------------------------
    # DATASET PATHS
    # -----------------------------

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # -----------------------------
    # DATASETS
    # -----------------------------
    # ImageFolder automatically:
    # 1. Reads subdirectories as class labels
    # 2. Assigns numerical labels
    # 3. Loads images

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=base_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=base_transform
    )

    # -----------------------------
    # DATALOADERS
    # -----------------------------
    # DataLoader handles:
    # batching
    # shuffling
    # parallel loading

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # shuffle training data
        num_workers=num_workers,
        pin_memory=True        # speeds up GPU transfer (safe even on CPU)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # never shuffle validation set
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
