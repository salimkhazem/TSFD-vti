import os
import pathlib
import random

from torch.utils.data import DataLoader  # type: ignore

from .dataset import DatasetVTI


def create_datasets(
    root_dir,
    num_slices=1,
    augmentation=None,
    validation_split=0.2,
    input_filename="xray.vti",
    output_filename="contours.vti",
):
    """
    Create training and validation datasets from VTI files.
    Args:
        root_dir (str): Root directory containing the VTI files.
        num_slices (int): Number of slices to extract from each VTI file.
        augmentation (callable): Optional augmentation function to be applied
        to the input data.
        validation_split (float): Fraction of the data to use as validation.
        input_filename (str): Name of the input file.
        output_filename (str): Name of the output file.
    Returns:
        train_dataset (DatasetVTI): Training dataset.
        val_dataset (DatasetVTI): Validation dataset.
        training_paths (list): List of paths to the training VTI files.
        validation_paths (list): List of paths to the validation VTI files.
    """
    all_data_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for diri in dirs:
            datapath = pathlib.Path(os.path.join(subdir, diri))
            input_filepath = datapath / input_filename
            output_filepath = datapath / output_filename
            if input_filepath.exists() and output_filepath.exists():
                all_data_paths.append(datapath)

    random.shuffle(all_data_paths)
    validation_size = int(len(all_data_paths) * validation_split)
    validation_paths = all_data_paths[:validation_size]
    training_paths = all_data_paths[validation_size:]

    train_dataset = DatasetVTI(
        training_paths,
        num_slices,
        augmentation,
        input_filename,
        output_filename,
    )
    val_dataset = DatasetVTI(
        validation_paths,
        num_slices,
        augmentation,
        input_filename,
        output_filename,
    )

    return train_dataset, val_dataset, training_paths, validation_paths


def create_dataloaders(
    train_dataset, valid_dataset, batch_size=8, shuffle=True
):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, valid_loader


if __name__ == "__main__":
    root_dir = "/mnt/WoodSeer/Slicing"
    train_dataset, valid_dataset, train_paths, valid_paths = create_datasets(
        root_dir, validation_split=0.2
    )
    train_loader, valid_loader = create_dataloaders(
        train_dataset, valid_dataset
    )
