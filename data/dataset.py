"""
This module contains the DatasetVTI class, which is a subclass of the PyTorch
Dataset class. It is used to load VTI files and extract slices from them.
"""

import argparse
import logging
import pathlib

import xarray as xr
import cv2
import torch
import tqdm
import vtk
import numpy as np
from vtk.util import numpy_support
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def vtkToNumpy(data):
    """
    Transform vtk data to numpy.
    """
    tmp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()
    numpy_data = tmp.reshape(dims[2], dims[1], dims[0])
    numpy_data = numpy_data.transpose(1, 2, 0)
    return numpy_data


def read_vti(file):
    vtk_f = vtk.vtkXMLImageDataReader()
    vtk_f.SetFileName(file)
    vtk_f.Update()
    np_array = vtkToNumpy(vtk_f.GetOutput())
    return np_array


def load_nc_chunk(nc_filepath, chunk_idx, num_slices):
    ds = xr.open_dataset(nc_filepath)
    z_vals = ds.data.isel(z=slice(chunk_idx, chunk_idx + num_slices))
    z_vals = z_vals.to_numpy()
    return z_vals


class DatasetVTI(Dataset):
    """
    Dataset class for VTI files.
    Args:
        data_paths (list): List of paths to the VTI files.
        num_slices (int): Number of slices to extract from each VTI file.
        augmentation (callable): Optional augmentation function to be applied
        to the input data.
        input_filename (str): Name of the input file.
        output_filename (str): Name of the output file.
    """

    def __init__(
        self,
        data_paths,
        num_slices=1,
        resize=256,
        augmentation=None,
        input_filename="xray.vti",
        output_filename="contours.vti",
    ):
        self.resize = resize
        self.num_slices = num_slices
        self.transform = augmentation
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.data_paths = data_paths

        self.num_chunks_per_file = []
        # Parallel processing of data paths
        self.num_chunks_per_file = Parallel(n_jobs=-1)(
            delayed(self.process_path)(
                dp, self.input_filename, self.output_filename, self.num_slices
            )
            for dp in tqdm.tqdm(self.data_paths, desc="Parsing VTI file")
        )
        self.total_num_chunks = sum(self.num_chunks_per_file)

    @staticmethod
    def process_path(dp, input_filename, output_filename, num_slices):
        input_vti = read_vti(dp / input_filename)
        output_vti = read_vti(dp / output_filename)
        if (
            input_vti is None
            or output_vti is None
            or input_vti.shape != output_vti.shape
        ):
            raise RuntimeError(
                f"Got different input/targets shape, {input_vti.shape} != {output_vti.shape}"
            )
        return input_vti.shape[2] // num_slices

    def __len__(self):
        return self.total_num_chunks * len(self.data_paths)

    def _process_chunk(self, dp, idx):
        input_vti = read_vti(dp / self.input_filename)
        input_chunk = input_vti[
            :, :, idx * self.num_slices : (idx + 1) * self.num_slices
        ]
        input_chunk = np.transpose(input_chunk, (2, 0, 1))
        hmin, hmax = 100, 1500
        input_chunk = np.clip(input_chunk, hmin, hmax)
        input_chunk = 2.0 * ((input_chunk - hmin) / (hmax - hmin) - 0.5)
        output_vti = read_vti(dp / self.output_filename)
        output_chunk = output_vti[
            :, :, idx * self.num_slices : (idx + 1) * self.num_slices
        ]
        output_chunk = np.transpose(output_chunk, (2, 0, 1))
        assert (
            input_chunk.shape == output_chunk.shape
        ), f"Expected {input_chunk.shape} but got {output_chunk.shape}"
        if self.resize is not None:
            input_chunk = cv2.resize(input_chunk[0], (self.resize, self.resize))
            input_chunk = np.expand_dims(input_chunk, axis=0)
            output_chunk = cv2.resize(output_chunk[0], (self.resize, self.resize))
            output_chunk = np.expand_dims(output_chunk, axis=0)
        if self.transform is not None:
            input_chunk = self.transform(input_chunk)
            output_chunk = self.transform(output_chunk)
        return input_chunk, output_chunk

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.total_num_chunks
        for idp, (dp, nc) in enumerate(zip(self.data_paths, self.num_chunks_per_file)):
            if idx < nc:
                break
            idx -= nc

        input_chunk, output_chunk = self._process_chunk(dp, idx)
        return {
            "input": torch.tensor(input_chunk, dtype=torch.float),
            "target": torch.tensor(output_chunk, dtype=torch.float),
        }


class DatasetNC(Dataset):
    """
    Dataset for the NC files
    Args:
        - rootdir (str | pathlib.Path): The rootdir of the dataset
        - num_slices (int): Number of slices to extract from each NC file.
        - transform (callable): Optional transform function to be applied
          to the input/target data.
        - input_filename (str): Name of the input file.
        - output_filename (str): Name of the output file.
    """

    def __init__(
        self,
        rootdir,
        num_slices=1,
        transform=None,
        input_filename="data.nc",
        output_filename="label.nc",
    ):
        self.rootdir = rootdir
        self.num_slices = num_slices
        self.transform = transform
        self.input_filename = input_filename
        self.output_filename = output_filename

        self.num_chunks_per_file = []

        # Recursively find all the input files in the root directory
        # and keep the input/output pairs when both files are co-present
        if isinstance(rootdir, str):
            rootdir = pathlib.Path(rootdir)

        self.data_paths = []
        for dp in rootdir.rglob(input_filename):
            if (dp.parent / output_filename).exists():
                self.data_paths.append(dp.parent)
        logging.debug(f"Found {self.data_paths} data paths")

        # Parallel processing of data paths
        self.num_chunks_per_file = Parallel(n_jobs=-1)(
            delayed(self.process_path)(
                dp, self.input_filename, self.output_filename, self.num_slices
            )
            for dp in tqdm.tqdm(self.data_paths, desc="Parsing NC file")
        )
        self.total_num_chunks = sum(self.num_chunks_per_file)
        logging.info(
            f"Total number of loaded chunks of {self.num_slices} slices: {self.total_num_chunks}"
        )

    @staticmethod
    def process_path(dp, input_filename, output_filename, num_slices):
        input_ds = xr.open_dataset(dp / input_filename)  # z, x, y
        output_ds = xr.open_dataset(dp / output_filename)  # z, x, y
        input_dims = [input_ds.sizes[d] for d in ["x", "y", "z"]]
        output_dims = [output_ds.sizes[d] for d in ["x", "y", "z"]]
        if input_dims != output_dims:
            raise RuntimeError(
                f"Got different input/targets shape, {input_dims} != {output_dims}"
            )
        return input_ds.sizes["z"] // num_slices

    def __len__(self):
        return self.total_num_chunks

    def __getitem__(self, idx):
        print(f"Getting idx {idx} ")
        assert idx >= 0 and idx < self.total_num_chunks
        for idp, (dp, nc) in enumerate(zip(self.data_paths, self.num_chunks_per_file)):
            # Locate the files that contain the requested chunk
            if idx < nc:
                break
            idx -= nc

        # We found the datapath from which to extract the  idx chunk
        input_chunk = load_nc_chunk(dp / self.input_filename, idx, self.num_slices)
        output_chunk = load_nc_chunk(dp / self.output_filename, idx, self.num_slices)

        if self.transform is not None:
            input_tensor, output_tensor = self.transform(input_chunk, output_chunk)
        else:
            input_tensor = torch.tensor(input_chunk)
            output_tensor = torch.tensor(output_chunk)

        return {"input": input_tensor, "target": output_tensor}


def test_nc_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", type=str, help="Root directory containing NC files")
    parser.add_argument(
        "--num_slices", type=int, default=1, help="Number of slices per sample"
    )
    parser.add_argument(
        "--input_filename", type=str, default="data.nc", help="Input NC filename"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="label.nc",
        help="Output NC filename",
    )

    args = parser.parse_args()

    dataset = DatasetNC(
        args.rootdir,
        args.num_slices,
        input_filename=args.input_filename,
        output_filename=args.output_filename,
    )

    # Let us randomly load some chunks
    for i in range(10):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        logging.info(
            f"Input shape: {sample['input'].shape}, Output shape: {sample['target'].shape}"
        )

    # Let us display one slice of a 4 chunks, with its input along with its target
    num = 4
    fig, axes = plt.subplots(nrows=num, ncols=2, figsize=(5, 20))
    for i in range(num):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        input_slice = sample["input"][0]
        target_slice = sample["target"][0]
        axes[i, 0].imshow(input_slice, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(target_slice, cmap="gray")
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig("sample.png", dpi=300)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_nc_dataset()
