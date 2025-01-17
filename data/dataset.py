"""
This module contains the DatasetVTI class, which is a subclass of the PyTorch
Dataset class. It is used to load VTI files and extract slices from them.
"""

import cv2
import torch
import tqdm
import vtk
import numpy as np
from vtk.util import numpy_support
from joblib import Parallel, delayed
from torch.utils.data import Dataset


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
            :, :, idx * self.num_slices: (idx + 1) * self.num_slices
        ]
        input_chunk = np.transpose(input_chunk, (2, 0, 1))
        hmin, hmax = 100, 1500
        input_chunk = np.clip(input_chunk, hmin, hmax)
        input_chunk = 2.0 * ((input_chunk - hmin) / (hmax - hmin) - 0.5)
        output_vti = read_vti(dp / self.output_filename)
        output_chunk = output_vti[
            :, :, idx * self.num_slices: (idx + 1) * self.num_slices
        ]
        output_chunk = np.transpose(output_chunk, (2, 0, 1))
        assert (
            input_chunk.shape == output_chunk.shape
        ), f"Expected {input_chunk.shape} but got {output_chunk.shape}"
        if self.resize is not None:
            input_chunk = cv2.resize(
                input_chunk[0], (self.resize, self.resize)
            )
            input_chunk = np.expand_dims(input_chunk, axis=0)
            output_chunk = cv2.resize(
                output_chunk[0], (self.resize, self.resize)
            )
            output_chunk = np.expand_dims(output_chunk, axis=0)
        if self.transform is not None:
            input_chunk = self.transform(input_chunk)
            output_chunk = self.transform(output_chunk)
        return input_chunk, output_chunk

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.total_num_chunks
        for idp, (dp, nc) in enumerate(
            zip(self.data_paths, self.num_chunks_per_file)
        ):
            if idx < nc:
                break
            idx -= nc

        input_chunk, output_chunk = self._process_chunk(dp, idx)
        return {
            "input": torch.tensor(input_chunk, dtype=torch.float),
            "target": torch.tensor(output_chunk, dtype=torch.float),
        }
