"""
This module contains the DatasetVTI class, which is a subclass of the PyTorch
Dataset class. It is used to load VTI files and extract slices from them.
"""
import torch
import tqdm
import vtk
import numpy as np
from torch.utils.data import Dataset
from vtk.util import numpy_support


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
        augmentation=None,
        input_filename="xray.vti",
        output_filename="contours.vti",
    ):
        self.num_slices = num_slices
        self.transform = augmentation
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.data_paths = data_paths

        self.num_chunks_per_file = []
        for dp in tqdm.tqdm(self.data_paths, desc="Parsing VTI file"):
            input_vti = read_vti(dp / input_filename)
            output_vti = read_vti(dp / output_filename)
            if input_vti.shape != output_vti.shape:
                raise RuntimeError(
                    f"Got different input/targets shape, {input_vti.shape} != {output_vti.shape}"
                )
            self.num_chunks_per_file.append(
                input_vti.shape[2] // self.num_slices
            )
        self.total_num_chunks = sum(self.num_chunks_per_file)

    def __len__(self):
        return self.total_num_chunks

    def __getitem__(self, idx):
        for idp, (dp, nc) in enumerate(
            zip(self.data_paths, self.num_chunks_per_file)
        ):
            if idx < nc:
                break
            idx -= nc
        input_vti = read_vti(dp / self.input_filename)
        input_chunk = input_vti[
            :, :, (idx) * self.num_slices: (idx + 1) * self.num_slices
        ]
        input_chunk = np.transpose(input_chunk, (2, 0, 1))
        hmin, hmax = 100, 1500
        input_chunk = np.clip(input_chunk, hmin, hmax)
        input_chunk = 2.0 * ((input_chunk - hmin) / (hmax - hmin) - 0.5)
        if self.transform is not None:
            input_chunk = self.transform(input_chunk)
        output_vti = read_vti(dp / self.output_filename)
        output_chunk = output_vti[
            :, :, idx * self.num_slices: (idx + 1) * self.num_slices
        ]
        output_chunk = np.transpose(output_chunk, (2, 0, 1))

        return {
            "input": torch.tensor(input_chunk, dtype=torch.float),
            "target": torch.tensor(output_chunk, dtype=torch.float),
        }
