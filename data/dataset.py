"""
This module contains the DatasetVTI class, which is a subclass of the PyTorch
Dataset class. It is used to load VTI files and extract slices from them.
"""

import os
import cv2
import torch
import tqdm
import vtk
import numpy as np
import tempfile
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
        augmentation (callable): Optional augmentation function to be applied to the input data.
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

        # Create a temporary directory to store the cached data 
        tmp_dir = "./input"
        os.makedirs(tmp_dir, exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir="./input")

        # Process and cache data
        self.cache_data()

    def cache_data(self):
        self.cached_files = []
        # Parallel processing of data paths
        results = Parallel(n_jobs=-1)(
            delayed(self.process_path)(
                dp, self.input_filename, self.output_filename
            )
            for dp in tqdm.tqdm(self.data_paths, desc="Processing and caching data")
        )
        for result in results:
            self.cached_files.extend(result)

    def process_path(self, dp, input_filename, output_filename):
        cached_chunks = []
        input_vti = read_vti(os.path.join(dp, input_filename))
        output_vti = read_vti(os.path.join(dp, output_filename))
        for idx in range(input_vti.shape[2] // self.num_slices):
            input_chunk, output_chunk = self._process_chunk(input_vti, output_vti, idx)
            # Fix path concatenation
            base_name = os.path.basename(dp).replace('.vti', '')  # Assuming '.vti' in dp, adjust if necessary
            input_cache = os.path.join(self.temp_dir.name, f"input_{base_name}_{idx}.npy")
            output_cache = os.path.join(self.temp_dir.name, f"output_{base_name}_{idx}.npy")
            np.save(input_cache, input_chunk)
            np.save(output_cache, output_chunk)
            cached_chunks.append((input_cache, output_cache))
        return cached_chunks

    def _process_chunk(self, input_vti, output_vti, idx):
        input_chunk = input_vti[:, :, idx * self.num_slices: (idx + 1) * self.num_slices]
        input_chunk = np.transpose(input_chunk, (2, 0, 1))
        output_chunk = output_vti[:, :, idx * self.num_slices: (idx + 1) * self.num_slices]
        output_chunk = np.transpose(output_chunk, (2, 0, 1))
        if self.resize is not None:
            input_chunk = cv2.resize(input_chunk[0], (self.resize, self.resize))
            input_chunk = np.expand_dims(input_chunk, axis=0)
            output_chunk = cv2.resize(output_chunk[0], (self.resize, self.resize))
            output_chunk = np.expand_dims(output_chunk, axis=0)
        return input_chunk, output_chunk

    def __getitem__(self, idx):
        input_path, output_path = self.cached_files[idx]
        input_chunk = np.load(input_path)
        output_chunk = np.load(output_path)
        if self.transform:
            input_chunk = self.transform(input_chunk)
            output_chunk = self.transform(output_chunk)
        return {
            "input": torch.tensor(input_chunk, dtype=torch.float),
            "target": torch.tensor(output_chunk, dtype=torch.float),
        }

    def __len__(self):
        return len(self.cached_files)

    def __del__(self):
        self.temp_dir.cleanup()  # Clean up the temporary directory

