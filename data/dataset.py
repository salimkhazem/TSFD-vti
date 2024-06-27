import os
import sys 
import vtk 
import tqdm 
import torch 
import pathlib 
import numpy as np
from vtk.util import numpy_support 
from torch.utils.data import Dataset 


def vtkToNumpy(data): 
    """
    Transform vtk data to numpy 

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
    def __init__(self, root_dir: str, num_slices: int=1, augmentation=None, input_filename="xray.vti", output_filename="contours.vti"): 
        self.num_slices = num_slices 
        self.transform = augmentation 
        self.input_filename = input_filename 
        self.output_filename = output_filename 
        self.data_paths = [] 
        for subdir, dirs, files in os.walk(root_dir): 
            for diri in dirs: 
                datapath = pathlib.Path(os.path.join(subdir, diri)) 
                input_filepath = datapath / input_filename 
                output_filepath = datapath / output_filename 
                if input_filepath.exists() and output_filepath.exists():
                    self.data_paths.append(datapath) 
        
        self.num_chunks_per_file = [] 
        self.input_vtis = [] 
        self.output_vtis = [] 
        for dp in tqdm.tqdm(self.data_paths, desc="Parsing VTI file"): 
            input_vti = read_vti(dp / input_filename) 
            output_vti = read_vti(dp / output_filename) 
            if input_vti.shape != output_vti.shape: 
                raise RuntimeError(f"Got different input/targets shape, {input_vti.shape} != {output_vti.shape}") 
            self.num_chunks_per_file.append(input_vti.shape[2] // num_slices) 
        self.total_num_chunks = sum(self.num_chunks_per_file) 
    
    def __len__(self): 
        return self.total_num_chunks 


    def __getitem__(self, idx): 
        for idp, (dp, nc) in enumerate(zip(self.data_paths, self.num_chunks_per_file)): 
            if idx < nc:
                break
            idx -= nc 
        input_vti = read_vti(dp / self.input_filename) 
        input_chunk = input_vti[:, :, (idx) * self.num_slices : (idx+1) * self.num_slices] 
        input_chunk = np.transpose(input_chunk, (2, 0, 1)) 
        hmin, hmax = 100, 1500
        input_chunk = np.clip(input_chunk, hmin, hmax)
        input_chunk = 2.0 * ((input_chunk - hmin) / (hmax - hmin) - 0.5)
        if self.transform is not None: 
            input_chunk = self.transform(input_chunk) 
        output_vti = read_vti(dp / self.output_filename) 
        output_chunk = output_vti[:, :, idx * self.num_slices: (idx+1) * self.num_slices] 
        output_chunk = np.transpose(output_chunk, (2, 0, 1)) 
        
        return {
                "input": torch.tensor(input_chunk, dtype=torch.float), 
                "target": torch.tensor(output_chunk, dtype=torch.float) 
            }


if __name__ == "__main__": 
    root_dir = sys.argv(1) if len(sys.argv) > 1 else "/mnt/WoodSeer/Slicing" 
    dataset = DatasetVTI(root_dir=root_dir, input_filename="xray.vti", output_filename="mes_0_255_0.vti") 
    print(f"Input: {dataset[0]['input'].shape}\t Target: {dataset[0]['target'].shape}") 
    print(f"Len Dataset: {len(dataset)}")  
