import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import h5py
import torchvision

import torch
import torch.utils.data as data
import numpy as np
import h5py
import torchvision


class Dataset_SDE(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset_SDE, self).__init__()
        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        self.ms_crops = MS_crop(ms)
        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        self.lms_crops = LMS_crop(lms)
        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        self.pan_crops = PAN_crop(pan)

    def __getitem__(self, item):
        return self.ms_crops[item], self.lms_crops[item], self.pan_crops[item]

    def __len__(self):
        return len(self.ms_crops)
    
class Dataset(data.Dataset):
    def __init__(self, file_path, name):
        '''
        name:图片的名字，范围是[0,19]
        '''
        
        
        super(Dataset, self).__init__()

        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan



    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops

    def __len__(self):
        return 1

class ReducedDataset(data.Dataset):
    """预构造的 reduced 仿真数据
    文件由 prepare_reduced_data.py 生成，包含 lms/pan（Wald 降采样后）与 gt（原始低分辨率 ms）。
    """
    def __init__(self, file_path, name):
        super(ReducedDataset, self).__init__()

        dataset = h5py.File(file_path, 'r')

        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0
        gt = np.array(dataset['gt'][name], dtype=np.float32) / 2047.0

        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()

        self.lms_crops = lms
        self.pan_crops = pan
        self.gt_crops = gt

    def __getitem__(self, item):
        return self.lms_crops, self.pan_crops, self.gt_crops

    def __len__(self):
        return 1
