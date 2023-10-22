import pathlib
from typing import Callable, Tuple

import h5py
import torch
from torch.utils.data import Dataset

from PIL import Image
import os, shutil


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # outputpath = './experiments/datasets/'
        # facePath = preparePath(os.path.join(outputpath, f'{self.person_id_str}/Face/'))
        # leftEyePath = preparePath(os.path.join(outputpath, f'{self.person_id_str}/imEyeL/'))
        # rightEyePath = preparePath(os.path.join(outputpath, f'{self.person_id_str}/imEyeR/'))
        with h5py.File(self.dataset_path, 'r') as f:
            # if index<3000:
            imEyeL = f.get(f'{self.person_id_str}/imEyeL/{index:04}')[()]
            imEyeR = f.get(f'{self.person_id_str}/imEyeR/{index:04}')[()]
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            pose = f.get(f'{self.person_id_str}/pose/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]

            # imEyeL = f.get(f'{self.person_id_str}/imEyeL/{index+3000:04}')[()]
            # imEyeR = f.get(f'{self.person_id_str}/imEyeR/{index+3000:04}')[()]
            # image = f.get(f'{self.person_id_str}/image/{index+3000:04}')[()]
            # pose = f.get(f'{self.person_id_str}/pose/{index+3000:04}')[()]
            # gaze = f.get(f'{self.person_id_str}/gaze/{index+3000:04}')[()]

            # print(f'{self.person_id_str}', '%05d' % index, gaze, pose)
            # Image.fromarray(image).save(facePath + '%05dimage.jpg' % index, quality=95)
            # Image.fromarray(imEyeR).save(rightEyePath+'%05dL.jpg' % index, quality=95)
            # Image.fromarray(imEyeL).save(leftEyePath+'%05d R.jpg' % index, quality=95)

        imEyeL = self.transform(imEyeL)
        # print(f'{self.person_id_str}'+'%05d' % index, 'imEyeL shape:', imEyeL.shape)
        imEyeR = self.transform(imEyeR)
        # print(f'{self.person_id_str}'+'%05d' % index, 'imEyeR shape:', imEyeR.shape)
        image = self.transform(image)
        # print(f'{self.person_id_str}'+'%05d' % index, 'image shape:', image.shape)
        pose = torch.from_numpy(pose)
        # print(f'{self.person_id_str}'+'%05d' % index, 'pose shape', pose.shape)
        gaze = torch.from_numpy(gaze)
        # print(f'{self.person_id_str}'+'%05d' % index, 'gaze shape', gaze.shape)
        return image, pose, gaze, imEyeL, imEyeR

    def __len__(self) -> int:
        return 3000

def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path

