import pathlib
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import torch
import imageio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os 


from rfd.utils import get_data_path

def load_image(path: Path) -> np.ndarray:
    return imageio.imread(path).astype('float32') / 255.0


class PredictFutureFrameDataset(Dataset):
    def __init__(self, root_dir:Path, predict_n_steps: int, transform: A.Compose  =None) -> None:
        super().__init__()

        assert predict_n_steps  >=0
        assert transform is None or isinstance(transform, A.Compose)

        self.predict_n_steps = predict_n_steps
        self.image_tuples = []

        self.to_tensor_transform = ToTensorV2()
        self.augmentation_transform = transform

        # inspired by https://github.com/neuroethology/BKinD/blob/main/dataloader/custom_dataset.py 
        # list all episodes in the root dir
        # list all frames within those episodes.
        # add the tuples to the list of tuples
        for episode in sorted(os.listdir(root_dir)):
            episode_path = root_dir / episode
            episode_frames = sorted(os.listdir(episode_path))
            episode_frames = [frame for frame in episode_frames if frame.split('.')[1] in ["jpg", "png"]]
            for index in range(len(episode_frames) - self.predict_n_steps):
                img_t_path = episode_path/ episode_frames[index]
                img_tn_path = episode_path / episode_frames[index + self.predict_n_steps]

                self.image_tuples.append((img_t_path, img_tn_path))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t_path, img_tn_path = self.image_tuples[index]
        img_t, img_tn = load_image(img_t_path), load_image(img_tn_path)
        
        # do transforms

        if self.augmentation_transform is not None:
            img_t = self.augmentation_transform(image=img_t)["image"]
            img_tn = self.augmentation_transform(image=img_tn)["image"]            

        img_t = self.to_tensor_transform(image=img_t)["image"]
        img_tn = self.to_tensor_transform(image=img_tn)["image"]
        return img_t, img_tn

    def __len__(self):
        return len(self.image_tuples)