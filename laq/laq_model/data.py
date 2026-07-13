from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()

        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.image_size = image_size

        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames

    def _sorted_frame_list(self, folder):
        """List only image frames in a scene folder, sorted by the digits in their filename.

        Filtering by extension matters once non-image siblings exist next to
        frames (e.g. cached *.heatmap.npy files from data/prepare_nuscenes_heatmaps.py) —
        without it, os.listdir would interleave them with the actual frames.
        """
        files = [f for f in os.listdir(os.path.join(self.folder, folder)) if f.lower().endswith(IMAGE_EXTENSIONS)]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))))

    def _sample_frame_pair_paths(self, index):
        folder = self.folder_list[index]
        img_list = self._sorted_frame_list(folder)

        first_frame_idx = random.randint(0, len(img_list) - 1)
        second_frame_idx = min(first_frame_idx + self.offset, len(img_list) - 1)

        first_path = os.path.join(self.folder, folder, img_list[first_frame_idx])
        second_path = os.path.join(self.folder, folder, img_list[second_frame_idx])
        return first_path, second_path

    def _load_and_transform(self, path):
        img = Image.open(path)
        return self.transform(img).unsqueeze(1)

    def __getitem__(self, index):
        try:
            first_path, second_path = self._sample_frame_pair_paths(index)

            transform_img = self._load_and_transform(first_path)
            next_transform_img = self._load_and_transform(second_path)

            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


class HeatmapVideoDataset(ImageVideoDataset):
    """ImageVideoDataset that also returns precomputed per-frame saliency heatmaps.

    Expects each frame `frameNNNN.jpg` to have a sibling `frameNNNN.heatmap.npy`
    written by data/prepare_nuscenes_heatmaps.py, at `patch_grid` resolution.
    A frame missing its cache falls back to an all-zero heatmap, which is an
    exact no-op for laq_model.heatmap_conditioning.apply_patch_heatmap — so
    partially-cached datasets degrade gracefully instead of crashing.

    __getitem__ returns (video, heatmap) where video is the same (c, 2, h, w)
    tensor as the base class and heatmap is (2, patch_h, patch_w).
    """

    def __init__(self, folder, image_size, offset=5, patch_grid=(8, 8)):
        super().__init__(folder, image_size, offset=offset)
        self.patch_grid = patch_grid

    @staticmethod
    def _npy_cache_path(frame_path, suffix):
        """frameNNNN.jpg -> frameNNNN<suffix>, e.g. suffix='.heatmap.npy'."""
        base, _ = os.path.splitext(frame_path)
        return base + suffix

    def _load_npy(self, frame_path, suffix, shape):
        cache_path = self._npy_cache_path(frame_path, suffix)
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
        else:
            arr = np.zeros(shape, dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, h, w)

    def _load_heatmap(self, frame_path):
        return self._load_npy(frame_path, '.heatmap.npy', self.patch_grid)

    def __getitem__(self, index):
        try:
            first_path, second_path = self._sample_frame_pair_paths(index)

            video = torch.cat(
                [self._load_and_transform(first_path), self._load_and_transform(second_path)], dim=1
            )
            heatmap = torch.cat(
                [self._load_heatmap(first_path), self._load_heatmap(second_path)], dim=0
            )  # (2, patch_h, patch_w)

            return video, heatmap
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))

class EgoMotionVideoDataset(ImageVideoDataset):
    """ImageVideoDataset that loads precomputed ego-motion saliency heatmaps.

    Expects each frame `frameNNNN.jpg` to have a sibling `frameNNNN.egomotion.npy`
    written by data/prepare_nuscenes_egomotion.py, at `patch_grid` resolution.
    The cache stores the optical-flow magnitude between frame N and frame N+1,
    normalised to [0, 1] — capturing where the ego camera's motion is most
    salient rather than where other agents are.

    For the last frame of a scene (no successor) and any missing cache, the
    heatmap falls back to all-zeros, which is an exact no-op for
    laq_model.heatmap_conditioning.apply_patch_heatmap.

    __getitem__ returns (video, heatmap) where:
      video   : (c, 2, h, w) tensor — same as ImageVideoDataset
      heatmap : (2, patch_h, patch_w) — the first frame's egomotion map
                broadcast to both temporal slots, since the flow represents
                the transition and is relevant to both frames.
    """

    def __init__(self, folder, image_size, offset=5, patch_grid=(8, 8)):
        super().__init__(folder, image_size, offset=offset)
        self.patch_grid = patch_grid

    def _load_egomotion(self, frame_path: str) -> torch.Tensor:
        base, _ = os.path.splitext(frame_path)
        cache_path = base + '.egomotion.npy'
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            if arr.ndim == 2:
                # Old single-channel format (mag only). Pad direction channels
                # with zeros so the gate degrades gracefully to magnitude-only.
                # Rerun prepare_nuscenes_egomotion.py --overwrite to upgrade.
                arr = np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=0)
        else:
            arr = np.zeros((3,) + self.patch_grid, dtype=np.float32)
        return torch.from_numpy(arr)  # (3, patch_h, patch_w)

    def __getitem__(self, index):
        try:
            first_path, second_path = self._sample_frame_pair_paths(index)

            video = torch.cat(
                [self._load_and_transform(first_path), self._load_and_transform(second_path)], dim=1
            )
            # Use the first frame's egomotion map for both temporal slots.
            egomotion = self._load_egomotion(first_path)  # (3, patch_h, patch_w)
            heatmap = egomotion.unsqueeze(0).expand(2, -1, -1, -1).clone()  # (2, 3, patch_h, patch_w)

            return video, heatmap
        except:
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


'''
class CostmapVideoDataset(HeatmapVideoDataset):
    """HeatmapVideoDataset that also returns precomputed BEV cost maps.

    Expects each frame `frameNNNN.jpg` to have a sibling `frameNNNN.costmap.npy`
    written by data/prepare_nuscenes_costmaps.py, at `costmap_grid` resolution.
    A frame missing its cache falls back to an all-zero cost map, which is an
    exact no-op for laq_model.costmap_loss (a zero cost map has zero cosine
    similarity with every other sample, so it contributes no pull).

    __getitem__ returns (video, heatmap, costmap), where costmap is
    (2, costmap_h, costmap_w).
    """

    def __init__(self, folder, image_size, offset=5, patch_grid=(8, 8), costmap_grid=(32, 32)):
        super().__init__(folder, image_size, offset=offset, patch_grid=patch_grid)
        self.costmap_grid = costmap_grid

    def _load_costmap(self, frame_path):
        return self._load_npy(frame_path, '.costmap.npy', self.costmap_grid)

    def __getitem__(self, index):
        try:
            first_path, second_path = self._sample_frame_pair_paths(index)

            video = torch.cat(
                [self._load_and_transform(first_path), self._load_and_transform(second_path)], dim=1
            )
            heatmap = torch.cat(
                [self._load_heatmap(first_path), self._load_heatmap(second_path)], dim=0
            )
            costmap = torch.cat(
                [self._load_costmap(first_path), self._load_costmap(second_path)], dim=0
            )

            return video, heatmap, costmap
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
'''