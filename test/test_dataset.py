import unittest
from pathlib import Path

import albumentations as A
import torch

from rfd.dataset.demonstration_dataset import PredictFutureFrameDataset


class TestPredictFutureDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = Path(__file__).parent / "test_data"

    def test_shapes_and_types(self):
        dataset = PredictFutureFrameDataset(self.dataset_path, 0)

        self.assertEqual(len(dataset), 10 + 19)

        imga, imgb = dataset[2]
        self.assertEqual(imga.shape, (3, 128, 128))
        self.assertEqual(imga.dtype, torch.float)

        self.assertTrue(torch.equal(imga, imgb))

    def test_augmentation(self):
        aug = A.Compose([A.RandomBrightness()])
        dataset = PredictFutureFrameDataset(self.dataset_path, 3, aug)
        self.assertEqual(len(dataset), 10 - 3 + 19 - 3)
        imga, _ = dataset[2]
        self.assertEqual(imga.shape, (3, 128, 128))
        self.assertEqual(imga.dtype, torch.float)
